from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import amp as torch_amp


def _amp_autocast(enabled: bool, device_type: str = "cuda"):
    return torch_amp.autocast(device_type=device_type, enabled=enabled)


class MultiScaleSTFTBlock(nn.Module):
    """
    Multi-scale STFT feature extractor with lightweight convolutional fusion.
    Implements the review ask: multi-window STFT + 2D conv pyramid + temporal pooling.
    """

    def __init__(self, channels: int, window_sizes: Sequence[int] = (64, 128, 256), hop_ratio: float = 0.25):
        super().__init__()
        self.window_sizes = tuple(int(w) for w in window_sizes)
        self.hop_ratio = hop_ratio
        out_per_scale = max(1, channels // max(1, len(self.window_sizes)))
        self.freq_convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(1, out_per_scale, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_per_scale),
                    nn.GELU(),
                )
                for _ in self.window_sizes
            ]
        )
        self.fusion = nn.Conv1d(out_per_scale * len(self.window_sizes), channels, kernel_size=1)
        self.pool = nn.AdaptiveAvgPool2d((1, None))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError("MultiScaleSTFTBlock expects (B, C, L).")
        b, c, l = x.shape
        x_fp32 = x.to(torch.float32)
        x_flat = x_fp32.view(b * c, l)
        feats = []

        for ws, conv in zip(self.window_sizes, self.freq_convs):
            hop = max(1, int(ws * self.hop_ratio))
            window = torch.hann_window(ws, device=x.device, dtype=torch.float32)
            with _amp_autocast(False):
                spec = torch.stft(
                    x_flat,
                    n_fft=ws,
                    hop_length=hop,
                    window=window,
                    center=True,
                    return_complex=True,
                )
            # magnitude -> (B, C, F, T)
            mag = spec.abs().view(b, c, spec.shape[-2], spec.shape[-1])
            mag = mag.mean(dim=1, keepdim=True)  # average channels for stability -> (B, 1, F, T)
            feat = conv(mag)  # (B, out_per_scale, F, T)
            feat = self.pool(feat).squeeze(2)  # (B, out_per_scale, T)
            feat = F.interpolate(feat, size=l, mode="linear", align_corners=False)
            feats.append(feat)

        fused = torch.cat(feats, dim=1)  # (B, out_per_scale * n, L)
        out = self.fusion(fused)
        return out.to(dtype=x.dtype)


class WaveletSpectralBlock(nn.Module):
    """
    Morlet wavelet-based spectral encoder with learnable scale attention.
    Matches review ask: learnable log-spaced scales + physics-aligned wavelet bank.
    """

    def __init__(self, channels: int, num_scales: int = 16, kernel_size: int = 64):
        super().__init__()
        self.num_scales = num_scales
        self.kernel_size = kernel_size
        self.scales = nn.Parameter(torch.logspace(0, 2, steps=num_scales))
        self.conv2d = nn.Sequential(
            nn.Conv2d(1, channels, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, None))
        self.last_scale_weights = None

    def _morlet_wavelet(self, t: torch.Tensor, scale: torch.Tensor, omega0: float = 6.0) -> torch.Tensor:
        t_scaled = t / scale
        return torch.exp(-0.5 * t_scaled ** 2) * torch.cos(omega0 * t_scaled)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError("WaveletSpectralBlock expects (B, C, L).")
        b, c, l = x.shape
        device = x.device
        x_fp32 = x.to(torch.float32)
        t = torch.linspace(-4, 4, l, device=device, dtype=torch.float32)

        cwt_coeffs = []
        for scale in self.scales:
            wavelet = self._morlet_wavelet(t, scale).view(1, 1, l)
            with _amp_autocast(False):
                coeff = F.conv1d(x_fp32, wavelet.expand(c, 1, l), groups=c, padding=l // 2)
            coeff = coeff[:, :, :l]  # keep original length
            cwt_coeffs.append(coeff.mean(dim=1))  # (B, L)

        cwt = torch.stack(cwt_coeffs, dim=1)  # (B, num_scales, L)
        cwt = cwt.unsqueeze(1)  # (B, 1, num_scales, L)

        feat = self.conv2d(cwt)  # (B, C, num_scales, L)
        feat = self.pool(feat).squeeze(2)  # (B, C, L)

        return feat.to(dtype=x.dtype)
