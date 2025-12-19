from typing import Optional, Sequence, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import amp as torch_amp


def _amp_autocast(enabled: bool, device_type: str = "cuda"):
    return torch_amp.autocast(device_type=device_type, enabled=enabled)


class MultiScaleSpectralPyramid(nn.Module):
    """
    Multi-scale spectral pyramid attention operating in the frequency domain.
    Splits the spectrum into sub-bands, enhances each band independently, and
    aggregates them with cross-band attention and positional encoding.
    """

    def __init__(
        self,
        dim: int,
        num_bands: int = 4,
        band_edges: Optional[Sequence[float]] = None,
        fall_aware: bool = True,
        adaptive_bands: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.num_bands = num_bands
        self.adaptive_bands = adaptive_bands
        if band_edges is None and fall_aware:
            # 50Hz 采样下的跌倒相关频带：0-2/2-8/8-20/>20Hz -> 归一化
            band_edges = (0.0, 0.04, 0.16, 0.4, 1.0)
        elif band_edges is None:
            band_edges = (0.0, 0.15, 0.35, 0.65, 1.0)
        if len(band_edges) != num_bands + 1:
            raise ValueError("band_edges length must be num_bands + 1.")
        init_edges = torch.tensor([float(b) for b in band_edges], dtype=torch.float32)
        init_edges = torch.clamp(init_edges, 0.0, 1.0)
        if self.adaptive_bands:
            init_deltas = torch.diff(init_edges)
            # Use softplus-parameterized deltas to ensure monotonic edges within [0,1]
            self.band_deltas = nn.Parameter(torch.log(torch.clamp(init_deltas, min=1e-3)))
        else:
            self.register_buffer("band_edges_tensor", init_edges, persistent=False)

        self.band_mlps = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(dim, dim, kernel_size=1),
                    nn.GELU(),
                    nn.Conv1d(dim, dim, kernel_size=3, padding=1, groups=dim),
                    nn.BatchNorm1d(dim),
                )
                for _ in range(num_bands)
            ]
        )
        self.band_attn = nn.Linear(num_bands, num_bands, bias=True)
        self.fuse = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(dim, dim, kernel_size=1),
        )
        self.out_norm = nn.GroupNorm(1, dim)
        self.last_band_weights = None

    def _get_band_edges(self, device, dtype) -> torch.Tensor:
        if self.adaptive_bands:
            deltas = F.softplus(self.band_deltas)
            deltas = deltas / (deltas.sum() + 1e-6)
            edges = torch.cumsum(torch.cat([torch.zeros(1, device=device, dtype=deltas.dtype), deltas], dim=0), dim=0)
            return torch.clamp(edges, 0.0, 1.0).to(dtype=dtype)
        return self.band_edges_tensor.to(device=device, dtype=dtype)

    def _calc_band_indices(self, freq_len: int, band_edges: torch.Tensor) -> List[Tuple[int, int]]:
        indices: List[Tuple[int, int]] = []
        edges_cpu = band_edges.detach().cpu()
        for i in range(self.num_bands):
            start = int(float(edges_cpu[i]) * (freq_len - 1))
            end = int(float(edges_cpu[i + 1]) * (freq_len - 1))
            start = min(start, freq_len - 1)
            end = max(start + 1, end)
            end = min(end, freq_len)
            indices.append((start, end))
        return indices

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError("MultiScaleSpectralPyramid expects (B, C, L).")

        with _amp_autocast(False):
            x_fp32 = x.to(torch.float32)
            X = torch.fft.rfft(x_fp32, dim=-1, norm="ortho")

        amp = torch.abs(X)
        phase = torch.angle(X)
        freq_len = amp.shape[-1]
        band_edges = self._get_band_edges(x.device, amp.dtype)
        freq_pos = torch.linspace(0, 1, steps=freq_len, device=x.device, dtype=amp.dtype).view(1, 1, freq_len)
        freq_pos_scale = 1.0 + 0.1 * freq_pos
        amp = amp * freq_pos_scale

        band_feats = []
        indices = self._calc_band_indices(freq_len, band_edges)
        for (s, e), mlp in zip(indices, self.band_mlps):
            band = amp[..., s:e]
            band_feats.append(mlp(band))

        expanded = []
        for feat in band_feats:
            if feat.shape[-1] != freq_len:
                feat = F.interpolate(feat, size=freq_len, mode="linear", align_corners=False)
            expanded.append(feat)

        band_stack = torch.stack(expanded, dim=1)  # (B, Bn, C, F)
        band_scores = band_stack.mean(dim=(2, 3))  # (B, Bn)
        weights = torch.softmax(self.band_attn(band_scores), dim=1).view(x.size(0), self.num_bands, 1, 1)
        self.last_band_weights = weights.detach()
        amp_enhanced = (band_stack * weights).sum(dim=1)
        amp_enhanced = self.fuse(amp_enhanced)
        amp_enhanced = F.relu(amp_enhanced) + 1e-8
        # Align dtype with phase for torch.polar
        amp_enhanced = amp_enhanced.to(dtype=phase.dtype)

        X_mod = torch.polar(amp_enhanced, phase)
        with _amp_autocast(False):
            x_out = torch.fft.irfft(X_mod, n=x.size(-1), dim=-1, norm="ortho")
        return self.out_norm(x_out).to(dtype=x.dtype)


class MultiScaleSpectralPyramidAttention(nn.Module):
    """
    Multi-scale spectral pyramid with learnable frequency attention.
    Uses multi-window STFTs to capture local time-frequency structure and
    aggregates them with band-wise attention before projecting back to 1D.
    """

    def __init__(self, channels: int, window_sizes=None):
        super().__init__()
        self.window_sizes = list(window_sizes or (64, 128, 256, 512))
        if len(self.window_sizes) == 0:
            raise ValueError("window_sizes must be a non-empty sequence.")
        per_scale = max(1, channels // len(self.window_sizes))
        out_channels = per_scale * len(self.window_sizes)

        self.stft_convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(1, per_scale, kernel_size=3, padding=1),
                    nn.BatchNorm2d(per_scale),
                    nn.GELU(),
                )
                for _ in self.window_sizes
            ]
        )

        attn_hidden = max(1, out_channels // 4)
        self.freq_attention = nn.Sequential(
            nn.Conv2d(out_channels, attn_hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(attn_hidden, out_channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.scale_fusion = nn.Conv1d(out_channels, channels, kernel_size=1)
        self.last_attention = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError("MultiScaleSpectralPyramidAttention expects (B, C, L).")
        B, C, L = x.shape
        scale_feats = []

        for ws, conv in zip(self.window_sizes, self.stft_convs):
            max_fft = max(2, 2 * max(L - 1, 1))
            n_fft = min(ws, max_fft)
            if n_fft % 2 == 1:
                n_fft += 1
            hop = max(1, n_fft // 4)
            win = torch.hann_window(n_fft, device=x.device, dtype=torch.float32)
            with _amp_autocast(False):
                spec = torch.stft(
                    x.view(B * C, L),
                    n_fft=n_fft,
                    hop_length=hop,
                    window=win,
                    return_complex=True,
                )
            # Numerical stability: add epsilon before abs() to handle edge cases
            # (e.g., all-zero inputs or extreme values that could cause NaN)
            mag = (spec.real ** 2 + spec.imag ** 2 + 1e-8).sqrt()
            mag = mag.view(B, C, spec.shape[-2], spec.shape[-1])
            mag = mag.mean(dim=1, keepdim=True)  # (B, 1, F, T)
            feat = conv(mag)  # (B, per_scale, F, T)
            scale_feats.append(feat)

        # Align scales to the largest freq/time resolution
        target_freq = max(f.shape[-2] for f in scale_feats)
        target_time = max(f.shape[-1] for f in scale_feats)
        scale_feats = [
            f if (f.shape[-2] == target_freq and f.shape[-1] == target_time)
            else F.interpolate(f, size=(target_freq, target_time), mode="bilinear", align_corners=False)
            for f in scale_feats
        ]

        multi_scale = torch.cat(scale_feats, dim=1)  # (B, out_channels, F, T)
        freq_ctx = multi_scale.mean(dim=3, keepdim=True)  # pool over time
        freq_attn = self.freq_attention(freq_ctx)
        freq_attn = torch.broadcast_to(freq_attn, multi_scale.shape)
        self.last_attention = freq_attn.detach()

        attended = multi_scale * freq_attn
        out = attended.mean(dim=2)  # average over frequency -> (B, out_channels, T)
        out = F.interpolate(out, size=L, mode="linear", align_corners=False)
        out = self.scale_fusion(out)
        return out
