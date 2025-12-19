import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import CoordinateAttention1d


class FallAwareAttention(nn.Module):
    """
    Physics-inspired fall-aware attention emphasizing multiple physical cues:
    - SVM (magnitude) for overall intensity
    - Jerk (first derivative) for impacts
    - Jerk rate (second derivative) to separate transient vs sustained
    
    Implements Noise-Robust Derivative Estimation via Gaussian smoothing
    prior to numerical differentiation to mitigate high-frequency noise
    amplification inherent in finite difference operations.
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        reduction: int = 4,
        use_axis_attention: bool = True,
        smooth_sigma: float = 0.5,
    ):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.jerk_conv = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            padding=padding,
            groups=channels,
            bias=False,
        )
        self.jerk_rate_conv = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            padding=padding,
            groups=channels,
            bias=False,
        )
        hidden = max(1, channels // reduction)
        self.context = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(hidden, channels, kernel_size=1),
        )
        self.gate = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.axis_attn = CoordinateAttention1d(channels, reduction=reduction, kernel_size=kernel_size) if use_axis_attention else None
        self.norm = nn.GroupNorm(1, channels)
        self.res_scale = nn.Parameter(torch.tensor(0.0))
        self.last_attention = None
        
        # Noise-Robust Derivative Estimation: Gaussian smoothing kernel
        # Mitigates high-frequency noise amplification in numerical differentiation
        smooth_kernel = self._create_gaussian_kernel(smooth_sigma)
        self.register_buffer("smooth_kernel", smooth_kernel, persistent=False)

    def _create_gaussian_kernel(self, sigma: float, kernel_size: int = 5) -> torch.Tensor:
        """Create 1D Gaussian smoothing kernel for noise-robust differentiation."""
        x = torch.arange(kernel_size, dtype=torch.float32) - (kernel_size - 1) / 2
        kernel = torch.exp(-0.5 * (x / max(sigma, 1e-6)) ** 2)
        kernel = kernel / kernel.sum()  # normalize
        return kernel.view(1, 1, kernel_size)

    def _smooth(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Gaussian smoothing to reduce noise before differentiation."""
        B, C, L = x.shape
        kernel = self.smooth_kernel.expand(C, 1, -1).to(x.device, x.dtype)
        padding = (kernel.shape[-1] - 1) // 2
        # Use reflect padding to avoid boundary artifacts
        x_padded = F.pad(x, (padding, padding), mode="reflect")
        return F.conv1d(x_padded, kernel, groups=C)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError("FallAwareAttention expects (B, C, L).")

        x_axis = self.axis_attn(x) if self.axis_attn is not None else x

        # Noise-Robust Derivative Estimation:
        # Apply Gaussian smoothing before differentiation to suppress
        # high-frequency noise amplification from finite differences
        x_smooth = self._smooth(x)

        # SVM (Signal Vector Magnitude) - overall intensity
        svm = torch.sqrt((x_smooth ** 2).sum(dim=1, keepdim=True) + 1e-8)
        
        # Jerk (first derivative) - impact detection
        jerk = x_smooth[..., 1:] - x_smooth[..., :-1]
        jerk = F.pad(jerk, (1, 0))  # causal padding
        
        # Jerk rate (second derivative) - transient vs sustained discrimination
        jerk_rate = jerk[..., 1:] - jerk[..., :-1]
        jerk_rate = F.pad(jerk_rate, (1, 0))

        local = self.jerk_conv(torch.abs(jerk)) + self.jerk_rate_conv(torch.abs(jerk_rate))
        context = self.context(x_axis) + self.context(svm.expand_as(x))
        attn = self.gate(local + context)
        self.last_attention = attn.detach()
        out = self.norm(x_axis * attn)
        return x + self.res_scale * out
