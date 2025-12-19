from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ECA1d(nn.Module):
    """Efficient Channel Attention for 1D signals."""

    def __init__(self, channels: int, k_size: int = 3):
        super().__init__()
        k_size = k_size if k_size % 2 == 1 else k_size + 1
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.avg_pool(x).transpose(1, 2)  # (B, 1, C)
        y = self.conv(y).transpose(1, 2)      # (B, C, 1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)


class _ChannelAttention1d(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.mlp = nn.Sequential(
            nn.Conv1d(channels, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden, channels, kernel_size=1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)


class _SpatialAttention1d(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
        self.conv = nn.Conv1d(2, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(x_cat))


class CBAM1d(nn.Module):
    """Channel + spatial attention for 1D signals."""

    def __init__(self, channels: int, reduction: int = 16, spatial_kernel: int = 7):
        super().__init__()
        self.channel_attn = _ChannelAttention1d(channels, reduction=reduction)
        self.spatial_attn = _SpatialAttention1d(kernel_size=spatial_kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.channel_attn(x)
        x = x * self.spatial_attn(x)
        return x


class EMA1d(nn.Module):
    """Multi-scale spatial attention adapted to 1D."""

    def __init__(self, channels: int, dilations: tuple = (1, 2, 4)):
        super().__init__()
        self.branches = nn.ModuleList(
            [
                nn.Conv1d(channels, channels, kernel_size=3, padding=d * 1, dilation=d, groups=channels, bias=False)
                for d in dilations
            ]
        )
        self.merge = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.norm = nn.GroupNorm(1, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = [b(x) for b in self.branches]
        attn = torch.stack(feats, dim=0).sum(dim=0)
        attn = self.sigmoid(self.merge(attn))
        out = x * attn
        return self.norm(out)


class CoordinateAttention1d(nn.Module):
    """
    Position-aware channel attention for 1D signals.
    Splits into channel gate (global) and position gate (local along length).
    """

    def __init__(self, channels: int, reduction: int = 16, kernel_size: int = 7):
        super().__init__()
        hidden = max(1, channels // reduction)
        kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1

        self.channel_mlp = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden, channels, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )
        self.position_conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.pos_sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ch_gate = self.channel_mlp(x)
        pos_ctx = torch.mean(x, dim=1, keepdim=True)
        pos_gate = self.pos_sigmoid(self.position_conv(pos_ctx))
        return x * ch_gate * pos_gate


class SimAM1d(nn.Module):
    """Parameter-free attention from SimAM, adapted for 1D."""

    def __init__(self, channels: Optional[int] = None, e_lambda: float = 1e-4):
        super().__init__()
        self.e_lambda = e_lambda

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
        energy = (x - mean) ** 2 / (4 * (var + self.e_lambda)) + 0.5
        attn = torch.sigmoid(energy)
        return x * attn


class ASPP1d(nn.Module):
    """Atrous Spatial Pyramid Pooling for 1D."""

    def __init__(self, channels: int, out_channels: Optional[int] = None, dilations: tuple = (1, 6, 12, 18)):
        super().__init__()
        out_channels = out_channels or channels
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(channels, out_channels, kernel_size=1, bias=False),
                    nn.BatchNorm1d(out_channels),
                    nn.GELU(),
                )
            ]
        )
        for d in dilations:
            self.branches.append(
                nn.Sequential(
                    nn.Conv1d(
                        channels,
                        out_channels,
                        kernel_size=3,
                        padding=d,
                        dilation=d,
                        bias=False,
                    ),
                    nn.BatchNorm1d(out_channels),
                    nn.GELU(),
                )
            )
        self.proj = nn.Sequential(
            nn.Conv1d(out_channels * len(self.branches), channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = [b(x) for b in self.branches]
        fused = torch.cat(feats, dim=1)
        return self.proj(fused)


class MCA1d(nn.Module):
    """Lightweight multi-dimensional collaborative attention for 1D."""

    def __init__(self, channels: int, spatial_kernel: int = 7):
        super().__init__()
        spatial_kernel = spatial_kernel if spatial_kernel % 2 == 1 else spatial_kernel + 1
        self.alpha = nn.Parameter(torch.ones(1, channels, 1))
        self.beta = nn.Parameter(torch.ones(1, channels, 1))
        self.spatial_conv = nn.Conv1d(1, 1, kernel_size=spatial_kernel, padding=(spatial_kernel - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = torch.mean(x, dim=-1, keepdim=True)
        std = torch.sqrt(torch.var(x, dim=-1, keepdim=True, unbiased=False) + 1e-6)
        ch_gate = torch.sigmoid(self.alpha * avg + self.beta * std)

        pos_ctx = torch.mean(x, dim=1, keepdim=True)
        pos_gate = self.sigmoid(self.spatial_conv(pos_ctx))
        return x * ch_gate * pos_gate


ATTN_FACTORY = {
    "eca": ECA1d,
    "cbam": CBAM1d,
    "ema": EMA1d,
    "ca": CoordinateAttention1d,
    "simam": SimAM1d,
    "aspp": ASPP1d,
    "mca": MCA1d,
}


def build_attention(kind: Optional[str], channels: int, **kwargs: Any) -> Optional[nn.Module]:
    if kind is None:
        return None
    if isinstance(kind, str):
        name = kind.lower()
    else:
        name = str(kind).lower()
    if name in ("none", "null", "no"):
        return None
    if name not in ATTN_FACTORY:
        raise ValueError(f"Unknown attention type: {kind}")
    return ATTN_FACTORY[name](channels=channels, **kwargs)
