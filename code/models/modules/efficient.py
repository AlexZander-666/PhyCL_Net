import math
from typing import Optional

import torch
import torch.nn as nn


class GhostConv1d(nn.Module):
    """
    Ghost module for 1D signals using a small primary conv and cheap depthwise expansion.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: Optional[int] = None,
        ratio: int = 2,
        dw_kernel_size: int = 3,
    ):
        super().__init__()
        self.out_channels = out_channels
        init_channels = math.ceil(out_channels / ratio)
        new_channels = out_channels - init_channels
        padding = (kernel_size - 1) // 2 if padding is None else padding

        self.primary_conv = nn.Sequential(
            nn.Conv1d(in_channels, init_channels, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm1d(init_channels),
            nn.GELU(),
        )
        groups = math.gcd(init_channels, new_channels) if new_channels > 0 else 1
        self.cheap_operation = nn.Sequential(
            nn.Conv1d(
                init_channels,
                new_channels,
                dw_kernel_size,
                stride=1,
                padding=(dw_kernel_size - 1) // 2,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm1d(new_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_primary = self.primary_conv(x)
        x_cheap = self.cheap_operation(x_primary)
        out = torch.cat([x_primary, x_cheap], dim=1)
        return out[:, : self.out_channels, :]


class SeparableConv1d(nn.Module):
    """
    Depthwise separable convolution for 1D signals.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: Optional[int] = None,
        bias: bool = False,
    ):
        super().__init__()
        padding = (kernel_size - 1) // 2 if padding is None else padding
        self.depthwise = nn.Conv1d(
            in_channels,
            in_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
            bias=bias,
        )
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.bn = nn.BatchNorm1d(out_channels)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.act(x)


def channel_shuffle(x: torch.Tensor, groups: int) -> torch.Tensor:
    """
    Channel shuffle utility used in ShuffleNet-style blocks.
    """
    if groups <= 1:
        return x
    b, c, l = x.size()
    if c % groups != 0:
        raise ValueError(f"Channels ({c}) not divisible by groups ({groups}).")
    x = x.view(b, groups, c // groups, l)
    x = x.transpose(1, 2).contiguous()
    return x.view(b, c, l)
