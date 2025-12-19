"""Submodules for AMS-Net v2."""

from .dks import DynamicKernelBlock
from .mspa import MultiScaleSpectralPyramid, MultiScaleSpectralPyramidAttention
from .faa import FallAwareAttention
from .tfcl import TimeFreqContrastiveLoss
from .spectral import MultiScaleSTFTBlock, WaveletSpectralBlock
from .efficient import GhostConv1d, SeparableConv1d, channel_shuffle
from .attention import (
    ASPP1d,
    CBAM1d,
    CoordinateAttention1d,
    ECA1d,
    EMA1d,
    MCA1d,
    SimAM1d,
    build_attention,
)

__all__ = [
    "DynamicKernelBlock",
    "MultiScaleSpectralPyramid",
    "MultiScaleSpectralPyramidAttention",
    "FallAwareAttention",
    "TimeFreqContrastiveLoss",
    "MultiScaleSTFTBlock",
    "WaveletSpectralBlock",
    "GhostConv1d",
    "SeparableConv1d",
    "channel_shuffle",
    "ECA1d",
    "CBAM1d",
    "EMA1d",
    "CoordinateAttention1d",
    "SimAM1d",
    "MCA1d",
    "ASPP1d",
    "build_attention",
]
