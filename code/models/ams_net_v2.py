import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from .modules.dks import DynamicKernelBlock
from .modules.faa import FallAwareAttention
from .modules.mspa import MultiScaleSpectralPyramid, MultiScaleSpectralPyramidAttention
from .modules.spectral import MultiScaleSTFTBlock, WaveletSpectralBlock
from .modules.efficient import GhostConv1d, SeparableConv1d, channel_shuffle
from .modules.attention import build_attention


def init_weights(m):
    if isinstance(m, (nn.Conv1d, nn.Linear)):
        nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
        if m.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(m.bias, -bound, bound)
    elif isinstance(m, (nn.BatchNorm1d, nn.GroupNorm)):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)


class CrossGatedFusion(nn.Module):
    """
    Cross-gated fusion that leverages one branch to gate the other.
    The enhanced variant adds multi-scale channel gates and a spatial refinement gate.
    """

    def __init__(
        self,
        dim: int,
        reduction: int = 4,
        variant: str = "enhanced",
        kernel_sizes: Sequence[int] = (3, 5, 7),
        spatial_kernel: int = 7,
    ):
        super().__init__()
        self.variant = (variant or "enhanced").lower()
        hidden = max(1, dim // reduction)
        spatial_kernel = spatial_kernel if spatial_kernel % 2 == 1 else spatial_kernel + 1

        if self.variant not in ("enhanced", "baseline"):
            self.variant = "baseline"

        if self.variant == "enhanced":
            self.time_gates = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.AdaptiveAvgPool1d(1),
                        nn.Conv1d(dim, hidden, 1),
                        nn.GELU(),
                        nn.Conv1d(hidden, dim, 1),
                    )
                    for _ in kernel_sizes
                ]
            )
            self.freq_gates = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.AdaptiveAvgPool1d(1),
                        nn.Conv1d(dim, hidden, 1),
                        nn.GELU(),
                        nn.Conv1d(hidden, dim, 1),
                    )
                    for _ in kernel_sizes
                ]
            )
            self.spatial_attn = nn.Conv1d(2, 1, kernel_size=spatial_kernel, padding=(spatial_kernel - 1) // 2)
        else:
            self.channel_fc_t = nn.Sequential(
                nn.AdaptiveAvgPool1d(1),
                nn.Conv1d(dim, hidden, 1),
                nn.GELU(),
                nn.Conv1d(hidden, dim, 1),
                nn.Sigmoid(),
            )
            self.channel_fc_f = nn.Sequential(
                nn.AdaptiveAvgPool1d(1),
                nn.Conv1d(dim, hidden, 1),
                nn.GELU(),
                nn.Conv1d(hidden, dim, 1),
                nn.Sigmoid(),
            )
            self.spatial_attn = None

        self.out_conv = nn.Conv1d(dim, dim, 1)
        self.norm = nn.GroupNorm(1, dim)
        self.res_scale = nn.Parameter(torch.tensor(0.0))

    def forward(self, x_time: torch.Tensor, x_freq: torch.Tensor) -> torch.Tensor:
        if x_time.shape != x_freq.shape:
            raise ValueError("CrossGatedFusion expects matching shapes.")

        if self.variant == "enhanced":
            t_gate = torch.stack([g(x_freq) for g in self.time_gates], dim=0).mean(dim=0)
            f_gate = torch.stack([g(x_time) for g in self.freq_gates], dim=0).mean(dim=0)
            t_gate = torch.sigmoid(t_gate)
            f_gate = torch.sigmoid(f_gate)
            fused = x_time * t_gate + x_freq * f_gate

            avg_pool = torch.mean(fused, dim=1, keepdim=True)
            max_pool = torch.max(fused, dim=1, keepdim=True).values
            spatial = torch.sigmoid(self.spatial_attn(torch.cat([avg_pool, max_pool], dim=1)))
            fused = fused * spatial
        else:
            g_t = self.channel_fc_t(x_freq)
            g_f = self.channel_fc_f(x_time)
            fused = x_time * g_t + x_freq * g_f

        fused = self.norm(fused)
        return x_time + self.res_scale * self.out_conv(fused)


def _build_freq_branch(
    channels: int,
    method: str,
    adaptive_bands: bool = True,
    band_edges: Optional[Sequence[float]] = None,
    num_bands: int = 4,
) -> nn.Module:
    m = (method or "fft").lower()
    if m == "stft":
        return MultiScaleSTFTBlock(channels)
    if m in ("cwt", "wavelet"):
        return WaveletSpectralBlock(channels)
    if m in ("fft_attn", "legacy_fft"):
        return MultiScaleSpectralPyramidAttention(channels)
    return MultiScaleSpectralPyramid(
        channels,
        num_bands=num_bands,
        band_edges=band_edges,
        fall_aware=True,
        adaptive_bands=adaptive_bands,
    )


def _make_attention(attn_cfg: Any, channels: int) -> Optional[nn.Module]:
    if attn_cfg is None:
        return None
    if isinstance(attn_cfg, nn.Module):
        return attn_cfg
    if isinstance(attn_cfg, dict):
        name = attn_cfg.get("type") or attn_cfg.get("name")
        if not name:
            return None
        kwargs = {k: v for k, v in attn_cfg.items() if k not in ("type", "name")}
        return build_attention(name, channels, **kwargs)
    return build_attention(attn_cfg, channels)


class AMSBlockV2(nn.Module):
    """
    Improved AMS block combining dynamic kernels, multi-scale spectral modeling,
    fall-aware attention, and cross-gated fusion.
    """

    def __init__(
        self,
        channels: int,
        use_mspa: bool = True,
        use_dks: bool = True,
        use_faa: bool = True,
        freq_method: str = "fft",
        sample_rate: float = 50.0,
        time_attn: Any = None,
        freq_attn: Any = None,
        fusion_attn: Any = None,
        fusion_variant: str = "enhanced",
        fusion_kernel_sizes: Sequence[int] = (3, 5, 7),
        adaptive_bands: bool = True,
        band_edges: Optional[Sequence[float]] = None,
        num_bands: int = 4,
        faa_axis_attn: bool = True,
    ):
        super().__init__()
        self.time_branch = DynamicKernelBlock(channels, sample_rate=sample_rate) if use_dks else nn.Sequential(
            nn.Conv1d(channels, channels, 3, padding=1, groups=channels),
            nn.Conv1d(channels, channels, 1),
        )
        self.freq_branch = _build_freq_branch(
            channels,
            freq_method,
            adaptive_bands=adaptive_bands,
            band_edges=band_edges,
            num_bands=num_bands,
        ) if use_mspa else nn.Identity()
        self.faa = FallAwareAttention(channels, use_axis_attention=faa_axis_attn) if use_faa else nn.Identity()
        self.fusion = CrossGatedFusion(
            channels,
            variant=fusion_variant,
            kernel_sizes=fusion_kernel_sizes,
        )
        self.res_scale = nn.Parameter(torch.tensor(0.0))
        self.time_attn = _make_attention(time_attn, channels)
        self.freq_attn = _make_attention(freq_attn, channels)
        self.fusion_attn = _make_attention(fusion_attn, channels)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        t_feat = self.time_branch(x)
        if self.time_attn is not None:
            t_feat = self.time_attn(t_feat)

        f_feat = self.freq_branch(x)
        f_feat = self.faa(f_feat)
        if self.freq_attn is not None:
            f_feat = self.freq_attn(f_feat)

        fused = self.fusion(t_feat, f_feat)
        if self.fusion_attn is not None:
            fused = self.fusion_attn(fused)
        out = x + self.res_scale * fused
        return out, t_feat, f_feat


class DownsampleBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = SeparableConv1d(in_channels, out_channels, kernel_size=5, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        if out.size(1) % 2 == 0:
            return channel_shuffle(out, groups=2)
        return out


class AMSNetV2(nn.Module):
    """
    AMS-Net v2: multi-scale time-frequency network with lightweight stem, AMS blocks,
    projection head for contrastive learning, and classifier head.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 2,
        proj_dim: int = 128,
        ablation: Optional[Dict[str, bool]] = None,
        freq_method: str = "fft",
        sample_rate: float = 50.0,
        time_attn: Any = None,
        freq_attn: Any = None,
        fusion_attn: Any = None,
        fusion_variant: str = "enhanced",
        fusion_kernel_sizes: Sequence[int] = (3, 5, 7),
        adaptive_bands: bool = True,
        band_edges: Optional[Sequence[float]] = None,
        num_bands: int = 4,
        faa_axis_attn: bool = True,
    ):
        super().__init__()
        ablation = ablation or {"mspa": True, "dks": True, "faa": True}
        stem_channels = 48
        self.stem = GhostConv1d(in_channels, stem_channels, kernel_size=5)

        self.stage1 = nn.ModuleList(
            [
                AMSBlockV2(
                    stem_channels,
                    ablation.get("mspa", True),
                    ablation.get("dks", True),
                    ablation.get("faa", True),
                    freq_method,
                    sample_rate,
                    time_attn,
                    freq_attn,
                    fusion_attn,
                    fusion_variant,
                    fusion_kernel_sizes,
                    adaptive_bands,
                    band_edges,
                    num_bands,
                    faa_axis_attn,
                )
                for _ in range(2)
            ]
        )
        self.transition1 = DownsampleBlock(stem_channels, stem_channels * 2)

        mid_channels = stem_channels * 2
        self.stage2 = nn.ModuleList(
            [
                AMSBlockV2(
                    mid_channels,
                    ablation.get("mspa", True),
                    ablation.get("dks", True),
                    ablation.get("faa", True),
                    freq_method,
                    sample_rate,
                    time_attn,
                    freq_attn,
                    fusion_attn,
                    fusion_variant,
                    fusion_kernel_sizes,
                    adaptive_bands,
                    band_edges,
                    num_bands,
                    faa_axis_attn,
                )
                for _ in range(2)
            ]
        )
        self.transition2 = DownsampleBlock(mid_channels, mid_channels * 2)

        high_channels = mid_channels * 2
        self.stage3 = nn.ModuleList(
            [
                AMSBlockV2(
                    high_channels,
                    ablation.get("mspa", True),
                    ablation.get("dks", True),
                    ablation.get("faa", True),
                    freq_method,
                    sample_rate,
                    time_attn,
                    freq_attn,
                    fusion_attn,
                    fusion_variant,
                    fusion_kernel_sizes,
                    adaptive_bands,
                    band_edges,
                    num_bands,
                    faa_axis_attn,
                )
                for _ in range(2)
            ]
        )

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(high_channels, num_classes)
        # Per-stage projection heads for hierarchical TF contrastive
        stage_channels = (
            [stem_channels] * len(self.stage1)
            + [mid_channels] * len(self.stage2)
            + [high_channels] * len(self.stage3)
        )
        self.tf_proj_time = nn.ModuleList(
            [
                nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(ch, proj_dim))
                for ch in stage_channels
            ]
        )
        self.tf_proj_freq = nn.ModuleList(
            [
                nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(ch, proj_dim))
                for ch in stage_channels
            ]
        )
        self.apply(init_weights)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.stem(x)

        t_feats: List[torch.Tensor] = []
        f_feats: List[torch.Tensor] = []
        proj_idx = 0

        for block in self.stage1:
            x, t_feat, f_feat = block(x)
            t_feats.append(t_feat)
            f_feats.append(f_feat)
            proj_idx += 1

        x = self.transition1(x)
        for block in self.stage2:
            x, t_feat, f_feat = block(x)
            t_feats.append(t_feat)
            f_feats.append(f_feat)
            proj_idx += 1

        x = self.transition2(x)
        for block in self.stage3:
            x, t_feat, f_feat = block(x)
            t_feats.append(t_feat)
            f_feats.append(f_feat)
            proj_idx += 1

        logits = self.classifier(self.pool(x).squeeze(-1))
        # Build hierarchical embeddings
        z_time_list: List[torch.Tensor] = []
        z_freq_list: List[torch.Tensor] = []
        for idx, (t, f) in enumerate(zip(t_feats, f_feats)):
            z_time_list.append(self.tf_proj_time[idx](t))
            z_freq_list.append(self.tf_proj_freq[idx](f))
        return logits, z_time_list, z_freq_list
