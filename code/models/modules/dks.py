from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicKernelBlock(nn.Module):
    """
    Physics-aware dynamic kernel selection with biomechanical priors.
    """

    def __init__(
        self,
        channels: int,
        kernel_sizes: Sequence[int] = (7, 15, 31, 63),
        temp_init: float = 1.0,
        use_physics: bool = True,
        sample_rate: float = 50.0,
    ):
        super().__init__()
        self.kernel_sizes = list(kernel_sizes)
        self.use_physics = use_physics
        self.sample_rate = float(sample_rate)
        self.kernels = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(
                        channels,
                        channels,
                        k,
                        padding=(k - 1) // 2,
                        groups=channels,
                        bias=False,
                    ),
                    nn.BatchNorm1d(channels),
                )
                for k in self.kernel_sizes
            ]
        )
        self.physics_dim = 12  # expanded fall-aware cues
        hidden = max(64, channels // 2)
        self.complexity_net = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channels, hidden),
            nn.ReLU(),
            nn.Linear(hidden, len(self.kernel_sizes)),
        )
        # [Critical Fix] BatchNorm for physics features to avoid hard truncation
        # Physics features (impact_duration, spectral_centroid, etc.) have high dynamic range
        # Using BN instead of posinf=1.0 truncation preserves meaningful gradients
        self.physics_norm = nn.BatchNorm1d(self.physics_dim) if use_physics else None
        
        self.physics_encoder = (
            nn.Sequential(
                nn.Linear(self.physics_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 32),
                nn.ReLU(),
            )
            if use_physics
            else None
        )
        self.physics_gate = (
            nn.Sequential(
                nn.Linear(32, len(self.kernel_sizes)),
                nn.Sigmoid(),
            )
            if use_physics
            else None
        )
        self.temperature = nn.Parameter(torch.tensor(float(temp_init)))
        self.proj = nn.Sequential(
            nn.GroupNorm(1, channels),
            nn.GELU(),
            nn.Conv1d(channels, channels, kernel_size=1),
        )
        self.res_scale = nn.Parameter(torch.tensor(0.0))
        self.last_routing_weights = None

    def _compute_physics_features(self, x: torch.Tensor) -> torch.Tensor:
        b, c, l = x.shape
        if l < 2:
            return torch.zeros((b, self.physics_dim), device=x.device, dtype=x.dtype)
        svm = torch.norm(x, dim=1)  # (B, L)
        svm_max = svm.max(dim=-1).values
        svm_mean = svm.mean(dim=-1)
        svm_std = svm.std(dim=-1) + 1e-6

        jerk = torch.diff(x, dim=-1) * self.sample_rate
        jerk_abs = jerk.abs()
        jerk_max = jerk_abs.max(dim=-1).values.mean(dim=1)

        if jerk.shape[-1] > 1:
            jerk_rate = torch.diff(jerk, dim=-1) * self.sample_rate
            jerk_rate_abs = jerk_rate.abs()
            jerk_rate_max = jerk_rate_abs.max(dim=-1).values.mean(dim=1)
        else:
            jerk_rate_max = torch.zeros(b, device=x.device, dtype=x.dtype)

        sign_changes = ((x[:, :, :-1] * x[:, :, 1:]) < 0).float()
        zcr = sign_changes.sum(dim=-1).mean(dim=1) / float(l)

        threshold = svm_mean.unsqueeze(-1) + 2 * svm_std.unsqueeze(-1)
        impact_duration = (svm > threshold).float().sum(dim=-1) / float(l)
        hard_threshold = svm_mean.unsqueeze(-1) + 3 * svm_std.unsqueeze(-1)
        impact_ratio = (svm > hard_threshold).float().sum(dim=-1) / float(l)

        free_fall = (svm < (svm_mean.unsqueeze(-1) - svm_std.unsqueeze(-1))).float().sum(dim=-1) / float(l)

        second_half = svm[:, l // 2 :] if l >= 2 else svm
        post_stillness = torch.tanh(1.0 / (second_half.std(dim=-1) + 1e-6))

        # Fall phase progression: stillness -> impact -> stillness
        third = max(1, l // 3)
        pre_mean = svm[:, :third].mean(dim=-1)
        impact_peak = svm[:, third : 2 * third].max(dim=-1).values
        post_mean = svm[:, 2 * third :].mean(dim=-1) if l - 2 * third > 0 else svm[:, -third:].mean(dim=-1)
        fall_phase = torch.relu(impact_peak - pre_mean) + torch.relu(impact_peak - post_mean)
        fall_phase = fall_phase / (impact_peak.abs() + 1e-6)

        # Frequency-domain cues
        if torch.is_autocast_enabled():
            with torch.amp.autocast(device_type="cuda", enabled=False):
                x_fp32 = x.to(torch.float32)
                spectrum = torch.fft.rfft(x_fp32, dim=-1, norm="ortho")
        else:
            x_fp32 = x.to(torch.float32)
            spectrum = torch.fft.rfft(x_fp32, dim=-1, norm="ortho")
        power = (spectrum.abs() ** 2).sum(dim=1)  # (B, F)
        freqs = torch.linspace(0, 1, power.shape[-1], device=x.device, dtype=power.dtype)
        total_power = power.sum(dim=-1) + 1e-6
        spectral_centroid = (power * freqs).sum(dim=-1) / total_power
        high_mask = freqs > 0.4
        high_freq_ratio = (power[:, high_mask].sum(dim=-1)) / total_power

        features = torch.stack(
            [
                svm_max,
                svm_mean,
                jerk_max,
                jerk_rate_max,
                zcr,
                impact_duration,
                impact_ratio,
                post_stillness,
                free_fall,
                spectral_centroid,
                high_freq_ratio,
                fall_phase,
            ],
            dim=-1,
        )
        # [Critical Fix] Only handle NaN/Inf without hard truncation
        # posinf=1.0 would destroy high dynamic range features like impact_duration, spectral_centroid
        features = torch.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        return features

    def forward(self, x: torch.Tensor, return_weights: bool = False):
        kernel_feats = [conv(x) for conv in self.kernels]
        stacked = torch.stack(kernel_feats, dim=1)  # (B, K, C, L)

        complexity_logits = self.complexity_net(x)  # (B, K)
        if self.use_physics and self.physics_encoder is not None and self.physics_gate is not None:
            physics_feats = self._compute_physics_features(x)  # (B, 12)
            # Apply BatchNorm to normalize high dynamic range physics features
            physics_feats = self.physics_norm(physics_feats)
            physics_embed = self.physics_encoder(physics_feats)
            physics_weights = self.physics_gate(physics_embed)
        else:
            physics_weights = torch.ones_like(complexity_logits)

        combined = complexity_logits * physics_weights
        temp = torch.clamp(self.temperature, min=0.1)
        weights = F.softmax(combined / temp, dim=-1)
        weights_exp = weights.view(x.size(0), len(self.kernels), 1, 1)
        self.last_routing_weights = weights.detach()

        mixed = (stacked * weights_exp).sum(dim=1)
        out = self.proj(mixed)
        out = x + self.res_scale * out
        if return_weights:
            return out, weights
        return out
