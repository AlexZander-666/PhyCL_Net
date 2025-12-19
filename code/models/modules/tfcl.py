from typing import Dict, Tuple, Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F


class TimeFreqContrastiveLoss(nn.Module):
    """
    Time-frequency contrastive loss aligning time and frequency embeddings.
    Uses symmetric InfoNCE between modalities and an optional supervised contrastive term.
    """

    def __init__(self, temperature: float = 0.1, supervised_weight: float = 0.0):
        super().__init__()
        self.temperature = temperature
        self.supervised_weight = supervised_weight

    def forward(
        self,
        z_time: torch.Tensor,
        z_freq: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        if z_time.dim() != 2 or z_freq.dim() != 2:
            raise ValueError("TimeFreqContrastiveLoss expects 2D embeddings.")
        if z_time.shape != z_freq.shape:
            raise ValueError("z_time and z_freq must share shape.")

        z_t = F.normalize(z_time, dim=1)
        z_f = F.normalize(z_freq, dim=1)

        logits = torch.matmul(z_t, z_f.t()) / self.temperature
        targets = torch.arange(z_t.size(0), device=z_t.device)
        loss_tf = 0.5 * (F.cross_entropy(logits, targets) + F.cross_entropy(logits.t(), targets))

        loss_sup = torch.tensor(0.0, device=z_t.device, dtype=z_t.dtype)
        if labels is not None and self.supervised_weight > 0:
            loss_sup = self._supervised_contrastive(torch.cat([z_t, z_f], dim=0), labels.repeat(2))

        total_loss = loss_tf + self.supervised_weight * loss_sup
        stats = {"tfcl": float(loss_tf.detach().cpu()), "tfcl_sup": float(loss_sup.detach().cpu())}
        return total_loss, stats

    def _supervised_contrastive(self, feats: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        sim = torch.matmul(feats, feats.t()) / self.temperature
        logits_mask = torch.ones_like(sim) - torch.eye(sim.size(0), device=sim.device)
        sim = sim - 1e9 * torch.eye(sim.size(0), device=sim.device)

        label_mask = labels.view(-1, 1) == labels.view(1, -1)
        positives = torch.exp(sim) * label_mask * logits_mask
        negatives = torch.exp(sim) * logits_mask

        pos_sum = positives.sum(dim=1)
        neg_sum = negatives.sum(dim=1)
        denom = (pos_sum + neg_sum + 1e-12)
        log_prob = torch.log((pos_sum + 1e-12) / denom)
        return -(log_prob.mean())


class HierarchicalTFContrastiveLoss(nn.Module):
    """
    Hierarchical time-frequency contrastive loss:
    - Aligns time/freq embeddings at each stage (symmetric InfoNCE).
    - Adds cross-layer alignment (time_i vs freq_{i+1}).
    - Optional supervised contrastive term on final fused embeddings.
    """

    def __init__(self, temperature: float = 0.1, cross_layer_weight: float = 0.3, supervised_weight: float = 0.1):
        super().__init__()
        self.temp = temperature
        self.cross_weight = cross_layer_weight
        self.sup_weight = supervised_weight

    def _flatten_if_needed(self, z: torch.Tensor) -> torch.Tensor:
        # Accept (B, C, L) or (B, D)
        if z.dim() > 2:
            return z.mean(dim=tuple(range(2, z.dim())))
        return z

    def _infonce(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)
        logits = torch.mm(z1, z2.T) / self.temp
        labels = torch.arange(z1.size(0), device=z1.device)
        return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))

    def _supervised_contrastive(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        z = F.normalize(z, dim=-1)
        sim = torch.mm(z, z.T) / self.temp
        labels = labels.view(-1, 1)
        mask = (labels == labels.T).float()
        mask.fill_diagonal_(0)

        exp_sim = torch.exp(sim)
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-6)
        mean_log_prob = (mask * log_prob).sum(dim=1) / (mask.sum(dim=1) + 1e-6)
        return -mean_log_prob.mean()

    def forward(
        self,
        z_time_list: List[torch.Tensor],
        z_freq_list: List[torch.Tensor],
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        if len(z_time_list) == 0 or len(z_freq_list) == 0:
            raise ValueError("HierarchicalTFContrastiveLoss expects non-empty feature lists.")

        total_loss = 0.0
        n_layers = min(len(z_time_list), len(z_freq_list))
        stats: Dict[str, float] = {}

        for i in range(n_layers):
            z_t = self._flatten_if_needed(z_time_list[i])
            z_f = self._flatten_if_needed(z_freq_list[i])
            loss_same = self._infonce(z_t, z_f)
            total_loss = total_loss + loss_same
            stats[f"tfcl_layer{i}"] = float(loss_same.detach().cpu())

            if i + 1 < n_layers:
                z_f_next = self._flatten_if_needed(z_freq_list[i + 1])
                if z_t.size(-1) != z_f_next.size(-1):
                    proj = nn.Linear(z_t.size(-1), z_f_next.size(-1)).to(z_t.device)
                    z_t = proj(z_t)
                loss_cross = self._infonce(z_t, z_f_next)
                total_loss = total_loss + self.cross_weight * loss_cross
                stats[f"tfcl_cross{i}->{i+1}"] = float(loss_cross.detach().cpu())

        if labels is not None and n_layers > 0 and self.sup_weight > 0:
            z_final = torch.cat(
                [self._flatten_if_needed(z_time_list[-1]), self._flatten_if_needed(z_freq_list[-1])],
                dim=-1,
            )
            loss_sup = self._supervised_contrastive(z_final, labels)
            total_loss = total_loss + self.sup_weight * loss_sup
            stats["tfcl_sup"] = float(loss_sup.detach().cpu())

        total_loss = total_loss / float(n_layers)
        stats["tfcl_total"] = float(total_loss.detach().cpu())
        return total_loss, stats
