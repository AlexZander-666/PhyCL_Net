from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCELoss(nn.Module):
    """Standard InfoNCE loss using pairwise dot-product similarity."""

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, queries: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
        if queries.shape != keys.shape:
            raise ValueError("queries and keys must share shape for InfoNCE.")
        q = F.normalize(queries, dim=1)
        k = F.normalize(keys, dim=1)
        logits = torch.matmul(q, k.t()) / self.temperature
        targets = torch.arange(q.size(0), device=q.device)
        return F.cross_entropy(logits, targets)


class SupervisedContrastiveLoss(nn.Module):
    """Supervised contrastive loss encouraging same-class samples to cluster."""

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, feats: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if feats.dim() != 2:
            raise ValueError("SupervisedContrastiveLoss expects 2D features.")
        if feats.size(0) != labels.size(0):
            raise ValueError("Features and labels batch size mismatch.")

        feats = F.normalize(feats, dim=1)
        sim = torch.matmul(feats, feats.t()) / self.temperature
        logits_mask = torch.ones_like(sim) - torch.eye(sim.size(0), device=sim.device)
        sim = sim - 1e9 * torch.eye(sim.size(0), device=sim.device)

        label_mask = labels.view(-1, 1) == labels.view(1, -1)
        positives = torch.exp(sim) * label_mask * logits_mask
        negatives = torch.exp(sim) * logits_mask

        pos_sum = positives.sum(dim=1)
        neg_sum = negatives.sum(dim=1)
        denom = pos_sum + neg_sum + 1e-12
        log_prob = torch.log((pos_sum + 1e-12) / denom)
        return -(log_prob.mean())
