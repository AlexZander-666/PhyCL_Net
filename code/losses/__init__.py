from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from .contrastive import InfoNCELoss, SupervisedContrastiveLoss
from .center_loss import CenterLoss
from models.modules.tfcl import TimeFreqContrastiveLoss, HierarchicalTFContrastiveLoss

__all__ = [
    "InfoNCELoss",
    "SupervisedContrastiveLoss",
    "CenterLoss",
    "AMSNetLoss",
    "UncertaintyWeightedLoss",
]


class UncertaintyWeightedLoss(nn.Module):
    """
    Homoscedastic Uncertainty Weighting for Multi-Task Learning.
    
    Automatically learns task weights based on task-dependent uncertainty,
    eliminating the need for manual hyperparameter tuning of loss coefficients.
    
    Reference: Kendall et al., "Multi-Task Learning Using Uncertainty to Weigh
    Losses for Scene Geometry and Semantics", CVPR 2018.
    
    L_total = sum_i [ (1 / (2 * sigma_i^2)) * L_i + log(sigma_i) ]
    
    where sigma_i is the learned uncertainty for task i.
    """

    def __init__(self, num_tasks: int = 2, init_log_var: float = 0.0):
        super().__init__()
        # log(sigma^2) parameterization for numerical stability
        # init_log_var=0 corresponds to sigma=1, weight=0.5
        self.log_vars = nn.Parameter(torch.full((num_tasks,), init_log_var))

    def forward(self, *losses: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            *losses: Variable number of task losses (L_1, L_2, ..., L_n)
        
        Returns:
            total_loss: Uncertainty-weighted combined loss
            stats: Dictionary with per-task weights and weighted losses
        """
        if len(losses) != len(self.log_vars):
            raise ValueError(f"Expected {len(self.log_vars)} losses, got {len(losses)}")
        
        total = torch.tensor(0.0, device=losses[0].device, dtype=losses[0].dtype)
        stats: Dict[str, float] = {}
        
        for i, loss in enumerate(losses):
            # precision = 1 / sigma^2 = exp(-log_var)
            precision = torch.exp(-self.log_vars[i])
            # weighted_loss = (1/2) * precision * L + (1/2) * log_var
            # The (1/2) factor is absorbed into the formulation
            weighted = precision * loss + 0.5 * self.log_vars[i]
            total = total + weighted
            
            # Record stats for monitoring
            sigma = torch.exp(0.5 * self.log_vars[i])
            stats[f"task{i}_weight"] = float(precision.detach().cpu())
            stats[f"task{i}_sigma"] = float(sigma.detach().cpu())
            stats[f"task{i}_weighted"] = float(weighted.detach().cpu())
        
        stats["total_uncertainty"] = float(total.detach().cpu())
        return total, stats

    def get_weights(self) -> torch.Tensor:
        """Return current task weights (precisions)."""
        return torch.exp(-self.log_vars)


class AMSNetLoss(nn.Module):
    """
    Combined loss for AMS-Net v2.
    
    Supports two weighting modes:
    1. Fixed weights (default): L = L_ce + alpha * L_tfcl + beta * L_center
    2. Uncertainty weighting: Automatically learns task weights using homoscedastic
       uncertainty (Kendall et al., CVPR 2018), eliminating manual hyperparameter tuning.
    """

    def __init__(
        self,
        num_classes: int = 2,
        feat_dim: int = 128,
        alpha: float = 0.1,
        beta: float = 0.01,
        use_tfcl: bool = True,
        hierarchical_tfcl: bool = True,
        tf_cross_weight: float = 0.5,
        temperature: float = 0.1,
        supervised_weight: float = 0.0,
        label_smoothing: float = 0.0,
        class_weights: Optional[torch.Tensor] = None,
        use_uncertainty_weighting: bool = False,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.use_uncertainty_weighting = use_uncertainty_weighting
        self.ce = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
        self.hierarchical_tfcl = hierarchical_tfcl
        if use_tfcl:
            if hierarchical_tfcl:
                self.tfcl = HierarchicalTFContrastiveLoss(
                    temperature=temperature,
                    cross_layer_weight=tf_cross_weight,
                    supervised_weight=supervised_weight,
                )
            else:
                self.tfcl = TimeFreqContrastiveLoss(temperature=temperature, supervised_weight=supervised_weight)
        else:
            self.tfcl = None
        self.center = CenterLoss(num_classes, feat_dim) if beta > 0 else None
        
        # Automatic uncertainty weighting (Kendall et al., CVPR 2018)
        # num_tasks: CE, TFCL, Center
        self.uncertainty_weighter = UncertaintyWeightedLoss(num_tasks=3) if use_uncertainty_weighting else None

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        z_time: Optional[torch.Tensor] = None,
        z_freq: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        loss_ce = self.ce(logits, labels)
        loss_tfcl = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
        loss_center = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
        stats: Dict[str, float] = {"ce": float(loss_ce.detach().cpu())}

        if self.tfcl is not None and z_time is not None and z_freq is not None:
            if self.hierarchical_tfcl and isinstance(z_time, (list, tuple)) and isinstance(z_freq, (list, tuple)):
                tfcl_val, tfcl_stats = self.tfcl(list(z_time), list(z_freq), labels)
            else:
                tfcl_val, tfcl_stats = self.tfcl(z_time, z_freq, labels)
            loss_tfcl = tfcl_val
            stats.update(tfcl_stats)

        def _select_last(feat):
            if feat is None:
                return None
            if isinstance(feat, (list, tuple)):
                return feat[-1]
            return feat

        center_time = _select_last(z_time)
        center_freq = _select_last(z_freq)
        if self.center is not None and center_time is not None:
            center_feats = center_time if center_freq is None else 0.5 * (center_time + center_freq)
            loss_center = self.center(center_feats, labels)
            stats["center"] = float(loss_center.detach().cpu())

        # Combine losses
        if self.use_uncertainty_weighting and self.uncertainty_weighter is not None:
            # Automatic weighting via learned uncertainty
            total, uw_stats = self.uncertainty_weighter(loss_ce, loss_tfcl, loss_center)
            stats.update(uw_stats)
            # Record learned weights for monitoring
            weights = self.uncertainty_weighter.get_weights()
            stats["uw_ce"] = float(weights[0].detach().cpu())
            stats["uw_tfcl"] = float(weights[1].detach().cpu())
            stats["uw_center"] = float(weights[2].detach().cpu())
        else:
            # Fixed weighting (original behavior)
            total = loss_ce + self.alpha * loss_tfcl + self.beta * loss_center
        
        stats["total"] = float(total.detach().cpu())
        return total, stats
