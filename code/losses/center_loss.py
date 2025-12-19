import torch
import torch.nn as nn


class CenterLoss(nn.Module):
    """
    Center loss with momentum-style updates for class centroids.
    The centers are stored as buffers and only updated inside forward()
    when the module is in training mode.
    """

    def __init__(self, num_classes: int, feat_dim: int, lambda_c: float = 1.0, alpha: float = 0.5):
        super().__init__()
        self.num_classes = num_classes
        self.lambda_c = lambda_c
        self.alpha = alpha
        self.register_buffer('centers', torch.randn(num_classes, feat_dim))

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if features.dim() != 2:
            raise ValueError("CenterLoss expects 2D feature embeddings.")
        if labels.dim() != 1:
            raise ValueError("CenterLoss expects 1D labels.")
        if features.size(0) != labels.size(0):
            raise ValueError("Features and labels must share the batch dimension.")

        labels = labels.to(dtype=torch.long)
        centers_batch = self.centers[labels]
        loss = (features - centers_batch).pow(2).sum() / (2.0 * features.size(0))

        if self.training:
            with torch.no_grad():
                unique_labels = torch.unique(labels)
                for label in unique_labels:
                    mask = labels == label
                    if mask.sum() == 0:
                        continue
                    delta = features[mask].mean(dim=0) - self.centers[label]
                    self.centers[label] += self.alpha * delta

        return self.lambda_c * loss
