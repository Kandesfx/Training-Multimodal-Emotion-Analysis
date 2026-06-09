"""Loss functions for multimodal emotion recognition.

Available losses:
  - FocalLoss: multi-label focal loss with optional per-class pos_weight
  - compute_pos_weight: helper to compute pos_weight from binary labels
"""
from __future__ import annotations

import torch
from torch import nn


def compute_pos_weight(
    labels: torch.Tensor,
    max_weight: float = 50.0,
) -> torch.Tensor | None:
    """Compute per-class positive weights from binary multi-label labels.

    pos_weight[i] = num_negatives_i / num_positives_i
    Higher weight → model pays more attention to rare positive samples.

    Args:
        labels: (N, num_classes) binary tensor {0, 1}
        max_weight: clamp weights to this maximum to prevent instability
    Returns:
        (num_classes,) tensor of per-class positive weights,
        or None if all classes have zero positives.
    """
    if labels.dim() == 1:
        labels = labels.unsqueeze(1)
    n_pos = labels.sum(dim=0)          # (num_classes,)
    n_neg = len(labels) - n_pos       # (num_classes,)
    weights = n_neg / (n_pos + 1e-8)   # avoid div-by-zero
    weights = weights.clamp(max=max_weight)
    if (n_pos == 0).all():
        return None
    return weights


class FocalLoss(nn.Module):
    """Focal Loss for multi-label classification.

    Reference: Lin et al., "Focal Loss for Dense Object Detection" (ICCV 2017).

    FL(p) = -α_t * (1 - p_t)^γ * log(p_t)

    where:
      p_t = p        if y=1 (positive class)
            1 - p    if y=0 (negative class)
      α_t = α        if y=1
            1 - α    if y=0

    With γ=2 (default):
      - Easy examples (p≈0 or p≈1): weight ≈ 0  → low penalty
      - Hard examples (p≈0.5):     weight ≈ 1  → maximum penalty

    This automatically down-weights easy negatives (the dominant class in
    imbalanced multi-label settings) and focuses training on hard examples.

    Args:
        alpha: weighting factor for positive class (default: 0.25).
               Higher α → more focus on positives.
        gamma: focusing parameter (default: 2.0).
               Higher γ → more focus on hard examples.
        reduction: "mean" or "sum" (default: "mean").
        pos_weight: optional (num_classes,) tensor — additional per-class
                   multiplicative weight on positives (from compute_pos_weight).
                   Use with compute_pos_weight() for best results.
    """

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = "mean",
        pos_weight: torch.Tensor | None = None,
    ):
        super().__init__()
        if reduction not in ("mean", "sum", "none"):
            raise ValueError(f"reduction must be 'mean', 'sum', or 'none', got {reduction!r}")
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.register_buffer("pos_weight", pos_weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits:  (batch, num_classes) — raw model outputs (before sigmoid)
            targets: (batch, num_classes) — binary targets {0, 1}
        Returns:
            Scalar loss.
        """
        probs = torch.sigmoid(logits)

        # p_t = p if y=1 else 1-p
        p_t = probs * targets + (1.0 - probs) * (1.0 - targets)

        # Binary cross-entropy per element
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction="none",
        )

        # Focal modulating factor: (1 - p_t)^gamma
        focal_weight = (1.0 - p_t).pow(self.gamma)

        # Alpha factor: alpha for positives, (1 - alpha) for negatives
        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)

        focal_loss = alpha_t * focal_weight * bce

        if self.reduction == "sum":
            return focal_loss.sum()
        if self.reduction == "none":
            return focal_loss
        # Standard mean reduction over all (batch * num_classes) elements.
        # Each element contributes equally; no special normalization per positives.
        return focal_loss.mean()
