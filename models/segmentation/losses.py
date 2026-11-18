"""Numerically stable binary segmentation losses."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def dice_loss(logits: torch.Tensor, targets: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    """Soft Dice loss averaged per image; empty/empty has zero loss."""
    if logits.shape != targets.shape:
        raise ValueError("logits and targets must have identical shapes")
    if smooth <= 0:
        raise ValueError("smooth must be positive")
    probabilities = torch.sigmoid(logits)
    dimensions = tuple(range(1, logits.ndim))
    intersection = (probabilities * targets).sum(dim=dimensions)
    denominator = probabilities.sum(dim=dimensions) + targets.sum(dim=dimensions)
    return (1 - (2 * intersection + smooth) / (denominator + smooth)).mean()


class BCEDiceLoss(nn.Module):
    """Weighted sum of BCE-with-logits and soft Dice loss."""

    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5) -> None:
        super().__init__()
        if bce_weight < 0 or dice_weight < 0 or bce_weight + dice_weight <= 0:
            raise ValueError("loss weights must be non-negative and not both zero")
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.bce_weight * F.binary_cross_entropy_with_logits(logits, targets) + self.dice_weight * dice_loss(logits, targets)
