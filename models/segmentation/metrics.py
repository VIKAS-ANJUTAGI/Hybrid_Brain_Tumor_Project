"""Binary-mask metrics with explicit empty-mask conventions."""

from __future__ import annotations

import torch


def segmentation_metrics(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> dict[str, float]:
    """Return batch-mean binary segmentation metrics after thresholding logits.

    For each image, empty prediction versus empty target scores 1.0; an empty
    prediction paired with a nonempty target (or conversely) scores 0.0.
    Pixel precision is defined as 1.0 when there are no predicted positives,
    and pixel recall is defined as 1.0 when there are no target positives.
    These conventions keep all returned metrics finite.
    """
    if logits.shape != targets.shape:
        raise ValueError("logits and targets must have identical shapes")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1")
    predicted = torch.sigmoid(logits) >= threshold
    target = targets > 0.5
    dimensions = tuple(range(1, logits.ndim))
    intersection = (predicted & target).sum(dim=dimensions, dtype=torch.float32)
    predicted_size = predicted.sum(dim=dimensions, dtype=torch.float32)
    target_size = target.sum(dim=dimensions, dtype=torch.float32)
    false_positive = (predicted & ~target).sum(dim=dimensions, dtype=torch.float32)
    false_negative = (~predicted & target).sum(dim=dimensions, dtype=torch.float32)
    union = predicted_size + target_size - intersection
    both_empty = (predicted_size == 0) & (target_size == 0)
    dice = torch.where(both_empty, torch.ones_like(intersection), 2 * intersection / (predicted_size + target_size).clamp_min(1))
    iou = torch.where(both_empty, torch.ones_like(intersection), intersection / union.clamp_min(1))
    precision_denominator = intersection + false_positive
    recall_denominator = intersection + false_negative
    precision = torch.where(
        precision_denominator == 0,
        torch.ones_like(intersection),
        intersection / precision_denominator,
    )
    recall = torch.where(
        recall_denominator == 0,
        torch.ones_like(intersection),
        intersection / recall_denominator,
    )
    return {
        "dice": float(dice.mean().item()),
        "iou": float(iou.mean().item()),
        "precision": float(precision.mean().item()),
        "recall": float(recall.mean().item()),
    }
