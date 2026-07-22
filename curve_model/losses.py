"""Losses for sparse curve-stroke and centerline segmentation."""

from __future__ import annotations

from dataclasses import dataclass

try:
    import torch
    from torch import nn
    import torch.nn.functional as F
except Exception as exc:  # pragma: no cover
    torch = None
    nn = None
    F = None
    TORCH_IMPORT_ERROR = exc
else:
    TORCH_IMPORT_ERROR = None


def soft_dice_loss(logits, targets, epsilon: float = 1e-6):
    probabilities = torch.sigmoid(logits)
    dims = tuple(range(1, probabilities.ndim))
    intersection = (probabilities * targets).sum(dim=dims)
    denominator = probabilities.sum(dim=dims) + targets.sum(dim=dims)
    return (1.0 - (2.0 * intersection + epsilon) / (denominator + epsilon)).mean()


def skeleton_recall_loss(stroke_logits, centerline_targets, epsilon: float = 1e-6):
    stroke_probability = torch.sigmoid(stroke_logits)
    dims = tuple(range(1, stroke_probability.ndim))
    covered = (stroke_probability * centerline_targets).sum(dim=dims)
    total = centerline_targets.sum(dim=dims)
    recall = (covered + epsilon) / (total + epsilon)
    return (1.0 - recall).mean()


@dataclass(frozen=True)
class LossWeights:
    stroke_bce: float = 1.0
    stroke_dice: float = 1.0
    centerline_bce: float = 1.25
    centerline_dice: float = 1.25
    skeleton_recall: float = 0.75
    positive_weight: float = 8.0


class CurveDetectionLoss(nn.Module if nn is not None else object):
    def __init__(self, weights: LossWeights | None = None):
        if torch is None or nn is None:
            raise RuntimeError(f"PyTorch is required for neural losses: {TORCH_IMPORT_ERROR}")
        super().__init__()
        self.weights = weights or LossWeights()

    def forward(self, outputs: dict, targets: dict) -> dict:
        stroke_logits = outputs["stroke_logits"]
        centerline_logits = outputs["centerline_logits"]
        stroke_target = targets["stroke_mask"].to(stroke_logits.dtype)
        centerline_target = targets["centerline_mask"].to(centerline_logits.dtype)
        pos_weight = torch.tensor(self.weights.positive_weight, device=stroke_logits.device, dtype=stroke_logits.dtype)
        stroke_bce = F.binary_cross_entropy_with_logits(stroke_logits, stroke_target, pos_weight=pos_weight)
        centerline_bce = F.binary_cross_entropy_with_logits(centerline_logits, centerline_target, pos_weight=pos_weight)
        stroke_dice = soft_dice_loss(stroke_logits, stroke_target)
        centerline_dice = soft_dice_loss(centerline_logits, centerline_target)
        recall = skeleton_recall_loss(stroke_logits, centerline_target)
        total = (
            self.weights.stroke_bce * stroke_bce
            + self.weights.stroke_dice * stroke_dice
            + self.weights.centerline_bce * centerline_bce
            + self.weights.centerline_dice * centerline_dice
            + self.weights.skeleton_recall * recall
        )
        return {
            "total": total,
            "stroke_bce": stroke_bce,
            "stroke_dice": stroke_dice,
            "centerline_bce": centerline_bce,
            "centerline_dice": centerline_dice,
            "skeleton_recall": recall,
        }
