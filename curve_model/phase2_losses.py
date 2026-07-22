"""Configurable geometric, segmentation, grid, and connectivity losses."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from .losses import soft_dice_loss, skeleton_recall_loss, torch, nn, F, TORCH_IMPORT_ERROR


@dataclass(frozen=True)
class CapeConfig:
    enabled: bool = False
    start_epoch: int = 10
    weight: float = 0.1
    window_size: int = 64
    dilation_radius: int = 3


@dataclass(frozen=True)
class Phase2LossWeights:
    stroke_bce: float = 1.0
    stroke_dice: float = 1.0
    centerline_bce: float = 1.25
    centerline_dice: float = 1.25
    skeleton_recall: float = 0.75
    distance: float = 1.0
    direction: float = 0.5
    grid: float = 0.65
    positive_weight: float = 8.0
    grid_positive_weight: float = 4.0
    distance_base_weight: float = 0.2
    distance_center_bonus: float = 2.0
    direction_center_bonus: float = 2.0


def masked_direction_loss(prediction, target, valid_mask, centerline_target, center_bonus: float = 2.0):
    prediction = F.normalize(prediction, dim=1, eps=1e-6)
    target = F.normalize(target, dim=1, eps=1e-6)
    cosine = torch.sum(prediction * target, dim=1, keepdim=True).clamp(-1.0, 1.0)
    weights = valid_mask * (1.0 + float(center_bonus) * centerline_target)
    return ((1.0 - cosine) * weights).sum() / weights.sum().clamp_min(1.0)


def _masked_bce_with_logits(logits, target, validity, pos_weight):
    loss = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight, reduction="none")
    mask = validity.expand_as(loss)
    return (loss * mask).sum() / mask.sum().clamp_min(1.0)


def _masked_dice_loss(logits, target, validity, epsilon: float = 1e-6):
    probability = torch.sigmoid(logits)
    mask = validity.expand_as(probability)
    dims = tuple(range(1, probability.ndim))
    intersection = (probability * target * mask).sum(dim=dims)
    denominator = (probability * mask).sum(dim=dims) + (target * mask).sum(dim=dims)
    valid_samples = validity.reshape(validity.shape[0], -1).amax(dim=1) > 0
    losses = 1.0 - (2.0 * intersection + epsilon) / (denominator + epsilon)
    return losses[valid_samples].mean() if valid_samples.any() else logits.sum() * 0.0


def cape_connectivity_loss(centerline_logits, centerline_target, window_size: int = 64, dilation_radius: int = 3):
    """Differentiable local-path coverage surrogate used for optional CAPE fine-tuning."""
    probability = torch.sigmoid(centerline_logits)
    radius = max(1, int(dilation_radius))
    kernel = radius * 2 + 1
    local_support = F.max_pool2d(probability, kernel_size=(3, kernel), stride=1, padding=(1, radius))
    window_losses = []
    for start in range(0, centerline_target.shape[2], max(1, int(window_size))):
        end = min(centerline_target.shape[2], start + max(1, int(window_size)))
        target_window = centerline_target[:, :, start:end]
        support_window = local_support[:, :, start:end]
        probability_window = probability[:, :, start:end]
        target_rows = target_window.amax(dim=3, keepdim=True)
        covered = (support_window * target_window).sum(dim=(1, 2, 3))
        required = target_window.sum(dim=(1, 2, 3)).clamp_min(1.0)
        row_presence = (probability_window.amax(dim=3, keepdim=True) * target_rows).sum(dim=(1, 2, 3))
        row_count = target_rows.sum(dim=(1, 2, 3)).clamp_min(1.0)
        window_losses.append((1.0 - 0.7 * covered / required - 0.3 * row_presence / row_count).clamp_min(0.0))
    return torch.stack(window_losses, dim=0).mean()


class CurvePhase2Loss(nn.Module if nn is not None else object):
    def __init__(self, weights: Phase2LossWeights | None = None, cape: CapeConfig | None = None):
        if torch is None or nn is None:
            raise RuntimeError(f"PyTorch is required for Phase 2 losses: {TORCH_IMPORT_ERROR}")
        super().__init__()
        self.weights = weights or Phase2LossWeights()
        self.cape = cape or CapeConfig()

    def configuration(self) -> dict:
        return {"weights": asdict(self.weights), "cape": asdict(self.cape)}

    def cape_active(self, epoch: int) -> bool:
        return bool(self.cape.enabled and int(epoch) >= int(self.cape.start_epoch))

    def forward(self, outputs: dict, targets: dict, epoch: int = 0) -> dict:
        weights = self.weights
        stroke = targets["stroke_mask"].to(outputs["stroke_logits"].dtype)
        center = targets["centerline_mask"].to(outputs["centerline_logits"].dtype)
        distance = targets["distance_field"].to(outputs["distance_field"].dtype)
        direction = targets["direction_field"].to(outputs["direction"].dtype)
        valid_direction = targets["valid_direction_mask"].to(outputs["direction"].dtype)
        grid = targets["grid_mask"].to(outputs["grid_logits"].dtype)
        stroke_valid = targets.get("stroke_label_valid")
        grid_valid = targets.get("grid_label_valid")
        if stroke_valid is None:
            stroke_valid = torch.ones((stroke.shape[0], 1, 1, 1), device=stroke.device, dtype=stroke.dtype)
        else:
            stroke_valid = stroke_valid.to(device=stroke.device, dtype=stroke.dtype)
        if grid_valid is None:
            grid_valid = torch.ones((grid.shape[0], 1, 1, 1), device=grid.device, dtype=grid.dtype)
        else:
            grid_valid = grid_valid.to(device=grid.device, dtype=grid.dtype)
        pos = torch.tensor(weights.positive_weight, device=stroke.device, dtype=stroke.dtype)
        grid_pos = torch.tensor(weights.grid_positive_weight, device=stroke.device, dtype=stroke.dtype)

        parts = {
            "stroke_bce": _masked_bce_with_logits(outputs["stroke_logits"], stroke, stroke_valid, pos),
            "stroke_dice": _masked_dice_loss(outputs["stroke_logits"], stroke, stroke_valid),
            "centerline_bce": F.binary_cross_entropy_with_logits(outputs["centerline_logits"], center, pos_weight=pos),
            "centerline_dice": soft_dice_loss(outputs["centerline_logits"], center),
            "skeleton_recall": skeleton_recall_loss(outputs["stroke_logits"], center),
            "grid": _masked_bce_with_logits(outputs["grid_logits"], grid, grid_valid, grid_pos),
        }
        distance_weights = float(weights.distance_base_weight) + float(weights.distance_center_bonus) * distance
        parts["distance"] = (
            F.smooth_l1_loss(outputs["distance_field"], distance, reduction="none") * distance_weights
        ).sum() / distance_weights.sum().clamp_min(1.0)
        parts["direction"] = masked_direction_loss(
            outputs["direction"], direction, valid_direction, center, weights.direction_center_bonus
        )
        parts["cape"] = cape_connectivity_loss(
            outputs["centerline_logits"], center, self.cape.window_size, self.cape.dilation_radius
        ) if self.cape_active(epoch) else outputs["centerline_logits"].sum() * 0.0
        parts["cape_active"] = self.cape_active(epoch)
        parts["total"] = (
            weights.stroke_bce * parts["stroke_bce"]
            + weights.stroke_dice * parts["stroke_dice"]
            + weights.centerline_bce * parts["centerline_bce"]
            + weights.centerline_dice * parts["centerline_dice"]
            + weights.skeleton_recall * parts["skeleton_recall"]
            + weights.distance * parts["distance"]
            + weights.direction * parts["direction"]
            + weights.grid * parts["grid"]
            + (self.cape.weight * parts["cape"] if parts["cape_active"] else 0.0)
        )
        return parts
