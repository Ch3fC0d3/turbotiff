"""Shared-backbone, five-head network for geometry-aware curve detection."""

from __future__ import annotations

from .model import CurvePhase1UNet, require_torch, torch, nn


class CurvePhase2UNet(CurvePhase1UNet):
    """Phase 1-compatible U-Net backbone with geometric and grid heads."""

    model_version = "curve-phase2-unet-v2"
    model_format_version = 2
    outputs = ("stroke", "centerline", "distance", "direction", "grid")

    def __init__(self, in_channels: int = 3, base_channels: int = 16):
        super().__init__(in_channels=in_channels, base_channels=base_channels)
        base = int(base_channels)
        self.distance_head = nn.Conv2d(base, 1, 1)
        self.direction_head = nn.Conv2d(base, 2, 1)
        self.grid_head = nn.Conv2d(base, 1, 1)

    def shared_features(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        bottleneck = self.bottleneck(self.pool(e2))
        d2 = self._match(self.up2(bottleneck), e2)
        d2 = self.dec2(torch.cat((d2, e2), dim=1))
        d1 = self._match(self.up1(d2), e1)
        return self.dec1(torch.cat((d1, e1), dim=1))

    def forward(self, x):
        features = self.shared_features(x)
        raw_direction = self.direction_head(features)
        direction = torch.nn.functional.normalize(raw_direction, dim=1, eps=1e-6)
        return {
            "stroke_logits": self.stroke_head(features),
            "centerline_logits": self.centerline_head(features),
            "distance_field": torch.sigmoid(self.distance_head(features)),
            "direction": direction,
            "grid_logits": self.grid_head(features),
        }

    def configuration(self) -> dict:
        return {
            "in_channels": self.in_channels,
            "base_channels": self.base_channels,
            "model_version": self.model_version,
            "model_format_version": self.model_format_version,
            "phase": 2,
            "architecture": "small_unet_multitask",
            "outputs": list(self.outputs),
        }


def transfer_phase1_weights(model: CurvePhase2UNet, checkpoint: dict) -> dict:
    """Load every shape-compatible Phase 1 tensor and leave new heads initialized."""
    require_torch()
    source = checkpoint.get("state_dict") if isinstance(checkpoint, dict) else None
    if not isinstance(source, dict):
        raise ValueError("Phase 1 checkpoint must contain a state_dict")
    target = model.state_dict()
    compatible = {key: value for key, value in source.items() if key in target and target[key].shape == value.shape}
    if not compatible:
        raise ValueError("Phase 1 checkpoint has no compatible backbone tensors")
    missing, unexpected = model.load_state_dict(compatible, strict=False)
    return {
        "loaded_tensor_count": len(compatible),
        "missing_keys": list(missing),
        "unexpected_keys": list(unexpected),
        "source_model_version": checkpoint.get("model_version"),
    }

