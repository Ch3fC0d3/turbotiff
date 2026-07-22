"""Small two-head U-Net used by the Phase 1 curve detector."""

from __future__ import annotations

try:
    import torch
    from torch import nn
except Exception as exc:  # pragma: no cover - exercised in dependency-free deployments
    torch = None
    nn = None
    TORCH_IMPORT_ERROR = exc
else:
    TORCH_IMPORT_ERROR = None


def require_torch() -> None:
    if torch is None or nn is None:
        raise RuntimeError(f"PyTorch is required for the neural Phase 1 model: {TORCH_IMPORT_ERROR}")


class _ConvBlock(nn.Module if nn is not None else object):
    def __init__(self, in_channels: int, out_channels: int):
        require_torch()
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.layers(x)


class CurvePhase1UNet(nn.Module if nn is not None else object):
    """A compact fully convolutional RGB U-Net with stroke and centerline heads."""

    model_version = "curve-phase1-unet-v1"

    def __init__(self, in_channels: int = 3, base_channels: int = 16):
        require_torch()
        super().__init__()
        base = int(base_channels)
        self.in_channels = int(in_channels)
        self.base_channels = base
        self.enc1 = _ConvBlock(self.in_channels, base)
        self.enc2 = _ConvBlock(base, base * 2)
        self.bottleneck = _ConvBlock(base * 2, base * 4)
        self.pool = nn.MaxPool2d(2)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = _ConvBlock(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = _ConvBlock(base * 2, base)
        self.stroke_head = nn.Conv2d(base, 1, 1)
        self.centerline_head = nn.Conv2d(base, 1, 1)

    @staticmethod
    def _match(source, reference):
        if source.shape[-2:] == reference.shape[-2:]:
            return source
        return torch.nn.functional.interpolate(source, size=reference.shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        bottleneck = self.bottleneck(self.pool(e2))
        d2 = self._match(self.up2(bottleneck), e2)
        d2 = self.dec2(torch.cat((d2, e2), dim=1))
        d1 = self._match(self.up1(d2), e1)
        d1 = self.dec1(torch.cat((d1, e1), dim=1))
        return {
            "stroke_logits": self.stroke_head(d1),
            "centerline_logits": self.centerline_head(d1),
        }

    def configuration(self) -> dict:
        return {
            "in_channels": self.in_channels,
            "base_channels": self.base_channels,
            "model_version": self.model_version,
        }

