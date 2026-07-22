"""Prompt-optional Phase 4 lightweight model with row-level wrap evidence."""
from __future__ import annotations
from .phase2_model import CurvePhase2UNet
from .model import torch, nn

class CurvePhase4UNet(CurvePhase2UNet):
    model_version="curve-phase4-unet-v1"; model_format_version=4
    outputs=("stroke","centerline","distance","direction","grid","wrap")
    def __init__(self, in_channels=3, base_channels=16, prompt_conditioned=True):
        self.prompt_conditioned=bool(prompt_conditioned)
        super().__init__(in_channels=int(in_channels)+(1 if self.prompt_conditioned else 0), base_channels=base_channels)
        self.wrap_head=nn.Sequential(nn.AdaptiveAvgPool2d((None,1)),nn.Conv2d(base_channels,3,1))
    def forward(self,image,prompt_map=None,color_hint=None):
        if self.prompt_conditioned:
            if prompt_map is None: prompt_map=torch.zeros((image.shape[0],1,image.shape[2],image.shape[3]),device=image.device,dtype=image.dtype)
            image=torch.cat((image,prompt_map),dim=1)
        features=self.shared_features(image); raw=self.direction_head(features)
        return {"stroke_logits":self.stroke_head(features),"centerline_logits":self.centerline_head(features),
                "distance_field":torch.sigmoid(self.distance_head(features)),"direction":torch.nn.functional.normalize(raw,dim=1,eps=1e-6),
                "grid_logits":self.grid_head(features),"wrap_logits":self.wrap_head(features).squeeze(-1)}
    def configuration(self):
        result=super().configuration(); result.update(phase=4,architecture="small_unet_prompt_wrap",outputs=list(self.outputs),prompt_conditioned=self.prompt_conditioned); return result
