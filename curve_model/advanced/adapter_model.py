"""Experimental high-resolution residual-adapter curve model (no pretrained download)."""
from __future__ import annotations
from ..phase4_model import CurvePhase4UNet
from ..model import nn

class CurveAdapterModel(CurvePhase4UNet):
    model_version="curve-adapter-v1"
    def __init__(self,in_channels=3,base_channels=24,prompt_conditioned=True):
        super().__init__(in_channels,base_channels,prompt_conditioned)
        self.adapter=nn.Sequential(nn.Conv2d(base_channels,base_channels,1),nn.GELU(),nn.Conv2d(base_channels,base_channels,1))
    def shared_features(self,x):
        features=super().shared_features(x); return features+self.adapter(features)
    def configuration(self):
        result=super().configuration(); result["architecture"]="curve_adapter_v1"; return result
    def resource_report(self):
        total=sum(p.numel() for p in self.parameters()); trainable=sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"parameter_count":total,"trainable_parameter_count":trainable,"model_size_bytes_fp32":total*4}
