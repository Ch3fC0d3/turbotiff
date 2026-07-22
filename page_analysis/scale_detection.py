from __future__ import annotations
import numpy as np
from .models import ScaleDefinition
def classify_scale(vertical_lines,values=()):
    x=np.sort(np.asarray(vertical_lines,float)); spacing=np.diff(x); confidence={}; scale="unknown"; cycles=None
    if len(spacing)>=3:
        variation=float(np.std(spacing)/max(1e-6,np.mean(spacing)))
        if variation<.18: scale="linear"; confidence={"grid":max(0.,1-variation)}
        else: scale="logarithmic"; cycles=max(1,int(round(len(spacing)/9))); confidence={"grid":min(1.,variation)}
    values=list(values); minimum=min(values) if values else None; maximum=max(values) if values else None
    direction="increasing_right" if len(values)<2 or values[-1]>=values[0] else "increasing_left"
    return ScaleDefinition(scale,minimum,maximum,cycles,direction,confidence=confidence)
