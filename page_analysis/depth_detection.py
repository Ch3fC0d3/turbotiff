from __future__ import annotations
import numpy as np
from .models import DepthMapping

def optimize_depth_sequence(labels, increasing=True):
    ordered=sorted(labels,key=lambda item:item["row_position"]); selected=[]; previous=None
    for item in ordered:
        choices=[]
        for candidate in item.get("text_candidates",[]):
            try: value=float(str(candidate["text"]).replace(",",""))
            except ValueError: continue
            monotonic=previous is None or ((value>previous) if increasing else (value<previous))
            score=float(candidate.get("confidence",0))-(0 if monotonic else 2)
            choices.append((score,value,candidate))
        if choices:
            _,value,candidate=max(choices,key=lambda x:x[0]); selected.append({**item,"selected_value":value,"selected_candidate":candidate}); previous=value
    return selected

def fit_depth_mapping(labels,unit="FT"):
    points=np.array([(item["row_position"],item["selected_value"]) for item in labels],float)
    if len(points)<2: raise ValueError("At least two depth labels are required")
    slope,intercept=np.polyfit(points[:,0],points[:,1],1); residual=np.abs(points[:,1]-(slope*points[:,0]+intercept)); keep=residual<=max(1.,3*np.median(residual)+1e-6)
    kept=points[keep]; error=float(np.mean(residual[keep])); return DepthMapping([tuple(row) for row in kept],unit=unit,residual_error=error,confidence=float(np.exp(-error/max(1,np.ptp(kept[:,1])))))
