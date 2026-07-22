from __future__ import annotations
import numpy as np

def summarize_confidence(curve, low_threshold=.60):
    confidence=np.asarray(curve.confidence,float)
    if confidence.size==0:return {"mean_confidence":None,"minimum_confidence":None,"p05_confidence":None,"low_confidence_fraction":None,"longest_low_confidence_interval":0.}
    low=confidence<low_threshold;longest=0.;start=None
    for index,is_low in enumerate(np.r_[low,False]):
        if is_low and start is None:start=index
        elif not is_low and start is not None:
            if index>start:longest=max(longest,float(curve.depth[index-1]-curve.depth[start]))
            start=None
    return {"mean_confidence":float(np.mean(confidence)),"minimum_confidence":float(np.min(confidence)),"p05_confidence":float(np.percentile(confidence,5)),"low_confidence_fraction":float(np.mean(low)),"longest_low_confidence_interval":longest}
