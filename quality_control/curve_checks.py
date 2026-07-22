from __future__ import annotations
import numpy as np
def _runs(mask):
    padded=np.r_[False,mask,False];starts=np.flatnonzero(np.diff(padded.astype(np.int8))==1);ends=np.flatnonzero(np.diff(padded.astype(np.int8))==-1);return zip(starts,ends)
def check_curve(curve,config,add):
    values=np.asarray(curve.values,float);depth=np.asarray(curve.depth,float);confidence=np.asarray(curve.confidence,float);valid=np.isfinite(values)
    result={"curve_id":curve.curve_id,"mnemonic":curve.mnemonic,"unit":curve.unit,"description":curve.description,"valid_samples":int(valid.sum()),"null_fraction":float(1-valid.mean()) if len(values) else 1.,"mean_confidence":float(np.mean(confidence)) if len(confidence) else 0,"source_pages":list(curve.source_pages)}
    if len(values)!=len(depth):
        add("curve_structure","critical",curve,"Curve and depth arrays have different lengths",evidence={"depth":len(depth),"values":len(values)})
        return result
    if len(confidence)!=len(depth) or len(curve.quality_flags)!=len(depth):add("curve_structure","critical",curve,"Confidence or quality flags do not match depth length")
    if len(curve.provenance)!=len(depth):add("provenance","critical",curve,"Provenance array does not match depth length",evidence={"depth":len(depth),"provenance":len(curve.provenance)})
    if np.any(np.isinf(values)):add("curve_values","critical",curve,"Curve contains infinity")
    if curve.unit is None:add("unit","critical",curve,"Curve unit is unresolved")
    rule=config.curve_rules.get(curve.mnemonic)
    if rule and curve.unit not in rule.get("units",[]):add("unit","high",curve,"Curve unit conflicts with configured rule",evidence={"unit":curve.unit,"rule":rule})
    if rule and valid.any():
        outside=valid&((values<rule["hard_min"])|(values>rule["hard_max"]))
        if outside.any():add("range","high",curve,"Values outside configured hard limits",float(depth[outside][0]),float(depth[outside][-1]),{"rule":rule,"count":int(outside.sum())})
    for index,flags in enumerate(curve.quality_flags):
        normalized={str(flag).lower() for flag in flags}
        if "overlap_conflict" in normalized:add("join","high",curve,"Conflicting overlap samples remain unresolved",float(depth[index]),float(depth[index]),{"quality_flags":sorted(normalized)});break
        if "grid_lock_suspected" in normalized:add("grid_lock","medium",curve,"Decoder marked possible grid lock",float(depth[index]),float(depth[index]),{"quality_flags":sorted(normalized)});break
    if valid.sum()>=5:
        clean=values[valid];d=depth[valid];window=max(3,config.hampel_window|1);half=window//2
        spikes=[]
        for index in range(half,len(clean)-half):
            local=clean[index-half:index+half+1];median=np.median(local);mad=np.median(np.abs(local-median))*1.4826
            if mad>0 and abs(clean[index]-median)>config.spike_sigma*mad:spikes.append(index)
        for index in spikes:add("spike","medium",curve,"Possible isolated tracing spike",float(d[index]),float(d[index]),{"value":float(clean[index])})
        near=np.abs(np.diff(clean))<=config.flat_value_tolerance
        for start,end in _runs(near):
            if end<len(d) and d[end]-d[start]>=config.flat_minimum_depth:add("flat_line","medium",curve,"Possible flat line or grid lock",float(d[start]),float(d[end]),{"duration":float(d[end]-d[start])})
    if len(confidence)!=len(depth):return result
    low=confidence<config.low_confidence_threshold
    for start,end in _runs(low):
        if end>start:add("confidence","high" if np.min(confidence[start:end])<config.critical_confidence_threshold else "medium",curve,"Low-confidence interval",float(depth[start]),float(depth[end-1]),{"minimum":float(np.min(confidence[start:end]))})
    for index,value in enumerate(values):
        if np.isfinite(value) and (index>=len(curve.provenance) or not curve.provenance[index].sources):add("provenance","critical",curve,"Valid sample has no source provenance",float(depth[index]),float(depth[index]));break
    return result
