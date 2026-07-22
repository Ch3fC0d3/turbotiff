from __future__ import annotations
import numpy as np
FACTORS={"FT":1.,"FEET":1.,"M":3.280839895,"METERS":3.280839895}
def normalize_depth(value,unit,canonical="FT"):
    key=str(unit or "").upper()
    if key not in FACTORS: raise ValueError(f"Unknown depth unit: {unit}")
    feet=float(value)*FACTORS[key]; result=feet if canonical=="FT" else feet/FACTORS[canonical]
    return {"source_value":float(value),"source_unit":key,"canonical_value":result,"canonical_unit":canonical,"conversion_factor":FACTORS[key]/FACTORS[canonical]}
def relationship(a_bottom,b_top,config):
    delta=float(b_top-a_bottom)
    if abs(delta)<=config.exact_tolerance:return "exact_continuation",delta
    if delta<0:return ("small_overlap" if -delta<=config.small_overlap_max else "large_overlap"),delta
    return ("small_gap" if delta<=config.small_gap_max else "large_gap"),delta
def fit_alignment(source_depth,target_depth,maximum_stretch=.03):
    source=np.asarray(source_depth,float); target=np.asarray(target_depth,float)
    if source.size!=target.size or source.size<1:raise ValueError("Alignment vectors must have equal nonzero length")
    if source.size==1:return {"model":"constant_offset","scale":1.,"offset":float(target[0]-source[0]),"residual":0.}
    scale,offset=np.polyfit(source,target,1)
    if abs(scale-1)>maximum_stretch:raise ValueError("Required depth stretch exceeds safeguard")
    residual=float(np.mean(np.abs(target-(scale*source+offset))))
    return {"model":"constant_offset" if abs(scale-1)<1e-4 else "linear_stretch","scale":float(scale),"offset":float(offset),"residual":residual}
