from __future__ import annotations
import numpy as np
def shape_similarity(depth_a,value_a,depth_b,value_b,start,end,points=100):
    grid=np.linspace(start,end,points); a=np.interp(grid,depth_a,value_a); b=np.interp(grid,depth_b,value_b)
    if np.std(a)<1e-8 or np.std(b)<1e-8:return 0.
    return float(np.corrcoef(a,b)[0,1])
def best_depth_shift(curve_a,curve_b,start,end,max_shift=2.,steps=41):
    candidates=[]
    for shift in np.linspace(-max_shift,max_shift,steps): candidates.append((shape_similarity(curve_a.depth,curve_a.values,curve_b.depth+shift,curve_b.values,start,end),shift))
    score,shift=max(candidates,key=lambda item:(item[0],-abs(item[1]))); return {"depth_shift":float(shift),"similarity":score,"candidates":[{"shift":float(s),"similarity":float(c)} for c,s in candidates]}
def wrap_offset(curve_a,curve_b,maximum=3):
    scale=curve_a.scale or {}; cycle=float(scale.get("cycle_value",0) or 0)
    if not cycle:return {"offset":0,"confidence":.25}
    end=float(curve_a.values[-1]); start=float(curve_b.values[0]); candidates=[(abs(end-(start+k*cycle)),k) for k in range(-maximum,maximum+1)]
    error,offset=min(candidates); return {"offset":offset,"confidence":float(np.exp(-error/max(abs(cycle),1e-6)))}
