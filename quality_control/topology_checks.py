from __future__ import annotations
import numpy as np
def check_topology(curve,config,add):
    wrap=curve.whole_log_wrap_index
    if wrap is None:return {"curve_id":curve.curve_id,"wrap_events":0}
    wrap=np.asarray(wrap,int);change=np.diff(wrap);events=np.flatnonzero(change!=0)+1
    if np.any(np.abs(change)>1):add("wrap","critical",curve,"Wrap index changes by more than one cycle",evidence={"maximum_change":int(np.max(np.abs(change)))})
    if len(events)>=2 and np.any(np.diff(events)<=1):add("wrap","high",curve,"Rapid wrap/reverse-wrap oscillation",float(curve.depth[events[0]]),float(curve.depth[events[-1]]))
    for event in events:
        jump=abs(float(curve.values[event]-curve.values[event-1])) if np.all(np.isfinite(curve.values[event-1:event+1])) else 0
        local=np.nanmedian(np.abs(np.diff(curve.values[max(0,event-5):event+5])))
        if local>0 and jump>10*local:add("wrap","critical",curve,"Wrap event creates a large physical-value discontinuity",float(curve.depth[event-1]),float(curve.depth[event]),{"jump":jump})
    return {"curve_id":curve.curve_id,"wrap_events":len(events)}
