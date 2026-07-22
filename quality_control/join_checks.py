from __future__ import annotations
import numpy as np

def check_joins(whole_log,config,add):
    results=[]
    for index,join in enumerate(whole_log.joins):
        join_id=f"join_{index+1}";page_a=join.get("page_a");page_b=join.get("page_b")
        result={"join_id":join_id,"pages":[page_a,page_b],"status":join.get("status"),"value_discontinuities":{},"slope_discontinuities":{},"wrap_consistency":{},"identity_consistency":{},"findings":[]}
        if join.get("status") not in {"approved","manual"}:add("join","high",None,"Page join is not approved",evidence={"join":join,"join_id":join_id})
        if join.get("relationship")=="large_gap":add("join","high",None,"Large page gap remains unresolved",evidence={"join":join,"join_id":join_id})
        for curve in whole_log.curves:
            boundary=_page_boundary(curve,page_a,page_b)
            if boundary is None:continue
            left,right=boundary;jump=abs(float(curve.values[right]-curve.values[left]));window=np.asarray(curve.values[max(0,left-5):min(len(curve.values),right+6)],float);scale=float(np.nanstd(window));normalized=jump/max(scale,1e-12)
            result["value_discontinuities"][curve.curve_id]={"depth":float(curve.depth[right]),"absolute_jump":jump,"normalized_jump":normalized}
            if normalized>config.maximum_normalized_join_jump:add("join_boundary","high",curve,"Large page-boundary value discontinuity",float(curve.depth[left]),float(curve.depth[right]),{"join_id":join_id,"absolute_jump":jump,"normalized_jump":normalized})
        results.append(result)
    return results

def _page_boundary(curve,page_a,page_b):
    last_a=None
    for index,item in enumerate(curve.provenance):
        pages={source.get("source_page_id") for source in item.sources}
        if page_a in pages:last_a=index
        if page_b in pages and last_a is not None and index>last_a and np.isfinite(curve.values[last_a]) and np.isfinite(curve.values[index]):return last_a,index
    return None
