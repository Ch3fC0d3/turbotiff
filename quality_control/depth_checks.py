from __future__ import annotations
import numpy as np
def check_depth(curve,config,add):
    depth=np.asarray(curve.depth,float); result={"curve_id":curve.curve_id,"sample_count":len(depth)}
    if len(depth)!=len(curve.values):add("depth","critical",curve,"Curve and depth arrays have different lengths",evidence={"depth":len(depth),"values":len(curve.values)});return result
    if not len(depth):add("depth","critical",curve,"Depth array is empty");return result
    if not np.all(np.isfinite(depth)):add("depth","critical",curve,"Depth contains NaN or infinity")
    difference=np.diff(depth)
    if np.any(difference<=config.monotonic_tolerance):
        duplicates=np.flatnonzero(np.abs(difference)<=config.duplicate_tolerance)
        if len(duplicates):add("depth","critical",curve,"Duplicate depth samples remain",float(depth[duplicates[0]]),float(depth[duplicates[-1]+1]),{"count":len(duplicates)})
        if np.any(difference<-config.monotonic_tolerance):add("depth","critical",curve,"Depth is non-monotonic")
    positive=difference[difference>config.monotonic_tolerance]
    median=float(np.median(positive)) if len(positive) else None;result.update(median_step=median,min_step=float(np.min(positive)) if len(positive) else None,max_step=float(np.max(positive)) if len(positive) else None)
    if median and not config.allow_irregular_sampling:
        gaps=np.flatnonzero(difference>median*(1+config.expected_step_tolerance))
        for index in gaps:
            missing=max(0,round(difference[index]/median)-1);severity="high" if difference[index]>config.medium_gap_max_depth else ("medium" if missing>config.small_gap_max_samples else "low")
            add("depth_gap",severity,curve,"Unexpected depth gap",float(depth[index]),float(depth[index+1]),{"depth_before":float(depth[index]),"depth_after":float(depth[index+1]),"gap_size":float(difference[index]),"expected_step":median,"missing_sample_count":missing,"intentional":False,"interpolated":False})
    rounded=np.round(depth,config.depth_decimals)
    if len(np.unique(rounded))!=len(rounded):add("export_precision","high",curve,"Selected depth precision creates duplicate rows",evidence={"depth_decimals":config.depth_decimals})
    return result
