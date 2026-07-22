from __future__ import annotations
import numpy as np
def canonical_grid(pages,step=None):
    intervals=[]
    for page in pages:
        for curve in page.curves:
            diff=np.diff(np.sort(np.unique(curve.depth))); intervals.extend(diff[diff>1e-9])
    selected=float(step or (np.median(intervals) if intervals else 1.)); low=min(float(np.min(c.depth)) for p in pages for c in p.curves); high=max(float(np.max(c.depth)) for p in pages for c in p.curves)
    return np.arange(low,high+selected*.5,selected),selected
def resample_curve(curve,grid,maximum_gap):
    order=np.argsort(curve.depth); depth=curve.depth[order]; values=curve.values[order]; confidence=curve.confidence[order]
    output=np.full(grid.size,np.nan); conf=np.zeros(grid.size); valid=(grid>=depth[0])&(grid<=depth[-1]); output[valid]=np.interp(grid[valid],depth,values); conf[valid]=np.interp(grid[valid],depth,confidence)
    positions=np.searchsorted(depth,grid[valid]); left=np.clip(positions-1,0,len(depth)-1);right=np.clip(positions,0,len(depth)-1); bad=(depth[right]-depth[left])>maximum_gap; indices=np.flatnonzero(valid);output[indices[bad]]=np.nan;conf[indices[bad]]=0
    return output,conf
