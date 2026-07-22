from __future__ import annotations
from .models import EvidenceCrop

def create_evidence_crop(whole_log,curve_id,depth_start,depth_end):
    """Return lazy source-image references; raster loading is deliberately deferred."""
    curve=next((item for item in whole_log.curves if item.curve_id==curve_id),None)
    if curve is None:raise KeyError(curve_id)
    references=[]
    flags=[]
    for index,(depth,provenance) in enumerate(zip(curve.depth,curve.provenance)):
        if depth_start<=depth<=depth_end:
            flags.append(curve.quality_flags[index])
            for source in provenance.sources:references.append(dict(source,output_depth=float(depth)))
    return EvidenceCrop(curve_id,float(depth_start),float(depth_end),tuple(references),{"quality_flags":flags},status="references_only" if references else "source_unavailable")

__all__=["create_evidence_crop"]
