from __future__ import annotations
from copy import deepcopy
from datetime import datetime,timezone

CRITICAL=("track_bounds","depth_mapping","scale_type","scale_endpoints","scale_direction","curve_identity","wrap_topology")
def safety_status(result):
    if result.page_classification!="log_data": return "not_a_log_data_page"
    critical=result.confidence_summary.get("critical_fields",{})
    if any(critical.get(name,0)<.5 for name in CRITICAL): return "manual_setup_required"
    if result.warnings or any(critical.get(name,0)<.8 for name in CRITICAL): return "trace_with_warning"
    return "safe_for_auto_trace"

def apply_review(result,edits,reviewer,status="reviewed"):
    reviewed=deepcopy(result); history=[]
    for path,value in edits.items():
        target=reviewed
        parts=path.split(".")
        for part in parts[:-1]: target=getattr(target,part) if hasattr(target,part) else target[part]
        key=parts[-1]; automatic=getattr(target,key) if hasattr(target,key) else target.get(key)
        if hasattr(target,key): setattr(target,key,value)
        else: target[key]=value
        history.append({"field":path,"automatic_value":automatic,"reviewed_value":value,"edited":True})
    reviewed.review_status=status; reviewed.processing_metadata["review"]={"reviewer":reviewer,"reviewed_at":datetime.now(timezone.utc).isoformat(),"edit_history":history,"automatic_result_preserved":True}
    return reviewed

def to_tracing_requests(result):
    if result.review_status!="approved": raise PermissionError("Page setup must be approved before authoritative tracing")
    requests=[]
    depth=result.depth_columns[0] if result.depth_columns else {}
    for track in result.tracks:
        for candidate in track.curve_candidates:
            requests.append({"track_bounds":track.bounds.to_dict(),"curve_prompt":candidate,"scale_definition":track.scale.__dict__,"depth_mapping":depth,"evidence_mode":"neural_phase2","decoder_mode":"topology_dp","topology":"cylindrical" if track.scale.scale_type=="logarithmic" else "bounded"})
    return requests
