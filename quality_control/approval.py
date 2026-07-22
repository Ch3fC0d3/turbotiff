from __future__ import annotations
from dataclasses import dataclass,asdict
from datetime import datetime,timezone
import hashlib,json,uuid

@dataclass(frozen=True)
class ApprovalRecord:
    approval_id:str;log_id:str;whole_log_hash:str;qc_run_id:str;export_configuration_id:str;approved_by:str;approved_at:str;checklist_results:dict;unresolved_nonblocking_findings:list;notes:str|None;content_hash:str

REQUIRED=("depth_range_verified","depth_unit_verified","curve_identities_verified","curve_units_verified","scales_verified","page_order_verified","joins_reviewed","wrap_events_reviewed","critical_warnings_resolved","las_metadata_reviewed","output_preview_reviewed")
def approval_checklist(whole_log):
    required=list(REQUIRED)
    if not whole_log.joins:required.remove("joins_reviewed")
    if not any(getattr(curve,"whole_log_wrap_index",None) is not None and any(curve.whole_log_wrap_index) for curve in whole_log.curves):required.remove("wrap_events_reviewed")
    return required
def approve(qc,whole_log_hash,approved_by,roles,checklist,export_configuration_id="default",notes=None):
    if "approver" not in roles:raise PermissionError("Approver role is required")
    if qc.whole_log_hash!=whole_log_hash:raise ValueError("Whole-log data changed after QC")
    if qc.export_blockers:raise PermissionError("Unresolved QC blockers prevent approval")
    if qc.status!="reviewed":raise PermissionError("QC review must be complete before approval")
    if not all(checklist.get(key) for key in REQUIRED):raise PermissionError("Approval checklist is incomplete")
    payload={"log_id":qc.log_id,"whole_log_hash":whole_log_hash,"qc_run_id":qc.qc_run_id,"export_configuration_id":export_configuration_id,"approved_by":approved_by,"checklist_results":checklist};content_hash=hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()
    return ApprovalRecord(str(uuid.uuid4()),qc.log_id,whole_log_hash,qc.qc_run_id,export_configuration_id,approved_by,datetime.now(timezone.utc).isoformat(),dict(checklist),[f.finding_id for f in qc.findings if not f.blocks_approval and f.review_status in {"acknowledged","deferred"}],notes,content_hash)
def approval_valid(record,current_hash):return record.whole_log_hash==current_hash
