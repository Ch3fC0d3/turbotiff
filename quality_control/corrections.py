from copy import deepcopy
from datetime import datetime,timezone
RESOLVED={"acknowledged","corrected","accepted_as_real","false_positive","deferred","not_applicable"}
def review_finding(qc,finding_id,status,reviewer,notes="",related_correction=None,new_qc_run=None,allow_accepted_blocker=True):
    if status not in RESOLVED:raise ValueError("Invalid finding review state")
    revised=deepcopy(qc);finding=next((item for item in revised.findings if item.finding_id==finding_id),None)
    if finding is None:raise KeyError(finding_id)
    finding.review_status=status;finding.review={"reviewer":reviewer,"reviewed_at":datetime.now(timezone.utc).isoformat(),"resolution":status,"notes":notes,"related_correction":related_correction,"new_qc_run":new_qc_run}
    resolvable=status in {"corrected","false_positive","not_applicable"} or (status=="accepted_as_real" and allow_accepted_blocker)
    revised.export_blockers=[item for item in revised.export_blockers if item.finding_id!=finding_id or not resolvable]
    revised.status="blocked" if revised.export_blockers else ("needs_review" if any(item.review_status=="open" for item in revised.findings) else "reviewed");return revised
