from copy import deepcopy
from datetime import datetime,timezone
CRITICAL=("conflicting depth units","curve unit mismatch","large unexplained gap","unresolved overlap disagreement","low-confidence wrap offset","separate runs merged")
def export_status(result):
    unresolved=[warning for warning in result.warnings if warning.get("severity")=="high" and not warning.get("reviewed")]
    return "needs_review" if unresolved or any(join.get("status") not in {"approved","manual"} for join in result.joins) else "export_ready"
def apply_review(result,page_order=None,join_edits=None,reviewer=""):
    reviewed=deepcopy(result); history=[]
    if page_order is not None:history.append({"operation":"reorder_pages","automatic":reviewed.ordered_pages,"reviewed":page_order});reviewed.ordered_pages=list(page_order)
    for index,changes in (join_edits or {}).items():
        before=dict(reviewed.joins[int(index)]);reviewed.joins[int(index)].update(changes);history.append({"operation":"edit_join","index":int(index),"automatic":before,"reviewed":dict(reviewed.joins[int(index)])})
    reviewed.review={"reviewer":reviewer,"reviewed_at":datetime.now(timezone.utc).isoformat(),"edit_history":history};reviewed.status=export_status(reviewed);return reviewed
