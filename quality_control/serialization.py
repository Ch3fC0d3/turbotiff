from __future__ import annotations
import hashlib,json,os,uuid
from datetime import datetime,timezone
from pathlib import Path
from .approval import approval_valid
from .las_validation import serialize_las,validate_las_text

def write_las(path,whole_log,qc,metadata,config,approval=None,draft=False):
    path=Path(path)
    from . import whole_log_hash
    if whole_log_hash(whole_log)!=qc.whole_log_hash:raise PermissionError("Whole-log data changed after QC; rerun QC before export")
    if approval is None and not draft:raise PermissionError("Approved export requires an approval record; use draft=True for an unapproved LAS")
    if approval is not None:
        if qc.export_blockers:raise PermissionError("QC blockers prevent approved export")
        if not approval_valid(approval,qc.whole_log_hash):raise PermissionError("Approval data hash does not match current QC data")
        if approval.qc_run_id!=qc.qc_run_id:raise PermissionError("Approval belongs to a different QC run")
    text=serialize_las(whole_log,metadata,config,approved=approval is not None)
    validation=validate_las_text(text,len(whole_log.curves),config,whole_log.curves[0].depth)
    if not validation["passed"]:raise ValueError(f"LAS validation failed: {validation['findings']}")
    path.parent.mkdir(parents=True,exist_ok=True);temporary=path.with_suffix(path.suffix+".tmp");temporary.write_text(text,encoding="ascii",newline="\n");os.replace(temporary,path)
    return path,validation

def create_export_manifest(output_dir,las_path,qc,approval,created_by,versions=None,companion_files=()):
    if approval is None:raise PermissionError("Delivery manifest requires approval")
    if qc.export_blockers or not approval_valid(approval,qc.whole_log_hash):raise PermissionError("Current data are not approved")
    output=Path(output_dir);output.mkdir(parents=True,exist_ok=True);las_path=Path(las_path);digest=hashlib.sha256(las_path.read_bytes()).hexdigest();export_id=f"export_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}";path=output/f"{export_id}.json"
    version_data=dict(versions or {})
    payload={"export_id":export_id,"log_id":qc.log_id,"whole_log_hash":qc.whole_log_hash,"qc_run_id":qc.qc_run_id,"approval_id":approval.approval_id,"las_version":qc.processing_metadata["configuration"]["las_version"],"las_file":las_path.name,"las_sha256":digest,"companion_files":[Path(item).name for item in companion_files],"created_at":datetime.now(timezone.utc).isoformat(),"created_by":created_by,"software_commit":version_data.pop("software_commit",None),"model_versions":version_data.pop("model_versions",{}),"decoder_version":version_data.pop("decoder_version",None),"page_analysis_version":version_data.pop("page_analysis_version",None),"assembly_version":version_data.pop("assembly_version",None),"additional_versions":version_data}
    temporary=path.with_suffix(".tmp");temporary.write_text(json.dumps(payload,indent=2,sort_keys=True),encoding="utf-8");os.replace(temporary,path);return path,payload

def validate_manifest(path,las_path):
    payload=json.loads(Path(path).read_text(encoding="utf-8"));return payload["las_sha256"]==hashlib.sha256(Path(las_path).read_bytes()).hexdigest()

def compare_exports(old_manifest,new_manifest,old_las=None,new_las=None):
    old=json.loads(Path(old_manifest).read_text()) if not isinstance(old_manifest,dict) else old_manifest;new=json.loads(Path(new_manifest).read_text()) if not isinstance(new_manifest,dict) else new_manifest
    result={"metadata_changes":{key:[old.get(key),new.get(key)] for key in sorted(set(old)|set(new)) if old.get(key)!=new.get(key)},"las_changed":old.get("las_sha256")!=new.get("las_sha256"),"approval_changed":old.get("approval_id")!=new.get("approval_id")}
    if old_las and new_las:
        old_rows=_data_rows(old_las);new_rows=_data_rows(new_las);common=min(len(old_rows),len(new_rows));changed=0;maximum=0.
        for left,right in zip(old_rows[:common],new_rows[:common]):
            if left!=right:changed+=1;maximum=max(maximum,max((abs(a-b) for a,b in zip(left,right)),default=0.))
        result.update({"old_sample_count":len(old_rows),"new_sample_count":len(new_rows),"changed_rows":changed+abs(len(old_rows)-len(new_rows)),"maximum_value_change":maximum})
    return result

def _data_rows(path):
    lines=Path(path).read_text().splitlines();start=next((index for index,line in enumerate(lines) if line.strip().upper().startswith("~ASCII")),len(lines));return [[float(value) for value in line.split()] for line in lines[start+1:] if line.strip()]
