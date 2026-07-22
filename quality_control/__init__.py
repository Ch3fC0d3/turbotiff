from __future__ import annotations
import hashlib,json
import numpy as np
from dataclasses import asdict
from .models import QualityControlConfig,QualityControlResult,QualityFinding,EvidenceCrop
from .severity import blocks
from .depth_checks import check_depth
from .curve_checks import check_curve
from .topology_checks import check_topology
from .join_checks import check_joins
from .metadata_checks import check_metadata
from .las_validation import serialize_las,validate_las_text
from .confidence import summarize_confidence

def whole_log_hash(whole_log):
    digest=hashlib.sha256();digest.update(whole_log.log_id.encode());digest.update(json.dumps(whole_log.ordered_pages,sort_keys=True).encode())
    for curve in whole_log.curves:
        digest.update(curve.curve_id.encode());digest.update(str(curve.mnemonic).encode());digest.update(str(curve.unit).encode());digest.update(str(curve.description).encode())
        digest.update(np.asarray(curve.depth).tobytes());digest.update(np.asarray(curve.values).tobytes());digest.update(np.asarray(curve.confidence).tobytes());digest.update(np.asarray(curve.whole_log_wrap_index).tobytes() if curve.whole_log_wrap_index is not None else b"none");digest.update(json.dumps(curve.quality_flags,sort_keys=True).encode())
        digest.update(json.dumps([asdict(item) for item in curve.provenance],sort_keys=True,default=str).encode())
    for payload in (whole_log.joins,whole_log.gaps,whole_log.duplicate_intervals,whole_log.warnings,whole_log.confidence_summary):digest.update(json.dumps(payload,sort_keys=True,default=str).encode())
    return digest.hexdigest()

def evaluate_whole_log_quality(whole_log,config=None,metadata=None):
    config=config or QualityControlConfig();metadata=dict(metadata or {});findings=[]
    data_hash=whole_log_hash(whole_log)
    run_seed=json.dumps({"whole_log_hash":data_hash,"config":asdict(config),"metadata":metadata},sort_keys=True,default=str)
    qc_run_id="qc_"+hashlib.sha256(run_seed.encode()).hexdigest()[:16]
    def add(category,severity,curve,message,depth_start=None,depth_end=None,evidence=None,recommended_action=None):
        curve_id=getattr(curve,"curve_id",None);seed=json.dumps([category,severity,curve_id,depth_start,depth_end,message,evidence or {}],sort_keys=True,default=str)
        finding=QualityFinding(hashlib.sha256(seed.encode()).hexdigest()[:16],category,severity,curve_id,depth_start,depth_end,message,evidence or {},recommended_action,blocks(severity));findings.append(finding);return finding
    depth_results=[];curve_results=[]
    reference_depth=None
    for curve in whole_log.curves:
        depth_results.append(check_depth(curve,config,add))
        result=check_curve(curve,config,add);result.update(check_topology(curve,config,add));result["confidence_summary"]=summarize_confidence(curve,config.low_confidence_threshold);curve_results.append(result)
        if reference_depth is None:reference_depth=np.asarray(curve.depth)
        elif not np.array_equal(reference_depth,curve.depth):add("cross_curve","critical",curve,"Curves do not share the same final depth array")
    join_results=check_joins(whole_log,config,add);metadata_results=check_metadata(metadata,whole_log.curves,add,config)
    try:
        las=serialize_las(whole_log,metadata,config);las_result=validate_las_text(las,len(whole_log.curves),config,reference_depth)
    except Exception as exc:
        add("las_format","critical",None,"LAS serialization failed",evidence={"error":str(exc)});las_result={"passed":False,"findings":[str(exc)],"independent_validation":{"ran":False,"parser":None,"error":None}}
    if not las_result["passed"] and not any(item.category=="las_format" for item in findings):add("las_format","critical",None,"LAS validation failed",evidence=las_result)
    weights={"info":0,"low":2,"medium":8,"high":25,"critical":100}
    categories=("depth","curve","join","wrap","metadata","provenance","las","cross_curve")
    category_scores={name:max(0.,100.-sum(weights[f.severity] for f in findings if f.category.startswith(name))) for name in categories}
    overall=float(np.prod([max(score,1e-12)/100 for score in category_scores.values()])**(1/len(category_scores))*100)
    blockers=[f for f in findings if f.blocks_approval and f.review_status=="open"];status="blocked" if blockers else ("needs_review" if findings else "reviewed")
    processing={"configuration":asdict(config),"configuration_hash":hashlib.sha256(json.dumps(asdict(config),sort_keys=True,default=str).encode()).hexdigest(),"las_validation":las_result,"source_status":whole_log.status}
    return QualityControlResult(whole_log.log_id,qc_run_id,data_hash,status,overall,category_scores,curve_results,depth_results,join_results,metadata_results,[f for f in findings if f.severity=="critical"],[f for f in findings if f.severity in {"low","medium","high"}],[f for f in findings if f.severity=="info"],blockers,["Resolve or explicitly review every blocking finding","Complete the role-gated approval checklist","Confirm the approved data hash matches current data"],processing,findings=findings)

from .approval import ApprovalRecord,approve,approval_valid
from .corrections import review_finding
from .diagnostics import create_evidence_crop
from .preview import create_las_preview

__all__=["evaluate_whole_log_quality","whole_log_hash","QualityControlConfig","QualityControlResult","QualityFinding","EvidenceCrop","ApprovalRecord","approve","approval_valid","review_finding","create_evidence_crop","create_las_preview","serialize_las","validate_las_text"]
