from __future__ import annotations
import hashlib,numpy as np
from .models import *
from .ordering import order_pages
from .depth_alignment import normalize_depth,relationship
from .curve_matching import match_curves
from .resampling import canonical_grid,resample_curve
from .overlap import wrap_offset
from .review import export_status

def assemble_whole_log(pages,config=None):
    config=config or WholeLogConfig()
    if not pages:raise ValueError("At least one reviewed page trace is required")
    units={str(page.depth_unit).upper() for page in pages};warnings=[]
    try: ordered,alternatives=order_pages(pages,config.canonical_depth_unit)
    except ValueError as exc: raise ValueError(f"Conflicting or unknown depth units: {exc}")
    if len(units)>1:warnings.append({"type":"depth unit conversion","severity":"medium","units":sorted(units)})
    joins=[];gaps=[];duplicates=[]
    for first,second in zip(ordered,ordered[1:]):
        bottom=normalize_depth(first.depth_bottom,first.depth_unit,config.canonical_depth_unit)["canonical_value"];top=normalize_depth(second.depth_top,second.depth_unit,config.canonical_depth_unit)["canonical_value"]
        kind,delta=relationship(bottom,top,config);join={"page_a":first.page_id,"page_b":second.page_id,"relationship":kind,"depth_delta":delta,"status":"automatic_proposal","confidence":.9 if kind=="exact_continuation" else .65};joins.append(join)
        if "gap" in kind:
            severity="high" if kind=="large_gap" else "medium";gap={"depth_start":bottom,"depth_end":top,"gap_size":delta,"before_page":first.page_id,"after_page":second.page_id,"severity":severity,"possible_causes":["missing page","cropped page","depth OCR error"]};gaps.append(gap)
            if severity=="high":warnings.append({"type":"large unexplained gap","severity":"high",**gap})
        if first.source_hash and first.source_hash==second.source_hash:duplicates.append({"duplicate_group":[first.page_id,second.page_id],"recommended_source":max((first,second),key=lambda p:p.page_confidence.get("overall",0)).page_id,"reason":"identical source hash"})
    identities=match_curves([(page.page_id,curve) for page in ordered for curve in page.curves]);grid,step=canonical_grid(ordered,config.depth_step);whole_curves=[];provenance_index={}
    for identity in identities:
        if identity["conflicts"]:warnings.append({"type":"curve unit mismatch","severity":"high","curve":identity["curve_id"]})
        candidates=[];page_offsets={};previous=None;running_offset=0
        for page_id,curve in identity["members"]:
            if previous is not None:
                reconciled=wrap_offset(previous,curve,config.maximum_wrap_offset);running_offset+=int(reconciled["offset"])
                wrap_relevant=bool(previous.scale.get("cycle_value") or curve.scale.get("cycle_value") or np.any(previous.wrap_index) or np.any(curve.wrap_index))
                if wrap_relevant and reconciled["confidence"]<.5:warnings.append({"type":"low-confidence wrap offset","severity":"high","curve":identity["curve_id"],"page_id":page_id,"offset":running_offset,"confidence":reconciled["confidence"]})
            page_offsets[page_id]=running_offset;previous=curve
            values,confidence=resample_curve(curve,grid,max(config.maximum_small_gap,step*1.5));candidates.append((page_id,curve,values,confidence))
        output=np.full(grid.size,np.nan);conf=np.zeros(grid.size);flags=[[] for _ in grid];provenance=[];whole_wrap=np.zeros(grid.size,dtype=np.int32);previous_curve=None;running_offset=0
        for row,depth in enumerate(grid):
            available=[item for item in candidates if np.isfinite(item[2][row])]
            if not available:provenance.append(SampleProvenance(float(depth),[]));continue
            available.sort(key=lambda item:(-item[3][row],item[0]));best=available[0];selected=[best]
            if len(available)>1:
                span=max(abs(item[2][row]) for item in available)+1e-6;disagreement=max(item[2][row] for item in available)-min(item[2][row] for item in available)
                if disagreement/span<=config.value_agreement_tolerance:selected=available;flags[row].append("blended")
                else:flags[row].append("overlap_conflict")
            weights=np.array([max(item[3][row],1e-6) for item in selected]);output[row]=np.average([item[2][row] for item in selected],weights=weights);conf[row]=float(np.mean(weights));flags[row].append("original" if len(selected)==1 else "resampled")
            sources=[]
            for page_id,curve,values,confidence in selected:
                nearest=int(np.argmin(abs(curve.depth-depth)));adjusted_wrap=int(curve.wrap_index[nearest])+page_offsets[page_id];sources.append({"source_page_id":page_id,"source_curve_id":curve.curve_id,"source_image_row":float(curve.image_rows[nearest]),"source_depth":float(curve.depth[nearest]),"source_value":float(curve.values[nearest]),"local_wrap_index":int(curve.wrap_index[nearest]),"page_wrap_offset":page_offsets[page_id],"whole_log_wrap_index":adjusted_wrap,"confidence":float(curve.confidence[nearest]),"model_version":curve.model_version,"decoder_version":curve.decoder_version})
            whole_wrap[row]=sources[0]["whole_log_wrap_index"]
            provenance.append(SampleProvenance(float(depth),sources,resampled=any(abs(s["source_depth"]-depth)>1e-8 for s in sources),blended=len(selected)>1))
        curve=WholeLogCurve(identity["curve_id"],identity["mnemonic"],identity["unit"],None,grid,output,conf,flags,provenance,sorted({p for p,_ in identity["members"]}),whole_wrap,identity["conflicts"]);whole_curves.append(curve);provenance_index[curve.curve_id]=[p.sources for p in provenance]
    log_id=hashlib.sha256("|".join(page.page_id for page in ordered).encode()).hexdigest()[:16];result=WholeLogResult(log_id,[p.page_id for p in ordered],whole_curves,joins,gaps,duplicates,warnings,{"ordering_alternatives":alternatives,"weakest_join":min((j["confidence"] for j in joins),default=1.),"canonical_depth_step":step},provenance_index,initial_automatic_page_order=[p.page_id for p in ordered],initial_automatic_joins=[dict(j) for j in joins]);result.status=export_status(result);return result

__all__=["assemble_whole_log","PageTraceResult","PageCurve","WholeLogConfig","WholeLogResult"]
