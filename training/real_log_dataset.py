"""Read-only curation of real TIFF/LAS pairs for reviewed TurboTIFF training data.

The module deliberately separates source association, numerical validation,
pixel alignment proposals, and human approval. A same-well TIFF/LAS folder is
never treated as ground-truth curve geometry until its alignment is reviewed.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
import re
import warnings
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class PairCandidate:
    pair_id: str
    source: str
    well_id: str
    tiff_paths: tuple[str, ...]
    las_path: str
    association_basis: str = "same_source_well_folder"


@dataclass
class PairAudit:
    pair: PairCandidate
    status: str
    association_confidence: float
    tiff_summaries: list[dict] = field(default_factory=list)
    las_summary: dict = field(default_factory=dict)
    findings: list[dict] = field(default_factory=list)
    split: str = ""
    training_eligible: bool = False
    alignment_status: str = "not_started"

    def to_dict(self):
        return asdict(self)


def discover_pairs(dataset_root: str | Path, standalone_pairs: Iterable[PairCandidate] = ()) -> list[PairCandidate]:
    """Discover source-grouped pair folders without changing or copying them."""
    root = Path(dataset_root)
    candidates = list(standalone_pairs)
    if root.exists():
        for source_dir in sorted(path for path in root.iterdir() if path.is_dir()):
            pairs_dir = source_dir / "pairs"
            if not pairs_dir.exists():
                continue
            for well_dir in sorted(path for path in pairs_dir.iterdir() if path.is_dir()):
                tiffs = sorted([*well_dir.glob("*.tif"), *well_dir.glob("*.tiff")], key=lambda path: path.name.lower())
                las_files = sorted(well_dir.glob("*.las"), key=lambda path: path.name.lower())
                if not tiffs or not las_files:
                    continue
                for las_path in las_files:
                    pair_id = _stable_id(source_dir.name, well_dir.name, las_path.name)
                    candidates.append(PairCandidate(pair_id, source_dir.name.lower(), well_dir.name, tuple(str(path.resolve()) for path in tiffs), str(las_path.resolve())))
    unique = {candidate.pair_id: candidate for candidate in candidates}
    return sorted(unique.values(), key=lambda candidate: (candidate.source, candidate.well_id, candidate.pair_id))


def dataset_inventory(dataset_root: str | Path) -> dict:
    root = Path(dataset_root);by_source={};total_files=total_tiffs=total_las=paired_directories=total_bytes=0
    if not root.exists():
        return {"dataset_root":str(root),"exists":False,"total_files":0,"tiff_files":0,"las_files":0,"paired_directories":0,"total_bytes":0,"by_source":{}}
    for source_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        files=[path for path in source_dir.rglob("*") if path.is_file()];tiffs=[path for path in files if path.suffix.lower() in {".tif",".tiff"}];las_files=[path for path in files if path.suffix.lower()==".las"]
        pairs_dir=source_dir/"pairs";paired=0
        if pairs_dir.exists():
            for directory in pairs_dir.iterdir():
                if directory.is_dir() and any(directory.glob("*.las")) and (any(directory.glob("*.tif")) or any(directory.glob("*.tiff"))):paired+=1
        size=sum(path.stat().st_size for path in files);by_source[source_dir.name.lower()]={"files":len(files),"tiff_files":len(tiffs),"las_files":len(las_files),"paired_directories":paired,"bytes":size}
        total_files+=len(files);total_tiffs+=len(tiffs);total_las+=len(las_files);paired_directories+=paired;total_bytes+=size
    return {"dataset_root":str(root.resolve()),"exists":True,"total_files":total_files,"tiff_files":total_tiffs,"las_files":total_las,"paired_directories":paired_directories,"total_bytes":total_bytes,"by_source":by_source}


def audit_pair(candidate: PairCandidate, content_hashes: bool = False, split_seed: str = "turbotiff-real-v1") -> PairAudit:
    findings=[];tiff_summaries=[];fatal=False
    for path_text in candidate.tiff_paths:
        path=Path(path_text)
        try:tiff_summaries.append(_tiff_summary(path,content_hashes))
        except Exception as exc:
            fatal=True;findings.append(_finding("tiff_unreadable","critical",str(exc),str(path)))
    try:las_summary=_las_summary(Path(candidate.las_path),content_hashes)
    except Exception as exc:
        fatal=True;las_summary={};findings.append(_finding("las_unreadable","critical",str(exc),candidate.las_path))
    if las_summary:
        if not las_summary["depth_monotonic"]:fatal=True;findings.append(_finding("depth_nonmonotonic","critical","LAS depth is not strictly monotonic",candidate.las_path))
        if las_summary["duplicate_depth_count"]:fatal=True;findings.append(_finding("duplicate_depth","critical","LAS contains duplicate depth rows",candidate.las_path,{"count":las_summary["duplicate_depth_count"]}))
        if not las_summary["curves"]:fatal=True;findings.append(_finding("no_curves","critical","LAS has no non-depth curves",candidate.las_path))
        if str(las_summary.get("version")) not in {"1.2","1.20","2.0","2.00","3.0","3.00"}:findings.append(_finding("unusual_las_version","medium","LAS version requires parser review",candidate.las_path,{"version":las_summary.get("version")}))
    if len(candidate.tiff_paths)>1:findings.append(_finding("multiple_tiffs","medium","One LAS is associated with multiple TIFFs; individual runs/pages must be matched",candidate.well_id,{"count":len(candidate.tiff_paths)}))
    findings.append(_finding("run_identity_unverified","medium","Same-well association does not prove the TIFF and LAS are the same logging run",candidate.well_id))
    confidence=.90 if candidate.association_basis=="same_source_well_folder" else .97
    status="rejected" if fatal else "needs_alignment_review"
    return PairAudit(candidate,status,confidence,tiff_summaries,las_summary,findings,assign_split(candidate.well_id,split_seed),False,"not_started")


def assign_split(well_id: str, seed: str = "turbotiff-real-v1") -> str:
    bucket=int(hashlib.sha256(f"{seed}|{well_id}".encode()).hexdigest()[:8],16)%100
    return "train" if bucket<80 else ("validation" if bucket<90 else "test")


def select_pilot(candidates: list[PairCandidate], kgs_count: int = 16, wvgs_count: int = 4) -> list[PairCandidate]:
    """Select wells deterministically; standalone candidates are always retained."""
    selected=[]
    limits={"kgs":kgs_count,"wvgs":wvgs_count}
    for source,count in limits.items():
        pool=[item for item in candidates if item.source==source]
        pool.sort(key=lambda item:hashlib.sha256(f"pilot-v1|{item.well_id}".encode()).hexdigest())
        selected.extend(pool[:count])
    selected.extend(item for item in candidates if item.source not in limits)
    return sorted({item.pair_id:item for item in selected}.values(),key=lambda item:(item.source,item.well_id))


def validate_alignment(alignment: dict, las_summary: dict | None = None) -> list[str]:
    errors=[];points=alignment.get("depth_control_points",[])
    if len(points)<2:return ["At least two depth control points are required"]
    rows=np.asarray([point["row"] for point in points],float);depths=np.asarray([point["depth"] for point in points],float)
    if not np.all(np.diff(rows)>0):errors.append("Depth control-point rows must increase")
    if not (np.all(np.diff(depths)>0) or np.all(np.diff(depths)<0)):errors.append("Depth control-point values must be monotonic")
    track_ids=set()
    for track in alignment.get("curve_tracks",[]):
        name=str(track.get("mnemonic","")).upper()
        track_id=str(track.get("track_id") or name)
        if not name:errors.append("Every curve track requires a mnemonic")
        if track_id in track_ids:errors.append(f"Duplicate alignment track ID: {track_id}")
        track_ids.add(track_id)
        if track.get("x_left")>=track.get("x_right"):errors.append(f"Invalid X bounds for {name}")
        if track.get("value_left")==track.get("value_right"):errors.append(f"Scale endpoints are equal for {name}")
        if track.get("scale_type","linear") not in {"linear","logarithmic"}:errors.append(f"Unsupported scale for {name}")
    if las_summary:
        available={item["mnemonic"].upper() for item in las_summary.get("curves",[])}
        for name in {str(track.get("mnemonic","")).upper() for track in alignment.get("curve_tracks",[])}:
            if name not in available:errors.append(f"Aligned curve is absent from LAS: {name}")
    return errors


def project_las_curves(las_path: str | Path, alignment: dict, sample_depth_interval: float = 1.0) -> list[dict]:
    import lasio
    errors=validate_alignment(alignment)
    if errors:raise ValueError("; ".join(errors))
    las=lasio.read(str(las_path),engine="normal",ignore_header_errors=True);depth=np.asarray(las.index,float)
    controls=sorted(alignment["depth_control_points"],key=lambda point:point["depth"]);control_depth=np.asarray([point["depth"] for point in controls],float);control_row=np.asarray([point["row"] for point in controls],float)
    if control_depth[0]>control_depth[-1]:control_depth=control_depth[::-1];control_row=control_row[::-1]
    y=np.interp(depth,control_depth,control_row,left=np.nan,right=np.nan);records=[]
    curves={curve.mnemonic.upper():np.asarray(curve.data,float) for curve in las.curves}
    for track in alignment.get("curve_tracks",[]):
        mnemonic=track["mnemonic"].upper();track_id=str(track.get("track_id") or mnemonic);values=curves[mnemonic];left=float(track["value_left"]);right=float(track["value_right"]);valid=np.isfinite(values)&np.isfinite(y)
        if track.get("scale_type","linear")=="logarithmic":
            valid&=(values>0)&(left>0)&(right>0);fraction=np.log10(values/left)/np.log10(right/left)
        else:fraction=(values-left)/(right-left)
        valid&=(fraction>=0)&(fraction<=1)
        indexes=np.flatnonzero(valid);last_bucket=None
        for index in indexes:
            bucket=round(float(depth[index])/sample_depth_interval) if sample_depth_interval>0 else index
            if bucket==last_bucket:continue
            last_bucket=bucket;records.append({"pair_id":alignment["pair_id"],"track_id":track_id,"mnemonic":mnemonic,"depth":float(depth[index]),"value":float(values[index]),"x":float(track["x_left"]+fraction[index]*(track["x_right"]-track["x_left"])),"y":float(y[index]),"source":"reference_las_projection","alignment_review_status":alignment.get("review_status","automatic_draft"),"color":track.get("color","unknown")})
    return records


def score_color_alignment(image_path: str | Path, records: list[dict], radius: float = 3.0) -> dict:
    import cv2
    image=cv2.imread(str(image_path),cv2.IMREAD_COLOR)
    if image is None:raise ValueError(f"Cannot read image: {image_path}")
    blue,green,red=cv2.split(image);masks={
        "green":(green>100)&(green>red*1.25)&(green>blue*1.15),
        "blue":(blue>100)&(blue>green*1.20)&(blue>red*1.20),
        "red":(red>110)&(red>green*1.25)&(red>blue*1.25),
    };results={}
    for track_id in sorted({record.get("track_id",record["mnemonic"]) for record in records}):
        subset=[record for record in records if record.get("track_id",record["mnemonic"])==track_id and record.get("color") in masks]
        if not subset:continue
        color=subset[0]["color"];distance=cv2.distanceTransform((~masks[color]).astype(np.uint8),cv2.DIST_L2,3);xs=np.clip(np.rint([item["x"] for item in subset]).astype(int),0,image.shape[1]-1);ys=np.clip(np.rint([item["y"] for item in subset]).astype(int),0,image.shape[0]-1);values=distance[ys,xs]
        results[track_id]={"mnemonic":subset[0]["mnemonic"],"point_count":len(subset),"color":color,"hit_fraction_within_radius":float(np.mean(values<=radius)),"radius_pixels":radius,"median_distance_pixels":float(np.median(values)),"p90_distance_pixels":float(np.percentile(values,90))}
    return results


def render_projection_overlay(image_path: str | Path, records: list[dict], output_path: str | Path) -> Path:
    import cv2
    image=cv2.imread(str(image_path),cv2.IMREAD_COLOR)
    if image is None:raise ValueError(f"Cannot read image: {image_path}")
    colors={"green":(255,0,255),"blue":(0,215,255),"red":(255,255,0),"black":(0,165,255),"unknown":(255,0,255)}
    for track_id in sorted({record.get("track_id",record["mnemonic"]) for record in records}):
        subset=sorted((record for record in records if record.get("track_id",record["mnemonic"])==track_id),key=lambda item:item["y"]);segments=[];current=[]
        for record in subset:
            point=(int(round(record["x"])),int(round(record["y"])))
            if current and point[1]-current[-1][1]>3:segments.append(current);current=[]
            current.append(point)
        if current:segments.append(current)
        color=colors.get(subset[0].get("color","unknown"),colors["unknown"])
        for segment in segments:
            if len(segment)>1:cv2.polylines(image,[np.asarray(segment,np.int32)],False,color,1,cv2.LINE_AA)
    output=Path(output_path);output.parent.mkdir(parents=True,exist_ok=True)
    if not cv2.imwrite(str(output),image):raise ValueError(f"Cannot write overlay: {output}")
    return output


def write_pilot_report(output_dir: str | Path, inventory: dict, audits: list[PairAudit]) -> list[Path]:
    output=Path(output_dir);output.mkdir(parents=True,exist_ok=True);inventory_path=output/"dataset_inventory.json";summary_path=output/"pilot_summary.json";manifest_path=output/"pilot_manifest.jsonl"
    _write_json(inventory_path,inventory)
    with manifest_path.open("w",encoding="utf-8",newline="\n") as handle:
        for audit in audits:handle.write(json.dumps(audit.to_dict(),sort_keys=True,default=str)+"\n")
    summary={"pilot_pairs":len(audits),"by_source":dict(Counter(audit.pair.source for audit in audits)),"by_status":dict(Counter(audit.status for audit in audits)),"by_alignment_status":dict(Counter(audit.alignment_status for audit in audits)),"by_split":dict(Counter(audit.split for audit in audits)),"training_eligible":sum(audit.training_eligible for audit in audits),"critical_findings":sum(item["severity"]=="critical" for audit in audits for item in audit.findings),"policy":"No pair becomes training-eligible until its pixel/depth/track alignment is reviewed."}
    _write_json(summary_path,summary);return [inventory_path,manifest_path,summary_path]


def write_alignment_bundle(output_dir: str | Path, audit: PairAudit, alignment: dict) -> list[Path]:
    alignment=deepcopy(alignment);alignment["pair_id"]=audit.pair.pair_id
    alignment["source_files"]={"tiffs":[{"path":item["path"],"content_sha256":item.get("content_sha256"),"stat_fingerprint":item["stat_fingerprint"]} for item in audit.tiff_summaries],"las":{"path":audit.las_summary.get("path"),"content_sha256":audit.las_summary.get("content_sha256"),"stat_fingerprint":audit.las_summary.get("stat_fingerprint")}}
    alignment["alignment_hash"]=_payload_hash(alignment)
    errors=validate_alignment(alignment,audit.las_summary)
    if errors:raise ValueError("; ".join(errors))
    output=Path(output_dir);output.mkdir(parents=True,exist_ok=True);alignment_path=output/"alignment.json";labels_path=output/"labels.jsonl";metrics_path=output/"alignment_metrics.json";overlay_path=output/"overlay.png"
    _write_json(alignment_path,alignment);records=project_las_curves(audit.pair.las_path,alignment)
    with labels_path.open("w",encoding="utf-8",newline="\n") as handle:
        for record in records:handle.write(json.dumps(record,sort_keys=True)+"\n")
    metrics=score_color_alignment(audit.pair.tiff_paths[0],records);_write_json(metrics_path,{"pair_id":audit.pair.pair_id,"review_status":alignment.get("review_status","automatic_draft"),"metrics":metrics});render_projection_overlay(audit.pair.tiff_paths[0],records,overlay_path)
    return [alignment_path,labels_path,metrics_path,overlay_path]


def review_alignment(alignment: dict, audit: PairAudit, reviewer: str, decision: str, notes: str = "") -> dict:
    """Create a reviewed alignment revision; this is the only eligibility gate."""
    if decision not in {"approved","rejected"}:raise ValueError("Alignment decision must be approved or rejected")
    if decision=="approved" and any(item["severity"]=="critical" for item in audit.findings):raise PermissionError("Critical pair-audit findings prevent alignment approval")
    revised=deepcopy(alignment);revised["pair_id"]=audit.pair.pair_id;revised["review_status"]="reviewed_approved" if decision=="approved" else "reviewed_rejected";revised["review_required"]=False;revised["review"]={"reviewer":reviewer,"decision":decision,"reviewed_at":datetime.now(timezone.utc).isoformat(),"notes":notes};revised.pop("alignment_hash",None);revised["alignment_hash"]=_payload_hash(revised);return revised


def alignment_training_eligible(alignment: dict, audit: PairAudit) -> bool:
    if alignment.get("review_status")!="reviewed_approved" or alignment.get("pair_id")!=audit.pair.pair_id:return False
    if alignment.get("alignment_hash")!=_payload_hash(alignment):return False
    if validate_alignment(alignment,audit.las_summary) or any(item["severity"]=="critical" for item in audit.findings):return False
    recorded=alignment.get("source_files",{});recorded_las=recorded.get("las",{})
    if recorded_las.get("content_sha256") and recorded_las.get("content_sha256")!=audit.las_summary.get("content_sha256"):return False
    current_tiffs={item["path"]:item for item in audit.tiff_summaries}
    for item in recorded.get("tiffs",[]):
        current=current_tiffs.get(item.get("path"),{})
        if item.get("content_sha256") and item.get("content_sha256")!=current.get("content_sha256"):return False
    return True


def _tiff_summary(path: Path, content_hash: bool) -> dict:
    from PIL import Image
    previous_limit=Image.MAX_IMAGE_PIXELS
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            Image.MAX_IMAGE_PIXELS=None
            with Image.open(path) as image:width,height=image.size;frames=getattr(image,"n_frames",1);mode=image.mode
        finally:Image.MAX_IMAGE_PIXELS=previous_limit
    stat=path.stat();return {"path":str(path.resolve()),"bytes":stat.st_size,"width":width,"height":height,"frames":frames,"mode":mode,"content_sha256":_sha256(path) if content_hash else None,"stat_fingerprint":_stat_fingerprint(path)}


def _las_summary(path: Path, content_hash: bool) -> dict:
    import lasio
    las=lasio.read(str(path),engine="normal",ignore_header_errors=True);depth=np.asarray(las.index,float);finite=depth[np.isfinite(depth)];difference=np.diff(finite);increasing=bool(len(finite)>1 and np.all(difference>0));decreasing=bool(len(finite)>1 and np.all(difference<0));curves=[]
    for curve in las.curves[1:]:
        try:values=np.asarray(curve.data,float);valid=np.isfinite(values);valid_values=values[valid]
        except Exception:values=np.array([]);valid=np.array([],dtype=bool);valid_values=np.array([])
        curves.append({"mnemonic":curve.mnemonic,"unit":curve.unit or "","description":curve.descr or "","sample_count":int(len(values)),"valid_count":int(valid.sum()),"null_fraction":float(1-valid.mean()) if len(valid) else 1.,"minimum":float(np.min(valid_values)) if len(valid_values) else None,"maximum":float(np.max(valid_values)) if len(valid_values) else None})
    well_item=getattr(las.well,"WELL",None);well_name=str(getattr(well_item,"value","") or "")
    stat=path.stat();return {"path":str(path.resolve()),"bytes":stat.st_size,"content_sha256":_sha256(path) if content_hash else None,"stat_fingerprint":_stat_fingerprint(path),"version":str(getattr(las.version.VERS,"value","")),"wrapped":str(getattr(las.version.WRAP,"value","NO")).upper(),"well_name":well_name,"depth_start":float(finite[0]),"depth_stop":float(finite[-1]),"depth_minimum":float(np.min(finite)),"depth_maximum":float(np.max(finite)),"depth_unit":las.curves[0].unit or "","sample_count":int(len(depth)),"depth_monotonic":increasing or decreasing,"depth_direction":"increasing" if increasing else ("decreasing" if decreasing else "unknown"),"duplicate_depth_count":int(np.sum(np.abs(difference)<=1e-9)),"median_step":float(np.median(difference)) if len(difference) else None,"curves":curves}


def _finding(code,severity,message,source,evidence=None):return {"code":code,"severity":severity,"message":message,"source":source,"evidence":evidence or {}}
def _stable_id(*parts):return hashlib.sha256("|".join(parts).encode()).hexdigest()[:20]
def _stat_fingerprint(path):
    stat=path.stat();return hashlib.sha256(f"{path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}".encode()).hexdigest()
def _sha256(path):
    digest=hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda:handle.read(4*1024*1024),b""):digest.update(chunk)
    return digest.hexdigest()
def _payload_hash(payload):
    normalized={key:value for key,value in payload.items() if key!="alignment_hash"};return hashlib.sha256(json.dumps(normalized,sort_keys=True,default=str).encode()).hexdigest()
def _write_json(path,payload):
    path=Path(path);temporary=path.with_suffix(path.suffix+".tmp");temporary.write_text(json.dumps(payload,indent=2,sort_keys=True,default=str)+"\n",encoding="utf-8");os.replace(temporary,path)


def main(argv=None):
    parser=argparse.ArgumentParser(description="Audit real TIFF/LAS pairs without modifying source data");parser.add_argument("--dataset-root",required=True);parser.add_argument("--output-dir",required=True);parser.add_argument("--kgs-count",type=int,default=16);parser.add_argument("--wvgs-count",type=int,default=4);parser.add_argument("--hash-files",action="store_true");parser.add_argument("--standalone-tiff");parser.add_argument("--standalone-las");parser.add_argument("--standalone-well-id",default="standalone");parser.add_argument("--alignment");parser.add_argument("--approve-alignment",action="store_true");parser.add_argument("--reviewer",default="");parser.add_argument("--review-notes",default="")
    args=parser.parse_args(argv);standalone=[]
    if args.standalone_tiff and args.standalone_las:
        standalone=[PairCandidate(_stable_id("standalone",args.standalone_well_id),"standalone",args.standalone_well_id,(str(Path(args.standalone_tiff).resolve()),),str(Path(args.standalone_las).resolve()),"explicit_user_pair")]
    candidates=discover_pairs(args.dataset_root,standalone);pilot=select_pilot(candidates,args.kgs_count,args.wvgs_count);audits=[audit_pair(item,args.hash_files) for item in pilot];paths=[]
    if args.alignment and standalone:
        alignment=json.loads(Path(args.alignment).read_text(encoding="utf-8"));alignment["pair_id"]=standalone[0].pair_id;audit=next(item for item in audits if item.pair.pair_id==standalone[0].pair_id);audit.alignment_status=alignment.get("review_status","automatic_draft");paths.extend(write_alignment_bundle(Path(args.output_dir)/"standalone_alignment",audit,alignment))
        if args.approve_alignment:
            if not args.reviewer:raise ValueError("--reviewer is required with --approve-alignment")
            bundled=json.loads((Path(args.output_dir)/"standalone_alignment"/"alignment.json").read_text(encoding="utf-8"));reviewed=review_alignment(bundled,audit,args.reviewer,"approved",args.review_notes);reviewed_dir=Path(args.output_dir)/"reviewed_alignment";paths.extend(write_alignment_bundle(reviewed_dir,audit,reviewed));reviewed_payload=json.loads((reviewed_dir/"alignment.json").read_text(encoding="utf-8"));audit.training_eligible=alignment_training_eligible(reviewed_payload,audit);audit.alignment_status=reviewed_payload["review_status"]
    paths.extend(write_pilot_report(args.output_dir,dataset_inventory(args.dataset_root),audits))
    print(json.dumps({"discovered_pairs":len(candidates),"pilot_pairs":len(audits),"outputs":[str(path) for path in paths]},indent=2))


if __name__=="__main__":main()
