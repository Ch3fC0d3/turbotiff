"""Build and evaluate the first reviewed real-log Phase 1 baseline."""
from __future__ import annotations
import argparse,hashlib,json,os
from pathlib import Path
import cv2,numpy as np
from curve_model.metrics import calculate_trace_metrics
from .real_log_dataset import project_las_curves

DEFAULT_DEPTH_BANDS={"train":(1000.,3000.),"validation":(3000.,4000.),"test":(4000.,5000.)}

def build_curve_crop_dataset(image_path,las_path,alignment,output_dir,track_id="GR_main",crop_height=128,depth_bands=None):
    if alignment.get("review_status")!="reviewed_approved":raise PermissionError("Only a reviewed-approved alignment may generate training data")
    expected=alignment.get("alignment_hash");payload={key:value for key,value in alignment.items() if key!="alignment_hash"};actual=hashlib.sha256(json.dumps(payload,sort_keys=True,default=str).encode()).hexdigest()
    if expected!=actual:raise PermissionError("Alignment content hash is invalid")
    image=cv2.imread(str(image_path),cv2.IMREAD_COLOR)
    if image is None:raise ValueError(f"Cannot read source image: {image_path}")
    records=[item for item in project_las_curves(las_path,alignment,sample_depth_interval=1.) if item.get("track_id")==track_id]
    if not records:raise ValueError(f"No projected labels for track {track_id}")
    track=next(item for item in alignment["curve_tracks"] if (item.get("track_id") or item["mnemonic"])==track_id);left=max(0,int(np.floor(track["x_left"])));right=min(image.shape[1],int(np.ceil(track["x_right"]))+1);bands=depth_bands or DEFAULT_DEPTH_BANDS
    output=Path(output_dir);directories={name:output/name for name in ("images","stroke_masks","centerline_masks","centerlines","metadata")}
    for directory in directories.values():directory.mkdir(parents=True,exist_ok=True)
    manifest=[]
    for split,(depth_start,depth_end) in bands.items():
        subset=[item for item in records if depth_start<=item["depth"]<(depth_end if split!="test" else depth_end+1e-9)]
        if not subset:continue
        first_row=int(np.ceil(min(item["y"] for item in subset)));last_row=int(np.floor(max(item["y"] for item in subset)))+1
        for top in range(first_row,last_row-int(crop_height)+1,int(crop_height)):
            bottom=top+int(crop_height);points=[item for item in subset if top<=item["y"]<bottom]
            point_span=(max(item["y"] for item in points)-min(item["y"] for item in points)) if len(points)>=2 else 0.
            if len(points)<2 or point_span<crop_height*.5:continue
            crop=image[top:bottom,left:right].copy();center=np.zeros(crop.shape[:2],np.uint8);x_by_row=np.full(crop.shape[0],np.nan,np.float32)
            for point in points:
                row=int(np.clip(round(point["y"]-top),0,crop.shape[0]-1));column=int(np.clip(round(point["x"]-left),0,crop.shape[1]-1));center[row,column]=255;x_by_row[row]=float(column)
            valid=np.isfinite(x_by_row)
            if valid.any():
                rows=np.arange(len(x_by_row));x_by_row[~valid]=np.interp(rows[~valid],rows[valid],x_by_row[valid]);center[:]=0;center[rows,np.clip(np.rint(x_by_row).astype(int),0,crop.shape[1]-1)]=255
            stroke=cv2.dilate(center,np.ones((3,3),np.uint8));stem=f"{track_id}_{split}_{top:05d}_{bottom:05d}";paths={"image":directories["images"]/f"{stem}.png","stroke_mask":directories["stroke_masks"]/f"{stem}.png","centerline_mask":directories["centerline_masks"]/f"{stem}.png","centerline_x":directories["centerlines"]/f"{stem}.npy","metadata":directories["metadata"]/f"{stem}.json"}
            cv2.imwrite(str(paths["image"]),crop);cv2.imwrite(str(paths["stroke_mask"]),stroke);cv2.imwrite(str(paths["centerline_mask"]),center);np.save(paths["centerline_x"],x_by_row,allow_pickle=False)
            metadata={"pair_id":alignment["pair_id"],"alignment_hash":alignment["alignment_hash"],"track_id":track_id,"mnemonic":track["mnemonic"],"split":split,"source_bounds":{"left":left,"right":right,"top":top,"bottom":bottom},"depth_range":[float(min(item["depth"] for item in points)),float(max(item["depth"] for item in points))]};paths["metadata"].write_text(json.dumps(metadata,indent=2,sort_keys=True)+"\n",encoding="utf-8")
            record={"id":stem,"source":"reviewed_real_log","split":split,"well_id":alignment["pair_id"],"track_id":track_id,"curve_color":track.get("color","unknown"),"curve_unit":track.get("unit",""),"curve_value_span":abs(float(track["value_right"])-float(track["value_left"])),"hard_case":False,**{key:str(path.relative_to(output)).replace("\\","/") for key,path in paths.items()}};manifest.append(record)
    manifest_path=output/"manifest.jsonl"
    with manifest_path.open("w",encoding="utf-8",newline="\n") as handle:
        for record in manifest:handle.write(json.dumps(record,sort_keys=True)+"\n")
    summary={"samples":len(manifest),"by_split":{split:sum(item["split"]==split for item in manifest) for split in bands},"track_id":track_id,"alignment_hash":alignment["alignment_hash"],"manifest_sha256":hashlib.sha256(manifest_path.read_bytes()).hexdigest(),"holdout_scope":"Depth-block diagnostic within one well; not a cross-well generalization estimate."};(output/"dataset_summary.json").write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n",encoding="utf-8");return summary

def evaluate_checkpoint(data_dir,checkpoint_path,split="test",device="cpu"):
    import torch
    from curve_model.dataset import SyntheticCurveDataset,load_manifest
    from curve_model.model import CurvePhase1UNet
    root=Path(data_dir);all_records=load_manifest(root);records=[item for item in all_records if item.get("split")==split];checkpoint=torch.load(str(checkpoint_path),map_location=device,weights_only=True);config=checkpoint["training_config"];dataset=SyntheticCurveDataset(root,records,tuple(config["target_size"]));model=CurvePhase1UNet(**{key:checkpoint["model_config"][key] for key in ("in_channels","base_channels")}).to(device);model.load_state_dict(checkpoint["state_dict"]);model.eval();predicted=[];truth=[];color_predicted=[];train_x=[]
    train_dataset=SyntheticCurveDataset(root,[item for item in all_records if item.get("split")=="train"],tuple(config["target_size"]))
    for sample in train_dataset:
        mask=sample["centerline_mask"][0].numpy()>0;rows=mask.sum(1)>0;train_x.extend(np.argmax(mask,axis=1)[rows].tolist())
    with torch.no_grad():
        for record,sample in zip(records,dataset):
            output=model(sample["image"].unsqueeze(0).to(device));pred=output["centerline_logits"][0,0].argmax(dim=1).cpu().numpy().astype(float);mask=sample["centerline_mask"][0].numpy()>0;rows=mask.sum(1)>0;target=np.argmax(mask,axis=1).astype(float);pred[~rows]=np.nan;target[~rows]=np.nan;predicted.extend(pred.tolist());truth.extend(target.tolist())
            rgb=sample["image"].numpy().transpose(1,2,0);color=str(record.get("curve_color","green")).lower()
            if color=="blue":color_mask=(rgb[:,:,2]>.39)&(rgb[:,:,2]>rgb[:,:,1]*1.18)&(rgb[:,:,2]>rgb[:,:,0]*1.18)
            elif color=="red":color_mask=(rgb[:,:,0]>.43)&(rgb[:,:,0]>rgb[:,:,1]*1.25)&(rgb[:,:,0]>rgb[:,:,2]*1.25)
            elif color=="black":color_mask=np.mean(rgb,axis=2)<.45
            else:color_mask=(rgb[:,:,1]>.39)&(rgb[:,:,1]>rgb[:,:,0]*1.25)&(rgb[:,:,1]>rgb[:,:,2]*1.15)
            color_x=np.full(color_mask.shape[0],np.nan)
            for row in range(color_mask.shape[0]):
                columns=np.flatnonzero(color_mask[row])
                if len(columns):color_x[row]=float(np.median(columns))
            color_x[~rows]=np.nan;color_predicted.extend(color_x.tolist())
    predicted=np.asarray(predicted);truth=np.asarray(truth);color_predicted=np.asarray(color_predicted);model_metrics=calculate_trace_metrics(predicted,truth);color_metrics=calculate_trace_metrics(color_predicted,truth)
    constant_metrics=None
    if train_x:
        constant=np.full(truth.shape,float(np.mean(train_x)));constant[~np.isfinite(truth)]=np.nan;constant_metrics=calculate_trace_metrics(constant,truth)
    spans=[float(item["curve_value_span"]) for item in records if item.get("curve_value_span") is not None];units_per_pixel=(float(np.median(spans))/config["target_size"][1]) if spans else None;units=sorted({str(item.get("curve_unit") or "") for item in records})
    return {"split":split,"samples":len(records),"wells":sorted({item.get("well_id") for item in records}),"model":model_metrics,"raster_color_baseline":color_metrics,"raster_colors":sorted({str(item.get("curve_color","unknown")) for item in records}),"constant_training_mean_baseline":constant_metrics,"approximate_curve_unit_mae":model_metrics["mean_absolute_error"]*units_per_pixel if model_metrics["mean_absolute_error"] is not None and units_per_pixel is not None else None,"curve_units":units,"scope":"Evaluation on the requested explicit dataset split"}

def write_model_provenance(output_dir,dataset_dir,alignment,training_summary,evaluation):
    output=Path(output_dir);manifest=Path(dataset_dir)/"manifest.jsonl";payload={"model_purpose":"first reviewed real-log diagnostic baseline","pair_id":alignment["pair_id"],"alignment_hash":alignment["alignment_hash"],"source_files":alignment.get("source_files",{}),"dataset_manifest_sha256":hashlib.sha256(manifest.read_bytes()).hexdigest(),"training_summary":training_summary,"evaluation":evaluation,"limitations":["One approved well only","Depth-block holdout is not independent-well validation","GR track only"]};path=output/"model_provenance.json";path.write_text(json.dumps(payload,indent=2,sort_keys=True,default=str)+"\n",encoding="utf-8");return path

def main(argv=None):
    parser=argparse.ArgumentParser(description="Train a reviewed real-log GR diagnostic baseline");parser.add_argument("--image",required=True);parser.add_argument("--las",required=True);parser.add_argument("--alignment",required=True);parser.add_argument("--output-dir",required=True);parser.add_argument("--track-id",default="GR_main");parser.add_argument("--dataset-split",choices=("train","validation","test"));parser.add_argument("--checkpoint");parser.add_argument("--epochs",type=int,default=12);parser.add_argument("--device",default="cpu");args=parser.parse_args(argv)
    from curve_model.train import train_phase1
    output=Path(args.output_dir);dataset_dir=output/"dataset";model_dir=output/"model";alignment=json.loads(Path(args.alignment).read_text(encoding="utf-8"));depth_bands=None
    if args.dataset_split:
        depths=[float(point["depth"]) for point in alignment["depth_control_points"]];depth_bands={args.dataset_split:(min(depths),max(depths))}
    dataset_summary=build_curve_crop_dataset(args.image,args.las,alignment,dataset_dir,track_id=args.track_id,depth_bands=depth_bands)
    if args.dataset_split and args.dataset_split!="train":
        result={"dataset":dataset_summary,"training_skipped":True,"reason":"A validation/test-only export cannot train a model by itself."}
        if args.checkpoint:
            result["evaluation"]=evaluate_checkpoint(dataset_dir,args.checkpoint,split=args.dataset_split,device=args.device);(output/"checkpoint_evaluation.json").write_text(json.dumps(result["evaluation"],indent=2,sort_keys=True,default=str)+"\n",encoding="utf-8")
        print(json.dumps(result,indent=2,default=str));return
    training=train_phase1(dataset_dir,model_dir,epochs=args.epochs,batch_size=4,learning_rate=1e-3,seed=20260720,target_size=(128,256),base_channels=4,device=args.device);evaluation=evaluate_checkpoint(dataset_dir,model_dir/"best.pt",device=args.device);provenance=write_model_provenance(model_dir,dataset_dir,alignment,training,evaluation);print(json.dumps({"dataset":dataset_summary,"training":training,"evaluation":evaluation,"provenance":str(provenance)},indent=2,default=str))

if __name__=="__main__":main()
