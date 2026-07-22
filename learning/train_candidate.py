"""Create a new Phase 4 candidate without mutating the production checkpoint."""
from __future__ import annotations
import argparse, hashlib, importlib.metadata, json, platform, subprocess, uuid
from datetime import datetime, timezone
from pathlib import Path

def _training_sample(dataset_root: Path, sample: dict, torch):
    import cv2, numpy as np
    image=cv2.imread(str(dataset_root/sample["image"]),cv2.IMREAD_COLOR)
    center=cv2.imread(str(dataset_root/sample["centerline_mask"]),cv2.IMREAD_GRAYSCALE)
    if image is None or center is None: raise ValueError(f"Missing training assets for {sample.get('id')}")
    rgb=cv2.cvtColor(image,cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
    center_tensor=torch.from_numpy((center>127).astype(np.float32)[None,None])
    labels=sample.get("labels",{})
    if sample.get("geometry"):
        geometry=np.load(dataset_root/sample["geometry"],allow_pickle=False); distance=geometry["distance_field"]; direction=geometry["direction_field"]; valid_direction=geometry["valid_direction_mask"]; wrap_class=geometry["wrap_event_class_by_row"]
    else:
        distance=np.load(dataset_root/sample["distance_field"],allow_pickle=False); direction=np.load(dataset_root/sample["direction_field"],allow_pickle=False)
        valid_direction=cv2.imread(str(dataset_root/sample["valid_direction_mask"]),cv2.IMREAD_GRAYSCALE)
        wrap_index=np.load(dataset_root/sample["wrap_index"],allow_pickle=False) if sample.get("wrap_index") else np.zeros(image.shape[0],dtype=np.int32)
        wrap_class=np.zeros(wrap_index.size,dtype=np.uint8); change=np.diff(wrap_index); wrap_class[1:][change>0]=1; wrap_class[1:][change<0]=2
        labels={"stroke":bool(sample.get("stroke_mask")),"centerline":True,"grid":bool(sample.get("grid_mask")),"wrap":True}
    wrap=torch.from_numpy(wrap_class.astype(np.int64)[None])
    targets={"centerline_mask":center_tensor,"distance_field":torch.from_numpy(distance[None,None]).float(),
             "direction_field":torch.from_numpy(direction[None]).float(),
             "valid_direction_mask":torch.from_numpy((valid_direction>0).astype(np.float32)[None,None]),
             "wrap_target":wrap,"label_available":{name:torch.tensor([float(labels.get(name,False))]) for name in ("stroke","centerline","grid","wrap")}}
    if labels.get("stroke"):
        stroke=cv2.imread(str(dataset_root/sample["stroke_mask"]),cv2.IMREAD_GRAYSCALE); targets["stroke_mask"]=torch.from_numpy((stroke>127).astype(np.float32)[None,None])
    if labels.get("grid"):
        grid=cv2.imread(str(dataset_root/sample["grid_mask"]),cv2.IMREAD_GRAYSCALE); targets["grid_mask"]=torch.from_numpy((grid>127).astype(np.float32)[None,None])
    return torch.from_numpy(np.transpose(rgb,(2,0,1))[None]).float(),targets

def _optimize(model, datasets_root: Path, dataset_ids: list[str], epochs: int, learning_rate: float, seed: int):
    import torch
    from curve_model.phase4_losses import CurvePhase4Loss
    records=[]
    for dataset_id in dataset_ids:
        directory=datasets_root/dataset_id
        for line in (directory/"samples.jsonl").read_text(encoding="utf-8").splitlines():
            if line.strip(): records.append((dataset_id,directory,json.loads(line)))
    if not records: raise ValueError("Training datasets contain no samples")
    optimizer=torch.optim.AdamW(model.parameters(),lr=float(learning_rate)); criterion=CurvePhase4Loss(); rng=__import__("random").Random(seed); logs=[]
    model.train()
    for epoch in range(int(epochs)):
        rng.shuffle(records); total=0.0; mix={}
        for dataset_id,directory,sample in records:
            image,targets=_training_sample(directory,sample,torch); optimizer.zero_grad(set_to_none=True)
            loss=criterion(model(image),targets,epoch)["total"]; loss.backward(); optimizer.step()
            total+=float(loss.detach()); mix[dataset_id]=mix.get(dataset_id,0)+1
        logs.append({"epoch":epoch,"mean_loss":total/len(records),"source_mix":mix})
    return logs

def _load_dataset(root: Path, dataset_id: str) -> tuple[dict, list[dict]]:
    directory=root/dataset_id; manifest_path=directory/"manifest.json"; samples_path=directory/"samples.jsonl"
    if not manifest_path.exists() or not samples_path.exists(): raise FileNotFoundError(f"Dataset is incomplete: {dataset_id}")
    manifest=json.loads(manifest_path.read_text(encoding="utf-8")); samples=[json.loads(line) for line in samples_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    canonical=json.dumps(samples,sort_keys=True,separators=(",", ":")).encode()
    if manifest.get("dataset_id")!=dataset_id or manifest.get("sample_count")!=len(samples) or manifest.get("content_hash")!=hashlib.sha256(canonical).hexdigest():
        raise ValueError(f"Dataset manifest integrity check failed: {dataset_id}")
    return manifest,samples

def preflight_datasets(datasets_root, training_dataset_ids, golden_dataset_ids):
    from .datasets import leakage_report
    root=Path(datasets_root); training=[]; golden=[]; manifests=[]
    for dataset_id in training_dataset_ids:
        manifest,samples=_load_dataset(root,dataset_id); manifests.append(manifest); training.extend(samples)
    for dataset_id in golden_dataset_ids: golden.extend(_load_dataset(root,dataset_id)[1])
    if not training_dataset_ids: raise ValueError("At least one versioned training dataset is required")
    if not golden_dataset_ids: raise ValueError("At least one frozen golden dataset is required")
    leakage=leakage_report(training,golden)
    if leakage["blocked"]: raise RuntimeError(f"Training/evaluation leakage detected: {leakage}")
    return {"training_manifests":manifests,"golden_dataset_ids":list(golden_dataset_ids),"leakage":leakage}

def create_candidate(base_model, output_dir, model_registry, dataset_ids, architecture="lightweight", seed=41,
                     datasets_root=None, golden_dataset_ids=(), evaluation_report=None, epochs=1, learning_rate=1e-4):
    import torch
    from curve_model.phase4_model import CurvePhase4UNet
    from curve_model.advanced import CurveAdapterModel
    from .model_registry import ModelRegistry
    preflight=preflight_datasets(datasets_root,dataset_ids,golden_dataset_ids) if datasets_root is not None else None
    if datasets_root is not None and evaluation_report is None: raise ValueError("A frozen candidate-versus-production evaluation report is required")
    evaluation=json.loads(Path(evaluation_report).read_text(encoding="utf-8")) if evaluation_report else None
    if evaluation is not None and (not isinstance(evaluation.get("suites"),dict) or not evaluation["suites"]):
        raise ValueError("Evaluation report must contain non-empty frozen-suite results")
    torch.manual_seed(seed); output_dir=Path(output_dir); output_dir.mkdir(parents=True,exist_ok=False)
    model=CurveAdapterModel() if architecture=="advanced" else CurvePhase4UNet()
    transfer=None
    if base_model:
        checkpoint=torch.load(str(base_model),map_location="cpu",weights_only=True); source=checkpoint.get("state_dict",checkpoint)
        compatible={k:v for k,v in source.items() if k in model.state_dict() and model.state_dict()[k].shape==v.shape}
        model.load_state_dict(compatible,strict=False); transfer={"loaded_tensor_count":len(compatible),"base_model":str(base_model)}
    training_log=_optimize(model,Path(datasets_root),dataset_ids,epochs,learning_rate,seed) if datasets_root is not None else []
    checkpoint_path=output_dir/"candidate.pt"
    torch.save({"state_dict":model.state_dict(),"model_config":model.configuration(),"phase":4,"model_version":model.model_version,"seed":seed,"training_dataset_ids":dataset_ids,"base_transfer":transfer},checkpoint_path)
    try: commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()
    except Exception: commit=None
    model_id=f"curve_{architecture}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    dependencies={name:importlib.metadata.version(name) for name in ("numpy","torch","opencv-contrib-python")}
    report={"model_id":model_id,"architecture":architecture,"seed":seed,"dataset_ids":dataset_ids,"code_commit":commit,"python":platform.python_version(),"dependencies":dependencies,"preflight":preflight,"evaluation":evaluation,"trained":bool(training_log),"training_log":training_log,"epochs":int(epochs),"learning_rate":float(learning_rate),"resource_report":getattr(model,"resource_report",lambda:{"parameter_count":sum(p.numel() for p in model.parameters())})()}
    (output_dir/"candidate_report.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    ModelRegistry(model_registry).register_candidate(model_id,checkpoint_path,architecture,dataset_ids,code_commit=commit,training_config=str(output_dir/"candidate_report.json"),metrics=evaluation or {},evaluation_completed=bool(evaluation))
    return report

def main():
    p=argparse.ArgumentParser(); p.add_argument("--base-model",type=Path); p.add_argument("--output",required=True,type=Path); p.add_argument("--registry",type=Path,default=Path("models")); p.add_argument("--datasets-root",required=True,type=Path); p.add_argument("--synthetic-dataset"); p.add_argument("--real-dataset"); p.add_argument("--hard-dataset"); p.add_argument("--golden-dataset",action="append",required=True); p.add_argument("--evaluation-report",required=True,type=Path); p.add_argument("--architecture",choices=("lightweight","advanced"),default="lightweight"); p.add_argument("--seed",type=int,default=41); p.add_argument("--epochs",type=int,default=1); p.add_argument("--learning-rate",type=float,default=1e-4); a=p.parse_args()
    datasets=[item for item in (a.synthetic_dataset,a.real_dataset,a.hard_dataset) if item]
    print(json.dumps(create_candidate(a.base_model,a.output,a.registry,datasets,a.architecture,a.seed,a.datasets_root,a.golden_dataset,a.evaluation_report,a.epochs,a.learning_rate),indent=2))
if __name__=="__main__": main()
