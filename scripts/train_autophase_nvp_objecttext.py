#!/usr/bin/env python3
"""Train the frozen paper-style ObjectText Autophase-NVP anchor."""
from __future__ import annotations
import argparse, gzip, json, math, os, random
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any
import numpy as np
import torch
from torch import nn

K, DIM = 50, 56
_ENV: Any = None

def _init_feature_worker() -> None:
    global _ENV
    import compiler_gym
    _ENV = compiler_gym.make("llvm-v0", reward_space=None)

def _feature(program_id: str) -> tuple[str, list[float]]:
    _ENV.reset(benchmark=program_id)
    raw = np.asarray(_ENV.observation["Autophase"], dtype=np.float32).reshape(-1)
    if raw.size != DIM or raw[51] <= 0:
        raise ValueError(f"invalid Autophase feature for {program_id}: shape={raw.shape}, total={raw[51] if raw.size > 51 else None}")
    return program_id, (raw / raw[51]).tolist()

def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]

def read_label_matrix(shards: Path) -> dict[str, list[dict[str, Any]]]:
    matrix = {}
    for path in sorted(shards.glob("*.json.gz")):
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            payload = json.load(handle)
        summary, records = payload["program_summary"], payload["records"]
        if summary["program_training_target_validity"] == "valid_complete_K50":
            matrix[summary["program_id"]] = sorted(records, key=lambda r: r["candidate_id"])
    return matrix

def extract_features(records: list[dict[str, Any]], workers: int) -> dict[str, list[float]]:
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_feature_worker) as pool:
        values = list(pool.map(_feature, (r["program_id"] for r in records), chunksize=32))
    return dict(values)

class AutophaseNVP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.autophase_nn = nn.Sequential(nn.Linear(DIM, 256), nn.ReLU())
        self.Q = nn.Sequential(nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, K))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.Q(self.autophase_nn(x))

def policy_metrics(logits: torch.Tensor, records: list[dict[str, Any]], matrix: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    by_dataset: dict[str, list[float]] = {}
    regrets: list[int] = []
    for score, target in zip(logits.cpu().tolist(), records):
        candidates = matrix[target["program_id"]]
        budget, observed = 45, []
        for candidate_id in sorted(range(K), key=lambda i: (-score[i], i)):
            prefix = candidates[candidate_id]["prefix_object_text_size_bytes"]
            take = min(budget, len(prefix))
            observed.extend(prefix[:take]); budget -= take
            if budget == 0: break
        if not observed: raise ValueError(f"empty policy rollout: {target['program_id']}")
        policy = min(observed); oracle = min(target["best_object_text_size"]); oz = target["S_Oz"]
        by_dataset.setdefault(target["dataset_id"], []).append((oz-policy)/oz)
        regrets.append(policy-oracle)
    per_dataset = {d: sum(v)/len(v) for d,v in sorted(by_dataset.items())}
    return {"per_dataset": per_dataset, "ValidationFinalMeanOverOz": sum(per_dataset.values())/len(per_dataset), "policy45_regret_mean_bytes": sum(regrets)/len(regrets), "N_total": len(records), "N_primary_valid": len(records), "N_failed_or_invalid": 0}

def main() -> int:
    p=argparse.ArgumentParser(); p.add_argument("--config",type=Path,required=True); p.add_argument("--output-dir",type=Path,required=True); p.add_argument("--train-shards",type=Path,required=True); p.add_argument("--validation-shards",type=Path,required=True); p.add_argument("--workers",type=int,default=12); a=p.parse_args()
    if a.output_dir.exists(): raise FileExistsError(a.output_dir)
    cfg=json.loads(a.config.read_text()); torch.manual_seed(cfg["seed"]); np.random.seed(cfg["seed"]); random.seed(cfg["seed"])
    train=read_jsonl(Path(cfg["target_files"]["train"])); val=read_jsonl(Path(cfg["target_files"]["validation"]))
    if len(train)!=28159 or len(val)!=4488: raise ValueError("unexpected frozen target population")
    train_feat=extract_features(train,a.workers); val_feat=extract_features(val,a.workers)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=AutophaseNVP().to(device); opt=torch.optim.Adam(model.parameters(),lr=cfg["lr"],weight_decay=cfg["weight_decay"])
    train_x=torch.tensor([train_feat[r["program_id"]] for r in train],device=device); train_y=torch.tensor([r["normalized_target"] for r in train],device=device)
    val_x=torch.tensor([val_feat[r["program_id"]] for r in val],device=device); matrix=read_label_matrix(a.validation_shards)
    if set(matrix)!=set(r["program_id"] for r in val): raise ValueError("validation target/label cohort mismatch")
    a.output_dir.mkdir(parents=True); (a.output_dir/"config.json").write_text(json.dumps(cfg,indent=2,sort_keys=True)+"\n")
    curve=[]; best=None; step=0; batch=cfg["batch_size"]; total=cfg["total_steps"]; warmup=500
    while step<total:
        for idx in torch.randperm(len(train),device=device).split(batch):
            step+=1; lr=cfg["lr"]*(step/warmup if step<=warmup else (0.01+0.99*0.5*(1+math.cos(math.pi*(step-warmup)/(total-warmup)))))
            for group in opt.param_groups: group["lr"]=lr
            logits=model(train_x[idx]); loss=-(train_y[idx]*torch.log_softmax(logits,dim=1)).sum(dim=1).mean(); opt.zero_grad(); loss.backward(); opt.step()
            if step%cfg["validation_cadence_steps"]==0 or step==total:
                model.eval()
                with torch.no_grad():
                    val_logits=model(val_x); val_ce=-(torch.tensor([r["normalized_target"] for r in val],device=device)*torch.log_softmax(val_logits,dim=1)).sum(dim=1).mean().item()
                metrics=policy_metrics(val_logits,val,matrix); metrics.update({"step":step,"train_loss":loss.item(),"validation_ce":val_ce,"lr":lr}); curve.append(metrics)
                if best is None or metrics["ValidationFinalMeanOverOz"]>best["ValidationFinalMeanOverOz"]:
                    best=metrics; torch.save({"state_dict":model.state_dict(),"step":step,"metrics":metrics},a.output_dir/"model.pt")
                model.train()
            if step==total: break
    (a.output_dir/"learning_curve.json").write_text(json.dumps(curve,indent=2,sort_keys=True)+"\n")
    report={"step_execution":"COMPLETE","train_programs":len(train),"validation_programs":len(val),"trainable_parameters":sum(x.numel() for x in model.parameters() if x.requires_grad),"selected":best,"selection_metric":"ValidationFinalMeanOverOz policy-45 dataset macro mean","sampling":False,"offline_label_matrix_only":True}
    (a.output_dir/"experiment_report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n"); print(json.dumps(report,indent=2,sort_keys=True)); return 0
if __name__=="__main__": raise SystemExit(main())
