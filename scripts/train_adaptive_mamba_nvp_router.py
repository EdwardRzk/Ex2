#!/usr/bin/env python3
"""Train and evaluate the one frozen-expert Program-Adaptive Mamba-NVP Router."""
from __future__ import annotations

import argparse
import collections
import copy
import gzip
import json
import math
import random
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn

if __package__:
    from scripts.evaluate_mamba_nvp_final_objecttext import aggregate, load_final_features, read_final_artifacts, read_results, regret_summary
    from scripts.train_controlled_nvp_stage_a import ControlledCandidateModel, load_candidates, read_jsonl, read_label_matrix
    from scripts.train_mamba_nvp_objecttext import load_feature_cache, load_frozen_nvp
else:
    from evaluate_mamba_nvp_final_objecttext import aggregate, load_final_features, read_final_artifacts, read_results, regret_summary
    from train_controlled_nvp_stage_a import ControlledCandidateModel, load_candidates, read_jsonl, read_label_matrix
    from train_mamba_nvp_objecttext import load_feature_cache, load_frozen_nvp


K, FEATURE_DIM, SUMMARY_DIM = 50, 56, 9


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_config(cfg: Mapping[str, Any]) -> None:
    if cfg["training"]["final_seed_set"] != [1, 2, 3]:
        raise ValueError("AMR requires exactly frozen seeds [1, 2, 3]")
    if cfg["candidate_representation"] != {"candidate_sequences": "configs/rlcompopt_action_seq_50.txt", "K": 50, "padded_length": 20, "pad_token_id": 124}:
        raise ValueError("frozen K50 candidate representation mismatch")
    if cfg["router"]["input_dimension"] != FEATURE_DIM + 3 * K + SUMMARY_DIM or cfg["router"]["layers"] != [128, 64, 1] or cfg["router"]["dropout"] != 0.1:
        raise ValueError("AMR router architecture must remain frozen")
    if cfg["routing_supervision"]["lambda_route"] != 0.25:
        raise ValueError("AMR route-loss coefficient must remain 0.25")
    if cfg["training"]["total_steps"] != 10000 or cfg["training"]["batch_size"] != 256 or cfg["training"]["checkpoint_evaluation_cadence_steps"] != 100:
        raise ValueError("AMR training budget must remain frozen")
    if cfg["threshold_selection"]["tau_values"] != [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        raise ValueError("AMR threshold list must remain frozen")


def seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


class ProgramAdaptiveRouter(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(FEATURE_DIM + 3 * K + SUMMARY_DIM, 128), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.network(inputs)).squeeze(-1)


def distribution_features(program: torch.Tensor, mamba_logits: torch.Tensor, nvp_logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    p_m, p_n = torch.softmax(mamba_logits, dim=1), torch.softmax(nvp_logits, dim=1)
    eps = torch.finfo(p_m.dtype).tiny
    entropy_m = -(p_m * p_m.clamp_min(eps).log()).sum(dim=1, keepdim=True)
    entropy_n = -(p_n * p_n.clamp_min(eps).log()).sum(dim=1, keepdim=True)
    top_m, top_n = p_m.topk(2, dim=1).values, p_n.topk(2, dim=1).values
    midpoint = 0.5 * (p_m + p_n)
    js = 0.5 * ((p_m * (p_m.clamp_min(eps).log() - midpoint.clamp_min(eps).log())).sum(dim=1, keepdim=True) + (p_n * (p_n.clamp_min(eps).log() - midpoint.clamp_min(eps).log())).sum(dim=1, keepdim=True))
    l1 = (p_m - p_n).abs().sum(dim=1, keepdim=True)
    agreement = (p_m.argmax(dim=1) == p_n.argmax(dim=1)).to(p_m.dtype).unsqueeze(1)
    summary = torch.cat((entropy_m, entropy_n, top_m[:, :1], top_n[:, :1], top_m[:, :1] - top_m[:, 1:], top_n[:, :1] - top_n[:, 1:], js, l1, agreement), dim=1)
    inputs = torch.cat((program, p_m, p_n, (p_m - p_n).abs(), summary), dim=1)
    if inputs.shape[1] != FEATURE_DIM + 3 * K + SUMMARY_DIM:
        raise RuntimeError("unexpected AMR router input width")
    return inputs, p_m, p_n


def thresholded_mixture(p_m: torch.Tensor, p_n: torch.Tensor, alpha: torch.Tensor, tau: float) -> tuple[torch.Tensor, torch.Tensor]:
    if tau == 1.0:
        alpha_eff = torch.zeros_like(alpha)
    else:
        alpha_eff = torch.where(alpha <= tau, torch.zeros_like(alpha), (alpha - tau) / (1.0 - tau))
    probabilities = (1.0 - alpha_eff.unsqueeze(1)) * p_m + alpha_eff.unsqueeze(1) * p_n
    return probabilities, alpha_eff


def policy45(scores: Sequence[float], records: Sequence[Mapping[str, Any]]) -> int:
    budget, observed = 45, []
    for candidate_id in sorted(range(K), key=lambda index: (-float(scores[index]), index)):
        prefix = records[candidate_id]["prefix_object_text_size_bytes"]
        take = min(budget, len(prefix)); observed.extend(prefix[:take]); budget -= take
        if budget == 0:
            break
    if budget != 0 or not observed:
        raise ValueError("policy45 did not consume exactly 45 frozen prefix measurements")
    return min(observed)


def policy45_utility(scores: Sequence[float], record: Mapping[str, Any], matrix: Mapping[str, Sequence[Mapping[str, Any]]]) -> float:
    oz = int(record["S_Oz"]); policy = policy45(scores, matrix[str(record["program_id"])])
    return (oz - policy) / oz


class SourceBalancedSampler:
    def __init__(self, dataset_ids: Sequence[str], seed: int) -> None:
        grouped: dict[str, list[int]] = collections.defaultdict(list)
        for index, dataset in enumerate(dataset_ids):
            grouped[str(dataset)].append(index)
        self.sources = sorted(grouped)
        if not self.sources or any(not grouped[source] for source in self.sources):
            raise ValueError("source-balanced sampler requires nonempty explicit sources")
        self.indices = [torch.tensor(grouped[source], dtype=torch.long) for source in self.sources]
        self.generator = torch.Generator().manual_seed(seed)

    def sample(self, batch_size: int, device: torch.device) -> torch.Tensor:
        source_ids = torch.randint(len(self.sources), (batch_size,), generator=self.generator)
        result = torch.empty(batch_size, dtype=torch.long)
        for source in source_ids.unique().tolist():
            mask = source_ids == source; pool = self.indices[source]
            result[mask] = pool[torch.randint(len(pool), (int(mask.sum()),), generator=self.generator)]
        return result.to(device, non_blocking=True)


def load_frozen_mamba(path: Path, seed: int, controlled: Mapping[str, Any], tokens: torch.Tensor, lengths: torch.Tensor) -> ControlledCandidateModel:
    payload = torch.load(path, map_location="cpu")
    if payload.get("stage") != "Route-A Stage B" or payload.get("architecture") != "Mamba" or payload.get("seed") != seed or payload.get("stage_a_checkpoint_reused") is not False:
        raise ValueError(f"not frozen Stage-B Mamba seed {seed}: {path}")
    model_cfg = {**controlled["candidate_representation"], **controlled["models"]["Mamba"]}
    if payload.get("model_config") != controlled["models"]["Mamba"]:
        raise ValueError("frozen Mamba checkpoint model configuration mismatch")
    model = ControlledCandidateModel("Mamba", model_cfg, tokens, lengths)
    model.load_state_dict(payload["state_dict"], strict=True)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model.eval()


def expert_logits(mamba: nn.Module, nvp: nn.Module, features: torch.Tensor, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    mamba_rows, nvp_rows = [], []
    with torch.no_grad():
        for start in range(0, len(features), batch_size):
            current = features[start:start + batch_size]
            mamba_rows.append(mamba(current)); nvp_rows.append(nvp(current))
    return torch.cat(mamba_rows), torch.cat(nvp_rows)


def metrics_for_scores(scores: torch.Tensor, targets: torch.Tensor, records: Sequence[Mapping[str, Any]], matrix: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    by_dataset: dict[str, list[float]] = collections.defaultdict(list); regrets, positives, correct = [], 0, 0
    for values, target, record in zip(scores.tolist(), targets.tolist(), records):
        policy = policy45(values, matrix[str(record["program_id"])])
        oracle = min(item["best_object_text_size_bytes"] for item in matrix[str(record["program_id"])])
        oz = int(record["S_Oz"]); reduction = (oz - policy) / oz
        by_dataset[str(record["dataset_id"])].append(reduction); regrets.append(policy - oracle); positives += int(reduction > 0)
        best = min(item["best_object_text_size_bytes"] for item in matrix[str(record["program_id"])])
        chosen = int(np.argmax(values)); correct += int(matrix[str(record["program_id"])][chosen]["best_object_text_size_bytes"] == best)
    per_dataset = {name: sum(values) / len(values) for name, values in sorted(by_dataset.items())}
    return {"per_dataset": per_dataset, "ValidationFinalMeanOverOz": sum(per_dataset.values()) / len(per_dataset), "policy45_regret_mean_bytes": float(np.mean(regrets)), "policy45_regret_median_bytes": float(np.median(regrets)), "positive_program_count_vs_Oz": positives, "top1_oracle_tie_accuracy": correct / len(records), "N_total": len(records), "N_primary_valid": len(records), "N_failed_or_invalid": 0}


def router_validation(router: ProgramAdaptiveRouter, inputs: torch.Tensor, p_m: torch.Tensor, p_n: torch.Tensor, targets: torch.Tensor, route_targets: torch.Tensor, records: Sequence[Mapping[str, Any]], matrix: Mapping[str, Sequence[Mapping[str, Any]]], batch_size: int) -> dict[str, Any]:
    router.eval(); alpha_parts = []
    with torch.no_grad():
        for start in range(0, len(inputs), batch_size): alpha_parts.append(router(inputs[start:start + batch_size]))
    alpha = torch.cat(alpha_parts); mixture, _ = thresholded_mixture(p_m, p_n, alpha, 0.0)
    task = -(targets * mixture.clamp_min(torch.finfo(mixture.dtype).tiny).log()).sum(dim=1).mean()
    route = nn.functional.binary_cross_entropy(alpha, route_targets)
    result = metrics_for_scores(mixture.cpu(), targets.cpu(), records, matrix)
    result.update({"validation_task_loss": float(task.cpu()), "validation_route_bce": float(route.cpu()), "average_alpha": float(alpha.mean().cpu())})
    return result


def checkpoint_payload(router: ProgramAdaptiveRouter, seed: int, step: int, metrics: Mapping[str, Any], advantage_scale: float) -> dict[str, Any]:
    return {"stage": "Program-Adaptive Mamba-NVP Expert Router v1", "architecture": "AMR", "seed": seed, "step": step, "metrics": dict(metrics), "state_dict": router.state_dict(), "advantage_scale": advantage_scale, "experts_frozen": True, "expert_mixing": "probability mixture only", "dataset_id_router_input": False}


def train_seed(cfg: Mapping[str, Any], seed: int, train_x: torch.Tensor, train_targets: torch.Tensor, train_inputs: torch.Tensor, train_pm: torch.Tensor, train_pn: torch.Tensor, train_route: torch.Tensor, train_datasets: Sequence[str], val_x: torch.Tensor, val_targets: torch.Tensor, val_inputs: torch.Tensor, val_pm: torch.Tensor, val_pn: torch.Tensor, val_route: torch.Tensor, validation: Sequence[Mapping[str, Any]], matrix: Mapping[str, Sequence[Mapping[str, Any]]], advantage_scale: float, output: Path, device: torch.device) -> dict[str, Any]:
    seed_everything(seed); output.mkdir(parents=True)
    router = ProgramAdaptiveRouter().to(device); optimizer = torch.optim.Adam(router.parameters(), lr=float(cfg["training"]["learning_rate"]), weight_decay=float(cfg["training"]["weight_decay"]))
    sampler = SourceBalancedSampler(train_datasets, seed); curve, best = [], None
    total, cadence, batch = int(cfg["training"]["total_steps"]), int(cfg["training"]["checkpoint_evaluation_cadence_steps"]), int(cfg["training"]["batch_size"])
    for step in range(1, total + 1):
        index = sampler.sample(batch, device); warmup, base = int(cfg["training"]["warmup_steps"]), float(cfg["training"]["learning_rate"])
        lr = base * (step / warmup if step <= warmup else (0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * (step - warmup) / (total - warmup)))))
        for group in optimizer.param_groups: group["lr"] = lr
        router.train(); alpha = router(train_inputs[index]); mix = (1-alpha.unsqueeze(1))*train_pm[index] + alpha.unsqueeze(1)*train_pn[index]
        task_loss = -(train_targets[index] * mix.clamp_min(torch.finfo(mix.dtype).tiny).log()).sum(dim=1).mean(); route_loss = nn.functional.binary_cross_entropy(alpha, train_route[index]); loss = task_loss + 0.25 * route_loss
        optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
        if step % cadence == 0 or step == total:
            metric = router_validation(router, val_inputs, val_pm, val_pn, val_targets, val_route, validation, matrix, int(cfg["training"]["evaluation_batch_size"]))
            metric.update({"step": step, "train_task_loss": float(task_loss.detach().cpu()), "train_route_bce": float(route_loss.detach().cpu()), "train_total_loss": float(loss.detach().cpu()), "lr": lr})
            curve.append(metric); print(json.dumps({"architecture":"AMR","seed":seed,**metric},sort_keys=True),flush=True)
            if best is None or metric["ValidationFinalMeanOverOz"] > best["ValidationFinalMeanOverOz"]:
                best = metric; torch.save(checkpoint_payload(router, seed, step, metric, advantage_scale), output / "model.pt")
    if best is None: raise RuntimeError("AMR produced no validation checkpoint")
    (output / "learning_curve.json").write_text(json.dumps(curve,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    return {"architecture":"AMR","seed":seed,"step_execution":"COMPLETE","trainable_parameters":sum(p.numel() for p in router.parameters()),"selection_metric":"unthresholded P_mix validation policy45 dataset-macro MeanOverOz","selected":best,"source_counts":dict(sorted(collections.Counter(train_datasets).items()))}


def evaluate_taus(reports: Sequence[Mapping[str, Any]], routers: Mapping[int, ProgramAdaptiveRouter], val_inputs: torch.Tensor, val_pm: torch.Tensor, val_pn: torch.Tensor, val_targets: torch.Tensor, validation: Sequence[Mapping[str, Any]], matrix: Mapping[str, Sequence[Mapping[str, Any]]], cfg: Mapping[str, Any]) -> tuple[dict[str, Any], float | None]:
    per_tau: dict[str, Any] = {}; nvp_validation = 0.06277100953471096
    for tau in cfg["threshold_selection"]["tau_values"]:
        seed_metrics=[]
        for seed in (1,2,3):
            router=routers[seed].eval()
            with torch.no_grad(): alpha=router(val_inputs); scores,_=thresholded_mixture(val_pm,val_pn,alpha,float(tau))
            seed_metrics.append(metrics_for_scores(scores.cpu(),val_targets.cpu(),validation,matrix))
        per_dataset={name:sum(item["per_dataset"][name] for item in seed_metrics)/3 for name in sorted(seed_metrics[0]["per_dataset"])}
        mean=sum(item["ValidationFinalMeanOverOz"] for item in seed_metrics)/3; nvp_per={name:next(x for x in load_json(Path("outputs/route_a_stage_b_v6/comparison_report.json"))["models"] if x["architecture"]=="NVP")["per_dataset_3seed"][name] for name in per_dataset}
        deltas={name:per_dataset[name]-nvp_per[name] for name in per_dataset}; median=float(np.median(list(deltas.values())))
        per_tau[str(tau)]={"three_seed_validation_mean_over_oz":mean,"per_seed":{str(seed):item["ValidationFinalMeanOverOz"] for seed,item in zip((1,2,3),seed_metrics)},"per_dataset":per_dataset,"delta_vs_nvp_per_dataset":deltas,"macro_delta_vs_nvp":mean-nvp_validation,"median_dataset_delta_vs_nvp":median,"positive_dataset_count":sum(value>0 for value in deltas.values()),"negative_dataset_count":sum(value<0 for value in deltas.values()),"S_val":mean-nvp_validation+0.5*median,"seed_metrics":seed_metrics}
    eligible=[(float(tau),row) for tau,row in per_tau.items() if row["three_seed_validation_mean_over_oz"] > float(cfg["threshold_selection"]["high_bar_validation_mean_over_oz"])]
    selected=None if not eligible else max(eligible,key=lambda item:(item[1]["S_val"],item[1]["positive_dataset_count"],item[1]["three_seed_validation_mean_over_oz"]))[0]
    return per_tau,selected


def load_router(path: Path, seed: int, device: torch.device) -> ProgramAdaptiveRouter:
    payload=torch.load(path,map_location="cpu")
    if payload.get("stage")!="Program-Adaptive Mamba-NVP Expert Router v1" or payload.get("architecture")!="AMR" or payload.get("seed")!=seed or payload.get("experts_frozen") is not True or payload.get("dataset_id_router_input") is not False:
        raise ValueError(f"invalid frozen AMR checkpoint: {path}")
    model=ProgramAdaptiveRouter(); model.load_state_dict(payload["state_dict"],strict=True); return model.to(device).eval()


def final_rows(seed: int, router: ProgramAdaptiveRouter, p_m: torch.Tensor, p_n: torch.Tensor, inputs: torch.Tensor, programs: Sequence[str], eligible: Sequence[str], matrix: Mapping[str, Sequence[Mapping[str, Any]]], summaries: Mapping[str, Mapping[str, Any]], tau: float, output: Path) -> dict[str, Any]:
    with torch.no_grad(): alpha=router(inputs); scores,alpha_eff=thresholded_mixture(p_m,p_n,alpha,tau)
    positions = {program: index for index, program in enumerate(eligible)}
    failures=collections.Counter(); per_dataset=collections.defaultdict(collections.Counter); output.parent.mkdir(parents=True,exist_ok=True)
    with gzip.open(output,"wt",encoding="utf-8") as handle:
        for index,program in enumerate(programs):
            summary=summaries[program]; row={"program_id":program,"dataset_id":summary["dataset_id"],"model":"AMR","seed":seed,"valid":False}
            if program not in matrix: reason="incomplete_K50"
            elif summary["ratio_metric_validity"]!="valid_for_ObjectText_ratio_metric": reason="invalid_ratio_denominator"
            else:
                position = positions[program]; policy=policy45(scores[position].cpu().tolist(),matrix[program]); oracle=min(x["best_object_text_size_bytes"] for x in matrix[program]); oz=int(summary["oz_object_text_size_bytes"]); choice=int(scores[position].argmax().cpu())
                row.update({"valid":True,"selected_candidate_id":choice,"alpha":float(alpha[position].cpu()),"alpha_eff":float(alpha_eff[position].cpu()),"policy45_object_text_size_bytes":policy,"oracle_object_text_size_bytes":oracle,"oz_object_text_size_bytes":oz,"mean_over_oz":(oz-policy)/oz,"policy45_regret_bytes":policy-oracle}); per_dataset[summary["dataset_id"]]["N_primary_valid"]+=1; handle.write(json.dumps(row,separators=(",",":"))+"\n"); continue
            row["failure_reason"]=reason; failures[reason]+=1; per_dataset[summary["dataset_id"]]["N_failed_or_invalid"]+=1; handle.write(json.dumps(row,separators=(",",":"))+"\n")
    return {"seed":seed,"result_file":str(output),"average_alpha":float(alpha.mean().cpu()),"average_alpha_eff":float(alpha_eff.mean().cpu()),"failure_count_by_reason":dict(failures),"per_dataset_method_validity":{d:dict(v) for d,v in sorted(per_dataset.items())}}


def final_top1_oracle_tie_accuracy(result_maps: Mapping[tuple[str, int], Mapping[str, Mapping[str, Any]]], matrix: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    """Derive the optional frozen top-1 diagnostic from selected candidate IDs only."""
    per_seed: dict[str, float] = {}
    for seed in (1, 2, 3):
        rows = result_maps[("AMR", seed)]
        valid = [row for program, row in rows.items() if row["valid"] and program in matrix]
        if not valid:
            raise ValueError("empty final AMR top1 cohort")
        correct = sum(matrix[row["program_id"]][int(row["selected_candidate_id"])]["best_object_text_size_bytes"] == min(item["best_object_text_size_bytes"] for item in matrix[row["program_id"]]) for row in valid)
        per_seed[str(seed)] = correct / len(valid)
    return {"per_seed": per_seed, "three_seed_mean": sum(per_seed.values()) / 3}


def main() -> int:
    parser=argparse.ArgumentParser(); parser.add_argument("--config",type=Path,required=True); parser.add_argument("--output-dir",type=Path,required=True); args=parser.parse_args()
    if args.output_dir.exists(): raise FileExistsError(f"refusing to overwrite existing output directory: {args.output_dir}")
    cfg=load_json(args.config); validate_config(cfg)
    if not torch.cuda.is_available(): raise RuntimeError("AMR training requires CUDA")
    device=torch.device("cuda"); controlled=load_json(Path(cfg["frozen_experts"]["controlled_config"])); candidate=cfg["candidate_representation"]
    train,validation=read_jsonl(Path(cfg["target_files"]["train"])),read_jsonl(Path(cfg["target_files"]["validation"]))
    if len(train)!=28159 or len(validation)!=4488: raise ValueError("frozen train/validation target population mismatch")
    train_matrix,validation_matrix=read_label_matrix(Path(cfg["label_shards"]["train"])),read_label_matrix(Path(cfg["label_shards"]["validation"]))
    if set(train_matrix)!={row["program_id"] for row in train} or set(validation_matrix)!={row["program_id"] for row in validation}: raise ValueError("frozen K50 matrix population mismatch")
    train_features=load_feature_cache(Path(cfg["autophase_feature_cache"]["train"]),"train",[row["program_id"] for row in train]); val_features=load_feature_cache(Path(cfg["autophase_feature_cache"]["validation"]),"validation",[row["program_id"] for row in validation])
    train_x=torch.tensor([train_features[row["program_id"]] for row in train],dtype=torch.float32,device=device); val_x=torch.tensor([val_features[row["program_id"]] for row in validation],dtype=torch.float32,device=device)
    train_q=torch.tensor([row["normalized_target"] for row in train],dtype=torch.float32,device=device); val_q=torch.tensor([row["normalized_target"] for row in validation],dtype=torch.float32,device=device)
    tokens,lengths=load_candidates(Path(candidate["candidate_sequences"]),pad_token_id=124,padded_length=20); tokens,lengths=tokens.to(device),lengths.to(device)
    args.output_dir.mkdir(parents=True); (args.output_dir/"config.json").write_text(json.dumps(cfg,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    reports=[]; routers={}; validation_expert_checks={}
    for seed in (1,2,3):
        nvp=load_frozen_nvp(Path(cfg["frozen_experts"]["nvp_checkpoint_root"])/f"seed{seed}"/"model.pt",seed).to(device).eval(); mamba=load_frozen_mamba(Path(cfg["frozen_experts"]["mamba_checkpoint_root"])/f"seed{seed}"/"model.pt",seed,controlled,tokens,lengths).to(device).eval()
        train_m,train_n=expert_logits(mamba,nvp,train_x,int(cfg["training"]["evaluation_batch_size"])); val_m,val_n=expert_logits(mamba,nvp,val_x,int(cfg["training"]["evaluation_batch_size"])); train_inputs,train_pm,train_pn=distribution_features(train_x,train_m,train_n); val_inputs,val_pm,val_pn=distribution_features(val_x,val_m,val_n)
        advantages=torch.tensor([policy45_utility(n.tolist(),record,train_matrix)-policy45_utility(m.tolist(),record,train_matrix) for m,n,record in zip(train_pm.cpu(),train_pn.cpu(),train)],dtype=torch.float32,device=device)
        nonzero=advantages.abs()[advantages!=0]
        if len(nonzero)==0: raise RuntimeError("AMR route advantage has no nonzero training values")
        scale=float(nonzero.median().cpu()); train_route=torch.sigmoid(advantages/scale)
        val_adv=torch.tensor([policy45_utility(n.tolist(),record,validation_matrix)-policy45_utility(m.tolist(),record,validation_matrix) for m,n,record in zip(val_pm.cpu(),val_pn.cpu(),validation)],dtype=torch.float32,device=device); val_route=torch.sigmoid(val_adv/scale)
        validation_expert_checks[str(seed)]={"mamba_policy45":metrics_for_scores(val_pm.cpu(),val_q.cpu(),validation,validation_matrix)["ValidationFinalMeanOverOz"],"nvp_policy45":metrics_for_scores(val_pn.cpu(),val_q.cpu(),validation,validation_matrix)["ValidationFinalMeanOverOz"],"advantage_scale":scale}
        report=train_seed(cfg,seed,train_x,train_q,train_inputs,train_pm,train_pn,train_route,[row["dataset_id"] for row in train],val_x,val_q,val_inputs,val_pm,val_pn,val_route,validation,validation_matrix,scale,args.output_dir/"checkpoints"/f"seed{seed}",device); reports.append(report); routers[seed]=load_router(args.output_dir/"checkpoints"/f"seed{seed}"/"model.pt",seed,device)
        if seed==1: cached_validation=(val_inputs,val_pm,val_pn,val_q)
        else:
            # The same validation input values are intentionally recomputed from seed-matched frozen experts.
            pass
    # Recompute the exact seed-matched validation distributions for threshold evaluation.
    tau_inputs={}
    for seed in (1,2,3):
        nvp=load_frozen_nvp(Path(cfg["frozen_experts"]["nvp_checkpoint_root"])/f"seed{seed}"/"model.pt",seed).to(device).eval(); mamba=load_frozen_mamba(Path(cfg["frozen_experts"]["mamba_checkpoint_root"])/f"seed{seed}"/"model.pt",seed,controlled,tokens,lengths).to(device).eval(); vm,vn=expert_logits(mamba,nvp,val_x,int(cfg["training"]["evaluation_batch_size"])); inputs,pm,pn=distribution_features(val_x,vm,vn); tau_inputs[seed]=(inputs,pm,pn)
    # evaluate_taus accepts one tensor triplet; run exact seed-matched scores here.
    stage_b_report=load_json(Path("outputs/route_a_stage_b_v6/comparison_report.json")); tau_results={}; nvp_row=next(x for x in stage_b_report["models"] if x["architecture"]=="NVP"); mamba_row=next(x for x in stage_b_report["models"] if x["architecture"]=="Mamba")
    for tau in cfg["threshold_selection"]["tau_values"]:
        seed_metrics=[]
        for seed in (1,2,3):
            inputs,pm,pn=tau_inputs[seed]
            with torch.no_grad(): score,_=thresholded_mixture(pm,pn,routers[seed](inputs),float(tau))
            seed_metrics.append(metrics_for_scores(score.cpu(),val_q.cpu(),validation,validation_matrix))
        per_dataset={d:sum(x["per_dataset"][d] for x in seed_metrics)/3 for d in sorted(seed_metrics[0]["per_dataset"])}; mean=sum(x["ValidationFinalMeanOverOz"] for x in seed_metrics)/3; delta={d:per_dataset[d]-nvp_row["per_dataset_3seed"][d] for d in per_dataset}; median=float(np.median(list(delta.values())))
        tau_results[str(tau)]={"three_seed_validation_mean_over_oz":mean,"per_seed":{str(s):m["ValidationFinalMeanOverOz"] for s,m in zip((1,2,3),seed_metrics)},"per_dataset":per_dataset,"delta_vs_nvp_per_dataset":delta,"macro_delta_vs_nvp":mean-nvp_row["ValidationFinalMeanOverOz_3seed"],"median_dataset_delta_vs_nvp":median,"positive_dataset_count":sum(v>0 for v in delta.values()),"negative_dataset_count":sum(v<0 for v in delta.values()),"S_val":mean-nvp_row["ValidationFinalMeanOverOz_3seed"]+0.5*median,"seed_metrics":seed_metrics}
    eligible=[(float(t),r) for t,r in tau_results.items() if r["three_seed_validation_mean_over_oz"]>float(cfg["threshold_selection"]["high_bar_validation_mean_over_oz"])]
    selected_tau=None if not eligible else max(eligible,key=lambda x:(x[1]["S_val"],x[1]["positive_dataset_count"],x[1]["three_seed_validation_mean_over_oz"]))[0]
    if not math.isclose(tau_results["1.0"]["three_seed_validation_mean_over_oz"], mamba_row["ValidationFinalMeanOverOz_3seed"], rel_tol=0.0, abs_tol=1e-10): raise RuntimeError("tau=1 did not reproduce frozen standalone Mamba validation policy45")
    validation_report={"step_execution":"COMPLETE","final_test_accessed":False,"compiler_gym_initialized":False,"llvm_execution":False,"candidate_rollouts":0,"objecttext_measurements":0,"label_regeneration":False,"frozen_expert_validation_checks":validation_expert_checks,"seed_reports":reports,"tau_results":tau_results,"selected_tau":selected_tau,"high_bar":cfg["threshold_selection"]["high_bar_validation_mean_over_oz"],"decision":"PASS_VALIDATION_HIGH_BAR" if selected_tau is not None else "FAIL_VALIDATION_HIGH_BAR"}
    (args.output_dir/"validation_report.json").write_text(json.dumps(validation_report,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    if selected_tau is None:
        (args.output_dir/"comparison_report.json").write_text(json.dumps({"step_execution":"COMPLETE","decision":"FAIL_VALIDATION_HIGH_BAR","validation":validation_report,"final_ood_executed":False},indent=2,sort_keys=True)+"\n",encoding="utf-8"); return 0
    # Only after the validation-selected checkpoints and tau are frozen, load the final/OOD artifacts once.
    programs,final_matrix,summaries=read_final_artifacts(Path(cfg["label_shards"]["final"])); eligible_final=[p for p in programs if p in final_matrix and summaries[p]["ratio_metric_validity"]=="valid_for_ObjectText_ratio_metric"]
    if len(eligible_final)!=4679: raise ValueError("frozen final complete-K50 population mismatch")
    final_features=load_final_features(Path(cfg["autophase_feature_cache"]["final"]),eligible_final); final_x=torch.tensor([final_features[p] for p in eligible_final],dtype=torch.float32,device=device)
    final_reports=[]; result_maps={}
    for seed in (1,2,3):
        nvp=load_frozen_nvp(Path(cfg["frozen_experts"]["nvp_checkpoint_root"])/f"seed{seed}"/"model.pt",seed).to(device).eval(); mamba=load_frozen_mamba(Path(cfg["frozen_experts"]["mamba_checkpoint_root"])/f"seed{seed}"/"model.pt",seed,controlled,tokens,lengths).to(device).eval(); fm,fn=expert_logits(mamba,nvp,final_x,int(cfg["training"]["evaluation_batch_size"])); inputs,pm,pn=distribution_features(final_x,fm,fn); path=args.output_dir/"final_results"/f"seed{seed}.jsonl.gz"; final_reports.append(final_rows(seed,routers[seed],pm,pn,inputs,programs,eligible_final,final_matrix,summaries,selected_tau,path)); result_maps[("AMR",seed)]=read_results(path); result_maps[("NVP",seed)]=read_results(Path(cfg["existing_final_references"]["nvp_result_root"])/f"seed{seed}.jsonl.gz"); result_maps[("Mamba",seed)]=read_results(Path(cfg["existing_final_references"]["mamba_result_root"])/f"seed{seed}.jsonl.gz")
    combined=aggregate(["NVP","Mamba","AMR"],programs,summaries,result_maps); amr=combined["dataset_macro"]["AMR"]["three_seed_mean"]; nvp=combined["dataset_macro"]["NVP"]["three_seed_mean"]; deltas={d:combined["per_dataset"][d]["AMR"]["three_seed_mean"]-combined["per_dataset"][d]["NVP"]["three_seed_mean"] for d in combined["per_dataset"]}; leave=[v for d,v in deltas.items() if d!="llvm-stress-v0"]
    comparison={"step_execution":"COMPLETE","decision":"COMPLETE","offline_only":True,"compiler_gym_initialized":False,"llvm_execution":False,"candidate_rollouts":0,"objecttext_measurements":0,"label_regeneration":False,"runtime":False,"selected_tau":selected_tau,"validation":validation_report,"final_population":{"N_total":4683,"N_complete_K50_valid":4679,"N_invalid":4},"final_seed_results":final_reports,"combined_comparison":combined,"amr":{"three_seed_mean_over_oz":amr,"delta_vs_NVP":amr-nvp,"delta_vs_Mamba":amr-combined["dataset_macro"]["Mamba"]["three_seed_mean"],"delta_vs_DirectMambaNVP":amr-cfg["existing_final_references"]["MambaNVP"],"delta_vs_CrossCandidate":amr-cfg["existing_final_references"]["CrossCandidateMambaNVP"],"delta_vs_Anchored":amr-cfg["existing_final_references"]["AnchoredMambaNVP"],"per_dataset_delta_vs_NVP":deltas,"median_dataset_delta_vs_NVP":float(np.median(list(deltas.values()))),"positive_dataset_count":sum(v>0 for v in deltas.values()),"negative_dataset_count":sum(v<0 for v in deltas.values()),"leave_llvm_stress_out_delta_vs_NVP":sum(leave)/len(leave),"three_seed_paired_delta_vs_NVP":{str(s):combined["dataset_macro"]["AMR"]["per_seed"][str(s)]-combined["dataset_macro"]["NVP"]["per_seed"][str(s)] for s in (1,2,3)},"oracle_recovery":amr/load_json(Path("outputs/route_a_final_objecttext_v6/comparison_report.json"))["offline_k50_oracle"]["dataset_macro"],"policy45_regret":regret_summary("AMR",result_maps),"top1_oracle_tie_accuracy":final_top1_oracle_tie_accuracy(result_maps,final_matrix)}}
    (args.output_dir/"comparison_report.json").write_text(json.dumps(comparison,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps(comparison,indent=2,sort_keys=True),flush=True); return 0


if __name__ == "__main__": raise SystemExit(main())
