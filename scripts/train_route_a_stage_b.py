#!/usr/bin/env python3
"""Run the frozen three-seed Route-A Stage-B replication."""
from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from scripts.train_autophase_nvp_objecttext import AutophaseNVP
from scripts.train_controlled_nvp_stage_a import (
    ControlledCandidateModel,
    evaluate,
    extract_features,
    learning_rate,
    load_candidates,
    read_jsonl,
    read_label_matrix,
    soft_cross_entropy,
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_stage_b_config(stage: Mapping[str, Any], nvp: Mapping[str, Any], controlled: Mapping[str, Any]) -> None:
    if stage["final_seed_set"] != [1, 2, 3]:
        raise ValueError("Stage B final_seed_set must be exactly [1, 2, 3]")
    if stage["model_order"] != ["NVP", "MLP", "LSTM", "Transformer", "Mamba"]:
        raise ValueError("Stage B model order is frozen")
    if stage["stage_a_checkpoint_reuse"] is not False:
        raise ValueError("Stage B must train fresh checkpoints")
    if nvp["seed"] != 0 or controlled["training"]["seed"] != 0:
        raise ValueError("Stage-A source configs must remain seed 0")
    if nvp["total_steps"] != 10000 or controlled["training"]["total_steps"] != 10000:
        raise ValueError("frozen training budgets must remain 10000 steps")
    if nvp["validation_cadence_steps"] != 100 or controlled["training"]["checkpoint_evaluation_cadence_steps"] != 100:
        raise ValueError("frozen checkpoint evaluation cadence must remain 100")
    if nvp["nvp_target_temperature"] != 0.05 or controlled["target_and_objective"]["target_temperature"] != 0.05:
        raise ValueError("frozen target temperature must remain 0.05")
    candidate = controlled["candidate_representation"]
    if candidate["K"] != 50 or candidate["padded_length"] != 20 or candidate["pad_token_id"] != 124:
        raise ValueError("frozen controlled candidate representation mismatch")


def seeded_config(base: Mapping[str, Any], *, seed: int, kind: str, architecture: str | None = None) -> dict[str, Any]:
    result = copy.deepcopy(dict(base))
    if kind == "NVP":
        result["seed"] = seed
    else:
        result["training"]["seed"] = seed
        if architecture is not None:
            result["models"] = {architecture: result["models"][architecture]}
    return result


def run_plan(stage: Mapping[str, Any]) -> list[tuple[str, int]]:
    return [(model, seed) for model in stage["model_order"] for seed in stage["final_seed_set"]]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def checkpoint_payload(model: torch.nn.Module, *, name: str, seed: int, model_config: Mapping[str, Any], step: int, metrics: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "stage": "Route-A Stage B",
        "architecture": name,
        "seed": seed,
        "stage_a_checkpoint_reused": False,
        "model_config": dict(model_config),
        "state_dict": model.state_dict(),
        "step": step,
        "metrics": dict(metrics),
    }


def train_one(
    *,
    name: str,
    seed: int,
    frozen_config: Mapping[str, Any],
    controlled_common: Mapping[str, Any],
    tokens: torch.Tensor,
    lengths: torch.Tensor,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    val_x: torch.Tensor,
    val_y: torch.Tensor,
    val_records: list[dict[str, Any]],
    matrix: dict[str, list[dict[str, Any]]],
    output_dir: Path,
) -> dict[str, Any]:
    seed_everything(seed)
    if name == "NVP":
        model = AutophaseNVP().cuda()
        training = {
            "learning_rate": frozen_config["lr"],
            "weight_decay": frozen_config["weight_decay"],
            "batch_size": frozen_config["batch_size"],
            "evaluation_batch_size": frozen_config["batch_size"],
            "total_steps": frozen_config["total_steps"],
            "warmup_steps": 500,
            "checkpoint_evaluation_cadence_steps": frozen_config["validation_cadence_steps"],
        }
        model_config: Mapping[str, Any] = {"architecture": frozen_config["architecture"]}
    else:
        model_config = frozen_config["models"][name]
        model = ControlledCandidateModel(name, {**controlled_common["candidate_representation"], **model_config}, tokens, lengths).cuda()
        training = controlled_common["training"]
    optimizer = torch.optim.Adam(model.parameters(), lr=float(training["learning_rate"]), weight_decay=float(training["weight_decay"]))
    generator = torch.Generator().manual_seed(seed)
    output_dir.mkdir(parents=True)
    (output_dir / "config.json").write_text(json.dumps(frozen_config, indent=2, sort_keys=True) + "\n")
    curve: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    step, batch, total = 0, int(training["batch_size"]), int(training["total_steps"])
    while step < total:
        order = torch.randperm(len(train_x), generator=generator)
        for begin in range(0, len(order), batch):
            step += 1
            index = order[begin:begin + batch].cuda(non_blocking=True)
            lr = learning_rate(training, step)
            for group in optimizer.param_groups:
                group["lr"] = lr
            model.train()
            loss = soft_cross_entropy(model(train_x[index]), train_y[index])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if step % int(training["checkpoint_evaluation_cadence_steps"]) == 0 or step == total:
                metric = evaluate(model, val_x, val_y, val_records, matrix, int(training["evaluation_batch_size"]))
                metric.update({"step": step, "train_loss": float(loss.detach().cpu()), "lr": lr})
                curve.append(metric)
                print(json.dumps({"architecture": name, "seed": seed, **metric}, sort_keys=True), flush=True)
                if best is None or metric["ValidationFinalMeanOverOz"] > best["ValidationFinalMeanOverOz"]:
                    best = metric
                    torch.save(checkpoint_payload(model, name=name, seed=seed, model_config=model_config, step=step, metrics=metric), output_dir / "model.pt")
            if step == total:
                break
    if best is None:
        raise RuntimeError("no checkpoint evaluation occurred")
    (output_dir / "learning_curve.json").write_text(json.dumps(curve, indent=2, sort_keys=True) + "\n")
    report = {
        "stage": "Route-A Stage B",
        "architecture": name,
        "seed": seed,
        "step_execution": "COMPLETE",
        "stage_a_checkpoint_reused": False,
        "trainable_parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "selection_metric": "ValidationFinalMeanOverOz policy-45 dataset macro mean",
        "selected": best,
    }
    (output_dir / "experiment_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def aggregate_model(seed_reports: list[dict[str, Any]], oracle: float) -> dict[str, Any]:
    ordered = sorted(seed_reports, key=lambda item: item["seed"])
    if [item["seed"] for item in ordered] != [1, 2, 3]:
        raise ValueError("aggregate requires exactly seeds 1, 2, 3")
    selected = [item["selected"] for item in ordered]
    datasets = sorted(selected[0]["per_dataset"])
    if any(sorted(metric["per_dataset"]) != datasets for metric in selected):
        raise ValueError("per-dataset validation membership mismatch")
    mean = sum(metric["ValidationFinalMeanOverOz"] for metric in selected) / 3
    return {
        "architecture": ordered[0]["architecture"],
        "seed_results": ordered,
        "ValidationFinalMeanOverOz_3seed": mean,
        "per_dataset_3seed": {dataset: sum(metric["per_dataset"][dataset] for metric in selected) / 3 for dataset in datasets},
        "policy45_regret_mean_bytes_3seed": sum(metric["policy45_regret_mean_bytes"] for metric in selected) / 3,
        "policy45_regret_median_bytes_per_seed": [metric["policy45_regret_median_bytes"] for metric in selected],
        "positive_program_count_vs_Oz_3seed_mean": sum(metric["positive_program_count_vs_Oz"] for metric in selected) / 3,
        "oracle_opportunity_recovered": mean / oracle,
        "validation_cohort_per_seed": [{key: metric[key] for key in ("N_total", "N_primary_valid", "N_failed_or_invalid")} for metric in selected],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    stage = load_json(args.config)
    nvp = load_json(Path(stage["nvp_config_source"]))
    controlled = load_json(Path(stage["controlled_config_source"]))
    validate_stage_b_config(stage, nvp, controlled)
    if not torch.cuda.is_available():
        raise RuntimeError("Stage B requires CUDA")
    train = read_jsonl(Path(stage["target_files"]["train"]))
    validation = read_jsonl(Path(stage["target_files"]["validation"]))
    if len(train) != stage["frozen_data_population"]["train_complete_k50"] or len(validation) != stage["frozen_data_population"]["validation_complete_k50"]:
        raise ValueError("frozen target population mismatch")
    candidate = controlled["candidate_representation"]
    tokens, lengths = load_candidates(Path(candidate["candidate_sequences"]), pad_token_id=int(candidate["pad_token_id"]), padded_length=int(candidate["padded_length"]))
    matrix = read_label_matrix(Path(stage["validation_label_shards"]))
    if set(matrix) != {row["program_id"] for row in validation}:
        raise ValueError("validation target/label cohort mismatch")
    train_features, validation_features = extract_features(train, args.workers), extract_features(validation, args.workers)
    train_x = torch.tensor([train_features[row["program_id"]] for row in train], device="cuda")
    train_y = torch.tensor([row["normalized_target"] for row in train], device="cuda")
    val_x = torch.tensor([validation_features[row["program_id"]] for row in validation], device="cuda")
    val_y = torch.tensor([row["normalized_target"] for row in validation], device="cuda")
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "stage_b_config.json").write_text(json.dumps(stage, indent=2, sort_keys=True) + "\n")
    reports: list[dict[str, Any]] = []
    for name, seed in run_plan(stage):
        base = nvp if name == "NVP" else controlled
        frozen = seeded_config(base, seed=seed, kind=name, architecture=None if name == "NVP" else name)
        reports.append(train_one(name=name, seed=seed, frozen_config=frozen, controlled_common=controlled, tokens=tokens.cuda(), lengths=lengths.cuda(), train_x=train_x, train_y=train_y, val_x=val_x, val_y=val_y, val_records=validation, matrix=matrix, output_dir=args.output_dir / name.lower() / f"seed{seed}"))
    models = [aggregate_model([report for report in reports if report["architecture"] == name], float(stage["fixed_route_a_oracle"])) for name in stage["model_order"]]
    controlled_means = {item["architecture"]: item["ValidationFinalMeanOverOz_3seed"] for item in models if item["architecture"] != "NVP"}
    report = {
        "stage": "Route-A Stage B",
        "step_execution": "COMPLETE",
        "final_seed_set": stage["final_seed_set"],
        "stage_a_configurations_unchanged": True,
        "stage_a_checkpoint_reused": False,
        "selection_metric": stage["frozen_validation"]["selection_metric"],
        "sampling": False,
        "offline_label_matrix_only": True,
        "fixed_route_a_oracle": stage["fixed_route_a_oracle"],
        "models": models,
        "mamba_remains_highest_controlled": max(controlled_means, key=controlled_means.get) == "Mamba",
        "mamba_vs_nvp_3seed_difference": next(item["ValidationFinalMeanOverOz_3seed"] for item in models if item["architecture"] == "Mamba") - next(item["ValidationFinalMeanOverOz_3seed"] for item in models if item["architecture"] == "NVP"),
    }
    (args.output_dir / "comparison_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
