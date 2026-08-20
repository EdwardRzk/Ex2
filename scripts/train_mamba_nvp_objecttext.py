#!/usr/bin/env python3
"""Train MambaNVP only from frozen targets, features, and offline labels."""
from __future__ import annotations

import argparse
import copy
import gzip
import json
import random
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn

if __package__:
    from scripts.train_autophase_nvp_objecttext import AutophaseNVP
    from scripts.train_controlled_nvp_stage_a import (
        AUTOPHASE_DIM, ControlledCandidateModel, evaluate, learning_rate, load_candidates, read_jsonl, read_label_matrix, soft_cross_entropy,
    )
else:
    from train_autophase_nvp_objecttext import AutophaseNVP
    from train_controlled_nvp_stage_a import (
        AUTOPHASE_DIM, ControlledCandidateModel, evaluate, learning_rate, load_candidates, read_jsonl, read_label_matrix, soft_cross_entropy,
    )


K = 50


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_config(cfg: Mapping[str, Any], controlled: Mapping[str, Any]) -> None:
    if cfg["final_seed_set"] != [1, 2, 3]:
        raise ValueError("MambaNVP seeds must be exactly [1, 2, 3]")
    if cfg["fusion"] != "final_logits = frozen_nvp_logits + mamba_residual_logits":
        raise ValueError("MambaNVP fusion is frozen")
    if cfg["target_and_objective"]["target_temperature"] != 0.05:
        raise ValueError("target temperature must remain 0.05")
    if cfg["training"]["total_steps"] != 10000 or cfg["training"]["checkpoint_evaluation_cadence_steps"] != 100:
        raise ValueError("training budget or validation cadence changed")
    candidate = controlled["candidate_representation"]
    if candidate["K"] != K or candidate["padded_length"] != 20 or candidate["pad_token_id"] != 124:
        raise ValueError("frozen candidate representation mismatch")
    if cfg["validation"]["sampling"] is not False or cfg["validation"]["scored_pass_budget"] != 45:
        raise ValueError("frozen offline policy semantics changed")


def load_feature_cache(path: Path, split: str, expected_program_ids: Sequence[str]) -> dict[str, list[float]]:
    features: dict[str, list[float]] = {}
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["split"] != split or len(row["raw_autophase"]) != AUTOPHASE_DIM or len(row["normalized_autophase"]) != AUTOPHASE_DIM:
                raise ValueError(f"invalid cached Autophase row in {path}")
            program_id = str(row["program_id"])
            if program_id in features:
                raise ValueError(f"duplicate cached feature: {program_id}")
            features[program_id] = list(row["normalized_autophase"])
    if set(features) != set(expected_program_ids):
        raise ValueError(f"cached feature population mismatch: {path}")
    return features


def load_frozen_nvp(path: Path, seed: int) -> AutophaseNVP:
    payload = torch.load(path, map_location="cpu")
    if payload.get("stage") != "Route-A Stage B" or payload.get("architecture") != "NVP" or payload.get("seed") != seed or payload.get("stage_a_checkpoint_reused") is not False:
        raise ValueError(f"not the frozen Stage-B NVP checkpoint for seed {seed}: {path}")
    model = AutophaseNVP()
    model.load_state_dict(payload["state_dict"], strict=True)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model.eval()


class MambaNVP(nn.Module):
    def __init__(self, nvp: AutophaseNVP, residual_cfg: Mapping[str, Any], tokens: torch.Tensor, lengths: torch.Tensor) -> None:
        super().__init__()
        self.nvp = nvp
        self.residual = ControlledCandidateModel("Mamba", residual_cfg, tokens, lengths)
        nn.init.zeros_(self.residual.value_head.weight)
        nn.init.zeros_(self.residual.value_head.bias)
        for parameter in self.nvp.parameters():
            parameter.requires_grad_(False)
        self.nvp.eval()

    def train(self, mode: bool = True) -> "MambaNVP":
        super().train(mode)
        self.nvp.eval()
        return self

    def forward(self, program: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            nvp_logits = self.nvp(program)
        return nvp_logits + self.residual(program)

    def residual_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.residual.parameters() if parameter.requires_grad)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def checkpoint_payload(model: MambaNVP, cfg: Mapping[str, Any], seed: int, step: int, metrics: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "stage": "Route-A MambaNVP v6",
        "architecture": "MambaNVP",
        "seed": seed,
        "step": step,
        "metrics": dict(metrics),
        "residual_state_dict": model.residual.state_dict(),
        "residual_trainable_parameters": model.residual_parameter_count(),
        "nvp_checkpoint": str(Path(cfg["nvp_checkpoint_root"]) / f"seed{seed}" / "model.pt"),
        "nvp_frozen": True,
        "fusion": cfg["fusion"],
        "config": copy.deepcopy(dict(cfg)),
    }


def train_seed(
    *,
    cfg: Mapping[str, Any],
    controlled: Mapping[str, Any],
    seed: int,
    tokens: torch.Tensor,
    lengths: torch.Tensor,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    val_x: torch.Tensor,
    val_y: torch.Tensor,
    validation: list[dict[str, Any]],
    matrix: Mapping[str, list[dict[str, Any]]],
    output: Path,
) -> dict[str, Any]:
    seed_everything(seed)
    nvp = load_frozen_nvp(Path(cfg["nvp_checkpoint_root"]) / f"seed{seed}" / "model.pt", seed)
    residual_cfg = {**controlled["candidate_representation"], **controlled["models"]["Mamba"]}
    model = MambaNVP(nvp, residual_cfg, tokens, lengths).cuda()
    if any(parameter.requires_grad for parameter in model.nvp.parameters()) or model.nvp.training:
        raise RuntimeError("frozen NVP branch invariant failed")
    optimizer = torch.optim.Adam(model.residual.parameters(), lr=float(cfg["training"]["learning_rate"]), weight_decay=float(cfg["training"]["weight_decay"]))
    generator = torch.Generator().manual_seed(seed)
    output.mkdir(parents=True)
    curve: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    step, batch_size, total_steps = 0, int(cfg["training"]["batch_size"]), int(cfg["training"]["total_steps"])
    while step < total_steps:
        order = torch.randperm(len(train_x), generator=generator)
        for begin in range(0, len(order), batch_size):
            step += 1
            index = order[begin : begin + batch_size].cuda(non_blocking=True)
            lr = learning_rate(cfg["training"], step)
            for group in optimizer.param_groups:
                group["lr"] = lr
            model.train()
            loss = soft_cross_entropy(model(train_x[index]), train_y[index])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if any(parameter.grad is not None for parameter in model.nvp.parameters()):
                raise RuntimeError("frozen NVP branch received gradients")
            optimizer.step()
            if step % int(cfg["training"]["checkpoint_evaluation_cadence_steps"]) == 0 or step == total_steps:
                metric = evaluate(model, val_x, val_y, validation, dict(matrix), int(cfg["training"]["evaluation_batch_size"]))
                metric.update({"step": step, "train_loss": float(loss.detach().cpu()), "lr": lr})
                curve.append(metric)
                print(json.dumps({"architecture": "MambaNVP", "seed": seed, **metric}, sort_keys=True), flush=True)
                if best is None or metric["ValidationFinalMeanOverOz"] > best["ValidationFinalMeanOverOz"]:
                    best = metric
                    torch.save(checkpoint_payload(model, cfg, seed, step, metric), output / "model.pt")
            if step == total_steps:
                break
    if best is None:
        raise RuntimeError("no validation checkpoint evaluated")
    (output / "learning_curve.json").write_text(json.dumps(curve, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report = {
        "stage": "Route-A MambaNVP v6",
        "architecture": "MambaNVP",
        "seed": seed,
        "step_execution": "COMPLETE",
        "nvp_frozen": True,
        "residual_trainable_parameters": model.residual_parameter_count(),
        "selection_metric": "ValidationFinalMeanOverOz policy-45 dataset macro mean",
        "selected": best,
    }
    (output / "experiment_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def aggregate(reports: list[dict[str, Any]]) -> dict[str, Any]:
    ordered = sorted(reports, key=lambda report: report["seed"])
    if [report["seed"] for report in ordered] != [1, 2, 3]:
        raise ValueError("MambaNVP aggregate requires seeds 1, 2, 3")
    selected = [report["selected"] for report in ordered]
    datasets = sorted(selected[0]["per_dataset"])
    if any(sorted(metric["per_dataset"]) != datasets for metric in selected):
        raise ValueError("validation dataset membership mismatch")
    return {
        "architecture": "MambaNVP",
        "seed_results": ordered,
        "ValidationFinalMeanOverOz_3seed": sum(metric["ValidationFinalMeanOverOz"] for metric in selected) / 3,
        "per_dataset_3seed": {dataset: sum(metric["per_dataset"][dataset] for metric in selected) / 3 for dataset in datasets},
        "policy45_regret_mean_bytes_3seed": sum(metric["policy45_regret_mean_bytes"] for metric in selected) / 3,
        "policy45_regret_median_bytes_per_seed": [metric["policy45_regret_median_bytes"] for metric in selected],
        "validation_cohort_per_seed": [{key: metric[key] for key in ("N_total", "N_primary_valid", "N_failed_or_invalid")} for metric in selected],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output directory: {args.output_dir}")
    if not torch.cuda.is_available():
        raise RuntimeError("MambaNVP formal training requires CUDA")
    cfg = load_json(args.config)
    controlled = load_json(Path(cfg["candidate_representation_source"]))
    validate_config(cfg, controlled)
    train, validation = read_jsonl(Path(cfg["target_files"]["train"])), read_jsonl(Path(cfg["target_files"]["validation"]))
    if len(train) != cfg["frozen_data_population"]["train_complete_k50"] or len(validation) != cfg["frozen_data_population"]["validation_complete_k50"]:
        raise ValueError("frozen target population mismatch")
    train_features = load_feature_cache(Path(cfg["autophase_feature_cache"]["train"]), "train", [row["program_id"] for row in train])
    validation_features = load_feature_cache(Path(cfg["autophase_feature_cache"]["validation"]), "validation", [row["program_id"] for row in validation])
    tokens, lengths = load_candidates(Path(controlled["candidate_representation"]["candidate_sequences"]), pad_token_id=int(controlled["candidate_representation"]["pad_token_id"]), padded_length=int(controlled["candidate_representation"]["padded_length"]))
    matrix = read_label_matrix(Path(cfg["validation_label_shards"]))
    if set(matrix) != {row["program_id"] for row in validation}:
        raise ValueError("validation target/label cohort mismatch")
    train_x = torch.tensor([train_features[row["program_id"]] for row in train], dtype=torch.float32, device="cuda")
    train_y = torch.tensor([row["normalized_target"] for row in train], dtype=torch.float32, device="cuda")
    val_x = torch.tensor([validation_features[row["program_id"]] for row in validation], dtype=torch.float32, device="cuda")
    val_y = torch.tensor([row["normalized_target"] for row in validation], dtype=torch.float32, device="cuda")
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "config.json").write_text(json.dumps(cfg, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    reports = [
        train_seed(cfg=cfg, controlled=controlled, seed=seed, tokens=tokens.cuda(), lengths=lengths.cuda(), train_x=train_x, train_y=train_y, val_x=val_x, val_y=val_y, validation=validation, matrix=matrix, output=args.output_dir / f"seed{seed}")
        for seed in cfg["final_seed_set"]
    ]
    report = {
        "step_execution": "COMPLETE",
        "training_only": True,
        "final_test_accessed": False,
        "ood_accessed": False,
        "runtime_accessed": False,
        "compiler_gym_initialized": False,
        "llvm_execution": False,
        "comparison": aggregate(reports),
    }
    (args.output_dir / "comparison_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
