#!/usr/bin/env python3
"""Train GatedCalibratedMambaNVP on frozen Route-A validation artifacts only."""
from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from mamba_ssm import Mamba
from torch import nn

if __package__:
    from scripts.train_controlled_nvp_stage_a import AUTOPHASE_DIM, learning_rate, soft_cross_entropy
    from scripts.train_mamba_nvp_objecttext import load_feature_cache, load_frozen_nvp
    from scripts.train_set_conditioned_mamba_ranker import K, CandidateInterface, load_candidates, load_json, policy_metrics, read_jsonl, read_label_matrix
else:
    from train_controlled_nvp_stage_a import AUTOPHASE_DIM, learning_rate, soft_cross_entropy
    from train_mamba_nvp_objecttext import load_feature_cache, load_frozen_nvp
    from train_set_conditioned_mamba_ranker import K, CandidateInterface, load_candidates, load_json, policy_metrics, read_jsonl, read_label_matrix


METHOD = "GatedCalibratedMambaNVP"


def kl_final_to_nvp(final_logits: torch.Tensor, nvp_logits: torch.Tensor) -> torch.Tensor:
    final_log_prob = torch.log_softmax(final_logits, dim=1)
    final_prob = final_log_prob.exp()
    return (final_prob * (final_log_prob - torch.log_softmax(nvp_logits, dim=1))).sum(dim=1).mean()


def kl_nvp_to_final(final_logits: torch.Tensor, nvp_logits: torch.Tensor) -> torch.Tensor:
    nvp_log_prob = torch.log_softmax(nvp_logits, dim=1)
    nvp_prob = nvp_log_prob.exp()
    return (nvp_prob * (nvp_log_prob - torch.log_softmax(final_logits, dim=1))).sum(dim=1).mean()


class GatedCalibratedMambaNVP(CandidateInterface):
    def __init__(self, nvp: nn.Module, cfg: Mapping[str, Any], tokens: torch.Tensor, lengths: torch.Tensor) -> None:
        super().__init__(cfg, tokens, lengths)
        d = self.d_model
        self.nvp = nvp
        self.block_norms = nn.ModuleList([nn.LayerNorm(d) for _ in range(int(cfg["layers"]))])
        self.blocks = nn.ModuleList([Mamba(d_model=d, d_state=int(cfg["d_state"]), d_conv=int(cfg["d_conv"]), expand=int(cfg["expand"]), use_fast_path=bool(cfg["use_fast_path"]), layer_idx=index) for index in range(int(cfg["layers"]))])
        self.output_norm = nn.LayerNorm(d)
        self.residual_head = nn.Linear(d, 1)
        self.gate = nn.Sequential(nn.Linear(d + AUTOPHASE_DIM, d), nn.ReLU(), nn.Linear(d, 1))
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)
        for parameter in self.nvp.parameters():
            parameter.requires_grad_(False)
        self.nvp.eval()

    def train(self, mode: bool = True) -> "GatedCalibratedMambaNVP":
        super().train(mode)
        self.nvp.eval()
        return self

    def embeddings(self, program: torch.Tensor) -> torch.Tensor:
        hidden, lengths = self.candidate_inputs(program)
        for norm, block in zip(self.block_norms, self.blocks):
            hidden = hidden + block(norm(hidden))
        rows = torch.arange(len(hidden), device=hidden.device)
        return self.output_norm(hidden[rows, lengths - 1]).reshape(program.shape[0], K, -1)

    def components(self, program: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            nvp_logits = self.nvp(program)
        embeddings = self.embeddings(program)
        residual = self.residual_head(embeddings).squeeze(-1)
        gate_input = torch.cat([embeddings, program[:, None, :].expand(-1, K, -1)], dim=-1)
        alpha = torch.sigmoid(self.gate(gate_input).squeeze(-1))
        return nvp_logits, residual, alpha, nvp_logits + alpha * residual

    def forward(self, program: torch.Tensor) -> torch.Tensor:
        return self.components(program)[-1]

    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)


def seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def validate_config(cfg: Mapping[str, Any], controlled: Mapping[str, Any]) -> None:
    assert cfg["final_seed_set"] == [1, 2, 3]
    assert cfg["frozen_data_population"] == {"train_complete_k50": 28159, "validation_complete_k50": 4488}
    assert cfg["target_and_objective"]["target_temperature"] == 0.05 and cfg["target_and_objective"]["lambda_kl"] == 0.1
    assert cfg["training"]["total_steps"] == 10000 and cfg["training"]["checkpoint_evaluation_cadence_steps"] == 100 and not cfg["training"]["early_stopping"]
    assert cfg["validation"] == {"sampling": False, "ranking": "descending final logits; candidate ID ascending tie break", "scored_pass_budget": 45, "selection_metric": "ValidationFinalMeanOverOz policy-45 dataset macro mean", "final_test_accessed": False, "ood_accessed": False, "runtime_accessed": False}
    assert controlled["candidate_representation"]["K"] == K and controlled["candidate_representation"]["padded_length"] == 20 and controlled["candidate_representation"]["pad_token_id"] == 124


def diagnostics(model: GatedCalibratedMambaNVP, features: torch.Tensor, targets: torch.Tensor, records: Sequence[Mapping[str, Any]], matrix: Mapping[str, Sequence[Mapping[str, Any]]], batch_size: int) -> dict[str, Any]:
    logits_parts, ce_total, kl_final_total, kl_nvp_total, alpha_total, top1_correct = [], 0.0, 0.0, 0.0, 0.0, 0
    model.eval()
    with torch.no_grad():
        for start in range(0, len(features), batch_size):
            end = min(start + batch_size, len(features)); current = features[start:end]
            nvp_logits, _, alpha, final_logits = model.components(current)
            logits_parts.append(final_logits.cpu()); count = len(current)
            ce_total += float(soft_cross_entropy(final_logits, targets[start:end]).cpu()) * count
            kl_final_total += float(kl_final_to_nvp(final_logits, nvp_logits).cpu()) * count
            kl_nvp_total += float(kl_nvp_to_final(final_logits, nvp_logits).cpu()) * count
            alpha_total += float(alpha.sum().cpu())
            for score, record in zip(final_logits.cpu().tolist(), records[start:end]):
                choice = min(range(K), key=lambda index: (-score[index], index))
                sizes = [item["best_object_text_size_bytes"] for item in matrix[record["program_id"]]]
                top1_correct += int(sizes[choice] == min(sizes))
    metrics = policy_metrics(torch.cat(logits_parts), list(records), dict(matrix))
    metrics.update({"validation_ce": ce_total / len(features), "calibration_kl_final_to_nvp": kl_final_total / len(features), "validation_kl_nvp_to_final": kl_nvp_total / len(features), "average_gate_alpha": alpha_total / (len(features) * K), "top1_accuracy": top1_correct / len(features)})
    return metrics


def frozen_references(cfg: Mapping[str, Any]) -> dict[str, float]:
    stage = load_json(Path(cfg["frozen_reference_reports"]["stage_b"]))
    models = {item["architecture"]: item for item in stage["models"]}
    mnvp = load_json(Path(cfg["frozen_reference_reports"]["mamba_nvp_v1"]))["comparison"]
    cross = load_json(Path(cfg["frozen_reference_reports"]["cross_candidate_mambanvp"]))["cross_candidate_mambanvp"]
    return {"NVP": models["NVP"]["ValidationFinalMeanOverOz_3seed"], "Mamba": models["Mamba"]["ValidationFinalMeanOverOz_3seed"], "MambaNVP_v1": mnvp["ValidationFinalMeanOverOz_3seed"], "CrossCandidateMambaNVP": cross["ValidationFinalMeanOverOz_3seed"]}


def train_seed(cfg: Mapping[str, Any], controlled: Mapping[str, Any], seed: int, tokens: torch.Tensor, lengths: torch.Tensor, train_x: torch.Tensor, train_y: torch.Tensor, val_x: torch.Tensor, val_y: torch.Tensor, validation: Sequence[Mapping[str, Any]], matrix: Mapping[str, Sequence[Mapping[str, Any]]], output: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    seed_everything(seed)
    model_cfg = {**controlled["candidate_representation"], **controlled["models"]["Mamba"]}
    model = GatedCalibratedMambaNVP(load_frozen_nvp(Path(cfg["nvp_checkpoint_root"]) / f"seed{seed}" / "model.pt", seed), model_cfg, tokens, lengths).cuda()
    assert not any(parameter.requires_grad for parameter in model.nvp.parameters()) and not model.nvp.training
    optimizer = torch.optim.Adam((parameter for parameter in model.parameters() if parameter.requires_grad), lr=float(cfg["training"]["learning_rate"]), weight_decay=float(cfg["training"]["weight_decay"]))
    generator = torch.Generator().manual_seed(seed); output.mkdir(parents=True); curve: list[dict[str, Any]] = []; best: dict[str, Any] | None = None; step = 0
    while step < int(cfg["training"]["total_steps"]):
        order = torch.randperm(len(train_x), generator=generator)
        for begin in range(0, len(train_x), int(cfg["training"]["batch_size"])):
            step += 1; index = order[begin:begin + int(cfg["training"]["batch_size"])].cuda(non_blocking=True); lr = learning_rate(cfg["training"], step)
            for group in optimizer.param_groups: group["lr"] = lr
            model.train(); nvp_logits, _, _, final_logits = model.components(train_x[index]); value_loss = soft_cross_entropy(final_logits, train_y[index]); calibration_loss = kl_final_to_nvp(final_logits, nvp_logits); loss = value_loss + float(cfg["target_and_objective"]["lambda_kl"]) * calibration_loss
            optimizer.zero_grad(set_to_none=True); loss.backward()
            if any(parameter.grad is not None for parameter in model.nvp.parameters()): raise RuntimeError("frozen NVP branch received gradients")
            optimizer.step()
            if step % int(cfg["training"]["checkpoint_evaluation_cadence_steps"]) == 0 or step == int(cfg["training"]["total_steps"]):
                metric = diagnostics(model, val_x, val_y, validation, matrix, int(cfg["training"]["evaluation_batch_size"])); metric.update({"step": step, "train_total_loss": float(loss.detach().cpu()), "train_value_loss": float(value_loss.detach().cpu()), "train_calibration_kl_final_to_nvp": float(calibration_loss.detach().cpu()), "lr": lr}); curve.append(metric); print(json.dumps({"architecture": METHOD, "seed": seed, **metric}, sort_keys=True), flush=True)
                if best is None or metric["ValidationFinalMeanOverOz"] > best["ValidationFinalMeanOverOz"]:
                    best = metric; torch.save({"stage": "Route-A Gated-Calibrated MambaNVP v2", "architecture": METHOD, "seed": seed, "step": step, "metrics": metric, "state_dict": model.state_dict(), "model_config": model_cfg, "nvp_checkpoint": str(Path(cfg["nvp_checkpoint_root"]) / f"seed{seed}" / "model.pt"), "nvp_frozen": True, "fusion": cfg["architecture"]["fusion"], "lambda_kl": cfg["target_and_objective"]["lambda_kl"]}, output / "model.pt")
            if step == int(cfg["training"]["total_steps"]): break
    if best is None: raise RuntimeError("no validation checkpoint evaluated")
    return {"architecture": METHOD, "seed": seed, "step_execution": "COMPLETE", "trainable_parameters": model.trainable_parameter_count(), "nvp_frozen": True, "selection_metric": "ValidationFinalMeanOverOz policy-45 dataset macro mean", "selected": best}, curve


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", type=Path, required=True); parser.add_argument("--output-dir", type=Path, required=True); args = parser.parse_args()
    if args.output_dir.exists(): raise FileExistsError(args.output_dir)
    if not torch.cuda.is_available(): raise RuntimeError("formal training requires CUDA")
    cfg = load_json(args.config); controlled = load_json(Path(cfg["candidate_representation_source"])); validate_config(cfg, controlled)
    train, validation = read_jsonl(Path(cfg["target_files"]["train"])), read_jsonl(Path(cfg["target_files"]["validation"])); assert len(train) == 28159 and len(validation) == 4488
    tokens, lengths = load_candidates(Path(controlled["candidate_representation"]["candidate_sequences"]), pad_token_id=124, padded_length=20); matrix = read_label_matrix(Path(cfg["validation_label_shards"])); assert set(matrix) == {row["program_id"] for row in validation}
    train_features = load_feature_cache(Path(cfg["autophase_feature_cache"]["train"]), "train", [row["program_id"] for row in train]); val_features = load_feature_cache(Path(cfg["autophase_feature_cache"]["validation"]), "validation", [row["program_id"] for row in validation])
    train_x = torch.tensor([train_features[row["program_id"]] for row in train], dtype=torch.float32, device="cuda"); train_y = torch.tensor([row["normalized_target"] for row in train], dtype=torch.float32, device="cuda"); val_x = torch.tensor([val_features[row["program_id"]] for row in validation], dtype=torch.float32, device="cuda"); val_y = torch.tensor([row["normalized_target"] for row in validation], dtype=torch.float32, device="cuda")
    args.output_dir.mkdir(parents=True); (args.output_dir / "config.json").write_text(json.dumps(cfg, indent=2, sort_keys=True) + "\n")
    reports, curves = [], {}
    for seed in cfg["final_seed_set"]:
        report, curve = train_seed(cfg, controlled, seed, tokens.cuda(), lengths.cuda(), train_x, train_y, val_x, val_y, validation, matrix, args.output_dir / "checkpoints" / f"seed{seed}"); reports.append(report); curves[str(seed)] = curve
    (args.output_dir / "training_curve.json").write_text(json.dumps(curves, indent=2, sort_keys=True) + "\n")
    mean = sum(report["selected"]["ValidationFinalMeanOverOz"] for report in reports) / 3; oracle = load_json(Path(cfg["frozen_reference_reports"]["stage_b"]))["fixed_route_a_oracle"]; references = frozen_references(cfg)
    summary = {"architecture": METHOD, "ValidationFinalMeanOverOz_3seed": mean, "oracle_recovery_3seed": mean / oracle, "policy45_regret_mean_bytes_3seed": sum(report["selected"]["policy45_regret_mean_bytes"] for report in reports) / 3, "top1_accuracy_3seed": sum(report["selected"]["top1_accuracy"] for report in reports) / 3, "validation_ce_3seed": sum(report["selected"]["validation_ce"] for report in reports) / 3, "calibration_kl_final_to_nvp_3seed": sum(report["selected"]["calibration_kl_final_to_nvp"] for report in reports) / 3, "validation_kl_nvp_to_final_3seed": sum(report["selected"]["validation_kl_nvp_to_final"] for report in reports) / 3, "average_gate_alpha_3seed": sum(report["selected"]["average_gate_alpha"] for report in reports) / 3, "trainable_parameters": reports[0]["trainable_parameters"], "seed_results": reports}
    report = {"step_execution": "COMPLETE", "training_only": True, "final_test_accessed": False, "ood_accessed": False, "runtime_accessed": False, "compiler_gym_initialized": False, "llvm_execution": False, "candidate_rollouts": 0, "objecttext_measurements": 0, "label_regeneration": False, "validation_cohort": {"N_total": 4488, "N_primary_valid": 4488, "N_failed_or_invalid": 0}, "gated_calibrated_mambanvp": summary, "frozen_references": references, "differences": {name: mean - value for name, value in references.items()}}
    (args.output_dir / "comparison_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n"); (args.output_dir / "experiment_report.json").write_text(json.dumps({"step_execution": "COMPLETE", "seeds": reports}, indent=2, sort_keys=True) + "\n"); print(json.dumps(report, indent=2, sort_keys=True)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
