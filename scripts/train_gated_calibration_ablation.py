#!/usr/bin/env python3
"""Train fixed Gate/KL ablations on frozen Route-A train/validation artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

if __package__:
    from scripts.train_controlled_nvp_stage_a import learning_rate, soft_cross_entropy
    from scripts.train_gated_calibrated_mambanvp import GatedCalibratedMambaNVP, diagnostics, frozen_references, kl_final_to_nvp, seed_everything
    from scripts.train_mamba_nvp_objecttext import load_feature_cache, load_frozen_nvp
    from scripts.train_set_conditioned_mamba_ranker import K, load_candidates, load_json, read_jsonl, read_label_matrix
else:
    from train_controlled_nvp_stage_a import learning_rate, soft_cross_entropy
    from train_gated_calibrated_mambanvp import GatedCalibratedMambaNVP, diagnostics, frozen_references, kl_final_to_nvp, seed_everything
    from train_mamba_nvp_objecttext import load_feature_cache, load_frozen_nvp
    from train_set_conditioned_mamba_ranker import K, load_candidates, load_json, read_jsonl, read_label_matrix

METHOD = "GatedCalibrationAblation"
VARIANTS = ("gated_full", "no_kl", "no_gate", "no_gate_no_kl")


class AblatedGatedCalibratedMambaNVP(GatedCalibratedMambaNVP):
    """The no-gate variants remove the gate module from the computation and state."""
    def __init__(self, nvp: torch.nn.Module, model_cfg: Mapping[str, Any], tokens: torch.Tensor, lengths: torch.Tensor, use_gate: bool) -> None:
        super().__init__(nvp, model_cfg, tokens, lengths)
        self.use_gate = use_gate
        if not use_gate:
            del self.gate

    def components(self, program: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            nvp_logits = self.nvp(program)
        embeddings = self.embeddings(program)
        residual = self.residual_head(embeddings).squeeze(-1)
        if self.use_gate:
            gate_input = torch.cat([embeddings, program[:, None, :].expand(-1, K, -1)], dim=-1)
            alpha = torch.sigmoid(self.gate(gate_input).squeeze(-1))
        else:
            alpha = torch.ones_like(residual)
        return nvp_logits, residual, alpha, nvp_logits + alpha * residual


def validate_config(cfg: Mapping[str, Any], controlled: Mapping[str, Any]) -> None:
    if cfg["final_seed_set"] != [1, 2, 3] or cfg["frozen_data_population"] != {"train_complete_k50": 28159, "validation_complete_k50": 4488}:
        raise ValueError("frozen population or seeds mismatch")
    if cfg["target_and_objective"]["target_temperature"] != 0.05:
        raise ValueError("target temperature changed")
    train = cfg["training"]
    if train["total_steps"] != 10000 or train["checkpoint_evaluation_cadence_steps"] != 100 or train["early_stopping"]:
        raise ValueError("training budget changed")
    if cfg["validation"] != {"sampling": False, "ranking": "descending final logits; candidate ID ascending tie break", "scored_pass_budget": 45, "selection_metric": "ValidationFinalMeanOverOz policy-45 dataset macro mean", "final_test_accessed": False, "ood_accessed": False, "runtime_accessed": False}:
        raise ValueError("validation protocol changed")
    candidate = controlled["candidate_representation"]
    if (candidate["K"], candidate["padded_length"], candidate["pad_token_id"], candidate["vocabulary_size"]) != (50, 20, 124, 125):
        raise ValueError("candidate representation changed")
    expected = {
        "gated_full": (True, 0.1), "no_kl": (True, 0.0), "no_gate": (False, 0.1), "no_gate_no_kl": (False, 0.0),
    }
    if set(cfg["variants"]) != set(VARIANTS) or any((cfg["variants"][name]["use_gate"], cfg["variants"][name]["lambda_kl"]) != pair for name, pair in expected.items()):
        raise ValueError("ablation definitions changed")


def train_seed(cfg: Mapping[str, Any], controlled: Mapping[str, Any], variant_name: str, variant: Mapping[str, Any], seed: int, tokens: torch.Tensor, lengths: torch.Tensor, train_x: torch.Tensor, train_y: torch.Tensor, val_x: torch.Tensor, val_y: torch.Tensor, validation: Sequence[Mapping[str, Any]], matrix: Mapping[str, Sequence[Mapping[str, Any]]], output: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    seed_everything(seed)
    model_cfg = {**controlled["candidate_representation"], **controlled["models"]["Mamba"]}
    checkpoint = Path(cfg["nvp_checkpoint_root"]) / f"seed{seed}" / "model.pt"
    model = AblatedGatedCalibratedMambaNVP(load_frozen_nvp(checkpoint, seed), model_cfg, tokens, lengths, bool(variant["use_gate"])).cuda()
    if any(parameter.requires_grad for parameter in model.nvp.parameters()) or model.nvp.training:
        raise RuntimeError("frozen NVP invariant failed")
    optimizer = torch.optim.Adam((parameter for parameter in model.parameters() if parameter.requires_grad), lr=float(cfg["training"]["learning_rate"]), weight_decay=float(cfg["training"]["weight_decay"]))
    generator = torch.Generator().manual_seed(seed)
    output.mkdir(parents=True)
    best: dict[str, Any] | None = None; curve: list[dict[str, Any]] = []; step = 0
    while step < int(cfg["training"]["total_steps"]):
        order = torch.randperm(len(train_x), generator=generator)
        for begin in range(0, len(train_x), int(cfg["training"]["batch_size"])):
            step += 1; index = order[begin:begin + int(cfg["training"]["batch_size"])].cuda(non_blocking=True); lr = learning_rate(cfg["training"], step)
            for group in optimizer.param_groups: group["lr"] = lr
            model.train(); nvp_logits, _, _, final_logits = model.components(train_x[index])
            value_loss = soft_cross_entropy(final_logits, train_y[index]); calibration = kl_final_to_nvp(final_logits, nvp_logits)
            loss = value_loss + float(variant["lambda_kl"]) * calibration
            optimizer.zero_grad(set_to_none=True); loss.backward()
            if any(parameter.grad is not None for parameter in model.nvp.parameters()):
                raise RuntimeError("frozen NVP branch received gradients")
            optimizer.step()
            if step % int(cfg["training"]["checkpoint_evaluation_cadence_steps"]) == 0 or step == int(cfg["training"]["total_steps"]):
                metric = diagnostics(model, val_x, val_y, validation, matrix, int(cfg["training"]["evaluation_batch_size"]))
                metric.update({"step": step, "train_total_loss": float(loss.detach().cpu()), "train_value_loss": float(value_loss.detach().cpu()), "train_calibration_kl_final_to_nvp": float(calibration.detach().cpu()), "lr": lr})
                curve.append(metric); print(json.dumps({"architecture": METHOD, "variant": variant_name, "seed": seed, **metric}, sort_keys=True), flush=True)
                if best is None or metric["ValidationFinalMeanOverOz"] > best["ValidationFinalMeanOverOz"]:
                    best = metric
                    torch.save({"stage": "Route-A Gated Calibration Ablation v1", "architecture": METHOD, "variant": variant_name, "seed": seed, "step": step, "metrics": metric, "state_dict": model.state_dict(), "model_config": model_cfg, "nvp_checkpoint": str(checkpoint), "nvp_frozen": True, "use_gate": bool(variant["use_gate"]), "lambda_kl": float(variant["lambda_kl"]), "fusion": variant["formula"]}, output / "model.pt")
            if step == int(cfg["training"]["total_steps"]): break
    if best is None: raise RuntimeError("no checkpoint evaluated")
    return {"architecture": METHOD, "variant": variant_name, "seed": seed, "step_execution": "COMPLETE", "trainable_parameters": model.trainable_parameter_count(), "nvp_frozen": True, "use_gate": bool(variant["use_gate"]), "lambda_kl": float(variant["lambda_kl"]), "selection_metric": "ValidationFinalMeanOverOz policy-45 dataset macro mean", "selected": best}, curve


def variant_summary(reports: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    selected = [item["selected"] for item in reports]
    fields = {"ValidationFinalMeanOverOz_3seed": "ValidationFinalMeanOverOz", "oracle_recovery_3seed": None, "policy45_regret_mean_bytes_3seed": "policy45_regret_mean_bytes", "top1_accuracy_3seed": "top1_accuracy", "validation_ce_3seed": "validation_ce", "calibration_kl_final_to_nvp_3seed": "calibration_kl_final_to_nvp", "average_gate_alpha_3seed": "average_gate_alpha"}
    result = {name: (sum(float(row[field]) for row in selected) / len(selected) if field else None) for name, field in fields.items()}
    result["trainable_parameters"] = reports[0]["trainable_parameters"]
    result["seed_results"] = list(reports)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", type=Path, required=True); parser.add_argument("--output-dir", type=Path, required=True); args = parser.parse_args()
    if args.output_dir.exists(): raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    if not torch.cuda.is_available(): raise RuntimeError("formal ablation training requires CUDA")
    cfg = load_json(args.config); controlled = load_json(Path(cfg["candidate_representation_source"])); validate_config(cfg, controlled)
    train, validation = read_jsonl(Path(cfg["target_files"]["train"])), read_jsonl(Path(cfg["target_files"]["validation"]))
    if len(train) != 28159 or len(validation) != 4488: raise ValueError("frozen train/validation population mismatch")
    matrix = read_label_matrix(Path(cfg["validation_label_shards"]))
    if set(matrix) != {row["program_id"] for row in validation}: raise ValueError("validation K50 labels mismatch")
    tokens, lengths = load_candidates(Path(controlled["candidate_representation"]["candidate_sequences"]), pad_token_id=124, padded_length=20)
    train_features = load_feature_cache(Path(cfg["autophase_feature_cache"]["train"]), "train", [row["program_id"] for row in train]); val_features = load_feature_cache(Path(cfg["autophase_feature_cache"]["validation"]), "validation", [row["program_id"] for row in validation])
    train_x = torch.tensor([train_features[row["program_id"]] for row in train], dtype=torch.float32, device="cuda"); train_y = torch.tensor([row["normalized_target"] for row in train], dtype=torch.float32, device="cuda")
    val_x = torch.tensor([val_features[row["program_id"]] for row in validation], dtype=torch.float32, device="cuda"); val_y = torch.tensor([row["normalized_target"] for row in validation], dtype=torch.float32, device="cuda")
    args.output_dir.mkdir(parents=True); (args.output_dir / "config.json").write_text(json.dumps(cfg, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    combined: dict[str, Any] = {}
    references = frozen_references({**load_json(Path(cfg["base_training_config"])), "frozen_reference_reports": cfg["frozen_reference_reports"]})
    for variant_name in VARIANTS:
        variant = cfg["variants"][variant_name]; variant_dir = args.output_dir / variant_name
        reports, curves = [], {}
        for seed in cfg["final_seed_set"]:
            report, curve = train_seed(cfg, controlled, variant_name, variant, seed, tokens.cuda(), lengths.cuda(), train_x, train_y, val_x, val_y, validation, matrix, variant_dir / "checkpoints" / f"seed{seed}")
            reports.append(report); curves[str(seed)] = curve
        summary = variant_summary(reports)
        stage = {"step_execution": "COMPLETE", "training_only": True, "final_test_accessed": False, "ood_accessed": False, "runtime_accessed": False, "compiler_gym_initialized": False, "llvm_execution": False, "candidate_rollouts": 0, "objecttext_measurements": 0, "label_regeneration": False, "validation_cohort": {"N_total": 4488, "N_primary_valid": 4488, "N_failed_or_invalid": 0}, "variant": dict(variant), "summary": summary, "frozen_references": references, "differences": {name: summary["ValidationFinalMeanOverOz_3seed"] - value for name, value in references.items()}}
        (variant_dir / "training_report.json").write_text(json.dumps({"step_execution": "COMPLETE", "variant": variant, "seeds": reports, "training_curve": curves}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (variant_dir / "comparison_report.json").write_text(json.dumps(stage, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        combined[variant_name] = stage
    root_report = {"step_execution": "COMPLETE", "training_only": True, "final_test_accessed": False, "ood_accessed": False, "runtime_accessed": False, "compiler_gym_initialized": False, "llvm_execution": False, "candidate_rollouts": 0, "objecttext_measurements": 0, "label_regeneration": False, "validation_cohort": {"N_total": 4488, "N_primary_valid": 4488, "N_failed_or_invalid": 0}, "variants": combined, "frozen_references": references}
    (args.output_dir / "comparison_report.json").write_text(json.dumps(root_report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(root_report, indent=2, sort_keys=True)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
