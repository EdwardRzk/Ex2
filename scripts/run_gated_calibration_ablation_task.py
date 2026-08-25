#!/usr/bin/env python3
"""Run one fixed Gate/KL ablation task with exact resumable optimizer/RNG state."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import torch

if __package__:
    from scripts.train_controlled_nvp_stage_a import learning_rate, soft_cross_entropy
    from scripts.train_gated_calibration_ablation import AblatedGatedCalibratedMambaNVP, METHOD, VARIANTS, validate_config
    from scripts.train_gated_calibrated_mambanvp import diagnostics, seed_everything, kl_final_to_nvp
    from scripts.train_mamba_nvp_objecttext import load_feature_cache, load_frozen_nvp
    from scripts.train_set_conditioned_mamba_ranker import load_candidates, load_json, read_jsonl, read_label_matrix
else:
    from train_controlled_nvp_stage_a import learning_rate, soft_cross_entropy
    from train_gated_calibration_ablation import AblatedGatedCalibratedMambaNVP, METHOD, VARIANTS, validate_config
    from train_gated_calibrated_mambanvp import diagnostics, seed_everything, kl_final_to_nvp
    from train_mamba_nvp_objecttext import load_candidates, load_json, read_jsonl, read_label_matrix
    from train_mamba_nvp_objecttext import load_feature_cache, load_frozen_nvp


def load_data(cfg: Mapping[str, Any], controlled: Mapping[str, Any]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[dict[str, Any]], dict[str, list[dict[str, Any]]], torch.Tensor, torch.Tensor]:
    train, validation = read_jsonl(Path(cfg["target_files"]["train"])), read_jsonl(Path(cfg["target_files"]["validation"]))
    if len(train) != 28159 or len(validation) != 4488:
        raise ValueError("frozen train/validation population mismatch")
    matrix = read_label_matrix(Path(cfg["validation_label_shards"]))
    if set(matrix) != {row["program_id"] for row in validation}:
        raise ValueError("frozen validation K50 matrix mismatch")
    tokens, lengths = load_candidates(Path(controlled["candidate_representation"]["candidate_sequences"]), pad_token_id=124, padded_length=20)
    train_features = load_feature_cache(Path(cfg["autophase_feature_cache"]["train"]), "train", [row["program_id"] for row in train])
    val_features = load_feature_cache(Path(cfg["autophase_feature_cache"]["validation"]), "validation", [row["program_id"] for row in validation])
    train_x = torch.tensor([train_features[row["program_id"]] for row in train], dtype=torch.float32, device="cuda")
    train_y = torch.tensor([row["normalized_target"] for row in train], dtype=torch.float32, device="cuda")
    val_x = torch.tensor([val_features[row["program_id"]] for row in validation], dtype=torch.float32, device="cuda")
    val_y = torch.tensor([row["normalized_target"] for row in validation], dtype=torch.float32, device="cuda")
    return train_x, train_y, val_x, val_y, validation, matrix, tokens.cuda(), lengths.cuda()


def checkpoint_payload(*, variant_name: str, variant: Mapping[str, Any], seed: int, step: int, model: AblatedGatedCalibratedMambaNVP, optimizer: torch.optim.Optimizer, generator: torch.Generator, order: torch.Tensor | None, next_begin: int, best: Mapping[str, Any] | None, curve: list[dict[str, Any]], model_cfg: Mapping[str, Any], nvp_checkpoint: Path) -> dict[str, Any]:
    return {"stage": "Route-A Gated Calibration Ablation v1 parallel-resume", "architecture": METHOD, "variant": variant_name, "seed": seed, "step": step, "use_gate": bool(variant["use_gate"]), "lambda_kl": float(variant["lambda_kl"]), "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "generator_state": generator.get_state(), "current_epoch_order": None if order is None else order.cpu(), "next_batch_begin": next_begin, "best": best, "curve": curve, "model_config": dict(model_cfg), "nvp_checkpoint": str(nvp_checkpoint), "nvp_frozen": True, "fusion": variant["formula"]}


def validate_state(payload: Mapping[str, Any], variant_name: str, variant: Mapping[str, Any], seed: int) -> None:
    expected = (variant_name, seed, bool(variant["use_gate"]), float(variant["lambda_kl"]))
    actual = (payload.get("variant"), payload.get("seed"), payload.get("use_gate"), payload.get("lambda_kl"))
    if actual != expected or payload.get("nvp_frozen") is not True:
        raise ValueError("resume state does not match frozen task")


def run_task(cfg: Mapping[str, Any], controlled: Mapping[str, Any], variant_name: str, seed: int, attempt_dir: Path, *, fresh_reconstruction: bool) -> dict[str, Any]:
    variant = cfg["variants"][variant_name]
    train_x, train_y, val_x, val_y, validation, matrix, tokens, lengths = load_data(cfg, controlled)
    model_cfg = {**controlled["candidate_representation"], **controlled["models"]["Mamba"]}
    nvp_checkpoint = Path(cfg["nvp_checkpoint_root"]) / f"seed{seed}" / "model.pt"
    state_path, best_path, report_path = attempt_dir / "training_state.pt", attempt_dir / "model.pt", attempt_dir / "task_report.json"
    attempt_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(seed)
    model = AblatedGatedCalibratedMambaNVP(load_frozen_nvp(nvp_checkpoint, seed), model_cfg, tokens, lengths, bool(variant["use_gate"])).cuda()
    optimizer = torch.optim.Adam((parameter for parameter in model.parameters() if parameter.requires_grad), lr=float(cfg["training"]["learning_rate"]), weight_decay=float(cfg["training"]["weight_decay"]))
    generator = torch.Generator().manual_seed(seed)
    step, order, next_begin, best, curve = 0, None, 0, None, []
    resumed = False
    if state_path.exists() and not fresh_reconstruction:
        payload = torch.load(state_path, map_location="cpu", weights_only=False); validate_state(payload, variant_name, variant, seed)
        model.load_state_dict(payload["model_state_dict"], strict=True); optimizer.load_state_dict(payload["optimizer_state_dict"]); generator.set_state(payload["generator_state"])
        step, order, next_begin, best, curve = int(payload["step"]), payload["current_epoch_order"], int(payload["next_batch_begin"]), payload["best"], list(payload["curve"]); resumed = True
    elif report_path.exists() or best_path.exists():
        raise FileExistsError(f"attempt directory already has final artifacts: {attempt_dir}")
    if any(parameter.requires_grad for parameter in model.nvp.parameters()) or model.nvp.training:
        raise RuntimeError("frozen NVP invariant failed")
    while step < int(cfg["training"]["total_steps"]):
        if order is None:
            order = torch.randperm(len(train_x), generator=generator); next_begin = 0
        index = order[next_begin:next_begin + int(cfg["training"]["batch_size"])].cuda(non_blocking=True)
        if not len(index):
            order = None; next_begin = 0; continue
        step += 1; next_begin += int(cfg["training"]["batch_size"])
        if next_begin >= len(train_x): order = None; next_begin = 0
        lr = learning_rate(cfg["training"], step)
        for group in optimizer.param_groups: group["lr"] = lr
        model.train(); nvp_logits, _, _, final_logits = model.components(train_x[index])
        value_loss = soft_cross_entropy(final_logits, train_y[index]); calibration = kl_final_to_nvp(final_logits, nvp_logits)
        loss = value_loss + float(variant["lambda_kl"]) * calibration
        optimizer.zero_grad(set_to_none=True); loss.backward()
        if any(parameter.grad is not None for parameter in model.nvp.parameters()): raise RuntimeError("frozen NVP branch received gradients")
        optimizer.step()
        if step % int(cfg["training"]["checkpoint_evaluation_cadence_steps"]) == 0 or step == int(cfg["training"]["total_steps"]):
            metric = diagnostics(model, val_x, val_y, validation, matrix, int(cfg["training"]["evaluation_batch_size"]))
            metric.update({"step": step, "train_total_loss": float(loss.detach().cpu()), "train_value_loss": float(value_loss.detach().cpu()), "train_calibration_kl_final_to_nvp": float(calibration.detach().cpu()), "lr": lr})
            curve.append(metric)
            if best is None or metric["ValidationFinalMeanOverOz"] > best["ValidationFinalMeanOverOz"]:
                best = metric
                torch.save({"stage": "Route-A Gated Calibration Ablation v1", "architecture": METHOD, "variant": variant_name, "seed": seed, "step": step, "metrics": metric, "state_dict": model.state_dict(), "model_config": model_cfg, "nvp_checkpoint": str(nvp_checkpoint), "nvp_frozen": True, "use_gate": bool(variant["use_gate"]), "lambda_kl": float(variant["lambda_kl"]), "fusion": variant["formula"], "attempt": attempt_dir.name}, best_path)
            torch.save(checkpoint_payload(variant_name=variant_name, variant=variant, seed=seed, step=step, model=model, optimizer=optimizer, generator=generator, order=order, next_begin=next_begin, best=best, curve=curve, model_cfg=model_cfg, nvp_checkpoint=nvp_checkpoint), state_path)
            print(json.dumps({"architecture": METHOD, "variant": variant_name, "seed": seed, "attempt": attempt_dir.name, "resumed": resumed, **metric}, sort_keys=True), flush=True)
    if best is None: raise RuntimeError("no validation checkpoint evaluated")
    result = {"step_execution": "COMPLETE", "training_only": True, "final_test_accessed": False, "ood_accessed": False, "runtime_accessed": False, "compiler_gym_initialized": False, "llvm_execution": False, "candidate_rollouts": 0, "objecttext_measurements": 0, "label_regeneration": False, "variant": dict(variant), "seed": seed, "attempt": attempt_dir.name, "fresh_reconstruction": fresh_reconstruction, "resumed_from_training_state": resumed, "trainable_parameters": model.trainable_parameter_count(), "selected": best, "curve": curve}
    report_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", type=Path, required=True); parser.add_argument("--output-dir", type=Path, required=True); parser.add_argument("--variant", choices=VARIANTS, required=True); parser.add_argument("--seed", type=int, choices=(1, 2, 3), required=True); parser.add_argument("--attempt", required=True); parser.add_argument("--fresh-reconstruction", action="store_true"); args = parser.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError("formal ablation training requires CUDA")
    cfg = load_json(args.config); controlled = load_json(Path(cfg["candidate_representation_source"])); validate_config(cfg, controlled)
    attempt = args.output_dir / args.variant / "checkpoints" / f"seed{args.seed}" / args.attempt
    result = run_task(cfg, controlled, args.variant, args.seed, attempt, fresh_reconstruction=args.fresh_reconstruction)
    print(json.dumps({"task_complete": result["step_execution"], "variant": args.variant, "seed": args.seed, "selected_step": result["selected"]["step"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
