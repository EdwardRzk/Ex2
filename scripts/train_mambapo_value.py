#!/usr/bin/env python3
"""Train the frozen two-layer MambaPO v0 sequence value model."""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from mamba_ssm import Mamba
from torch import nn

if __package__:
    from scripts.train_mlp_value_baseline import (
        batch_pairwise_loss, correlation, git_metadata, pairwise_accuracy,
        program_slices, rankdata, read_records, seed_everything, write_json,
    )
else:
    from train_mlp_value_baseline import (
        batch_pairwise_loss,
        correlation,
        git_metadata,
        pairwise_accuracy,
        program_slices,
        rankdata,
        read_records,
        seed_everything,
        write_json,
    )


class MambaValueModel(nn.Module):
    def __init__(self, representation: Mapping[str, Any], model: Mapping[str, Any]):
        super().__init__()
        d_model = int(model["d_model"])
        self.state_projection = nn.Linear(int(representation["state_dimension"]), d_model)
        self.action_embedding = nn.Embedding(
            int(representation["action_count"]) + 1, d_model
        )
        self.position_embedding = nn.Embedding(
            int(representation["max_sequence_length"]) + 1, d_model
        )
        self.input_norm = nn.LayerNorm(d_model)
        self.block_norms = nn.ModuleList(
            [nn.LayerNorm(d_model) for _ in range(int(model["layers"]))]
        )
        self.blocks = nn.ModuleList(
            [
                Mamba(
                    d_model=d_model,
                    d_state=int(model["d_state"]),
                    d_conv=int(model["d_conv"]),
                    expand=int(model["expand"]),
                    use_fast_path=bool(model["use_fast_path"]),
                    layer_idx=index,
                )
                for index in range(int(model["layers"]))
            ]
        )
        self.output_norm = nn.LayerNorm(d_model)
        self.value_head = nn.Linear(d_model, 1)

    def forward(
        self,
        states: torch.Tensor,
        previous_actions: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        positions = torch.arange(states.shape[1], device=states.device)
        hidden = (
            self.state_projection(states)
            + self.action_embedding(previous_actions)
            + self.position_embedding(positions)[None, :, :]
        )
        hidden = self.input_norm(hidden)
        for norm, block in zip(self.block_norms, self.blocks):
            hidden = hidden + block(norm(hidden))
        final_hidden = hidden[
            torch.arange(hidden.shape[0], device=hidden.device), lengths - 1
        ]
        return self.value_head(self.output_norm(final_hidden)).squeeze(-1)


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        config = json.load(file)
    required = {
        "experiment_name",
        "hypothesis",
        "dataset",
        "representation",
        "model",
        "objective",
        "training",
        "metrics",
        "pass_fail_gate",
    }
    missing = required - config.keys()
    if missing:
        raise ValueError(f"Config is missing fields: {sorted(missing)}")
    if config["model"]["d_model"] != 128 or config["model"]["layers"] != 2:
        raise ValueError("MambaPO v0 must use d_model=128 and two layers")
    if config["representation"]["max_sequence_length"] != 32:
        raise ValueError("MambaPO v0 maximum sequence length must remain 32")
    if config["representation"]["start_action_index"] != config["representation"]["action_count"]:
        raise ValueError("START must immediately follow the LLVM action indices")
    if config["training"]["early_stopping"] is not False:
        raise ValueError("Frozen v0 training must not use early stopping")
    return config


def state_statistics(
    records: list[dict[str, Any]], state_dimension: int
) -> tuple[np.ndarray, np.ndarray]:
    total = np.zeros(state_dimension, dtype=np.float64)
    total_squared = np.zeros(state_dimension, dtype=np.float64)
    count = 0
    for record in records:
        states = np.asarray(record["states"], dtype=np.float64)
        if states.shape[1] != state_dimension:
            raise ValueError("Dataset state dimension does not match frozen config")
        total += states.sum(axis=0)
        total_squared += np.square(states).sum(axis=0)
        count += len(states)
    mean = total / count
    variance = np.maximum(total_squared / count - np.square(mean), 0)
    std = np.sqrt(variance)
    std[std < 1e-6] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def build_sequence_tensors(
    records: list[dict[str, Any]],
    *,
    state_mean: np.ndarray,
    state_std: np.ndarray,
    action_count: int,
    start_action_index: int,
    max_sequence_length: int,
    target_name: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[str]]:
    ordered = sorted(records, key=lambda row: (row["program_id"], row["trajectory_index"]))
    token_count = max_sequence_length + 1
    states = np.zeros((len(ordered), token_count, len(state_mean)), dtype=np.float32)
    actions = np.full((len(ordered), token_count), start_action_index, dtype=np.int64)
    lengths = np.zeros(len(ordered), dtype=np.int64)
    targets = np.zeros(len(ordered), dtype=np.float32)
    programs: list[str] = []
    for index, record in enumerate(ordered):
        sequence_length = int(record["sequence_length"])
        if sequence_length > max_sequence_length:
            raise ValueError("Dataset sequence exceeds frozen maximum")
        if any(action < 0 or action >= action_count for action in record["action_indices"]):
            raise ValueError("Dataset action is outside the frozen action space")
        current_states = np.asarray(record["states"], dtype=np.float32)
        length = sequence_length + 1
        states[index, :length] = (current_states - state_mean) / state_std
        actions[index, 0] = start_action_index
        actions[index, 1:length] = record["action_indices"]
        lengths[index] = length
        targets[index] = float(record[target_name])
        programs.append(str(record["program_id"]))
    return (
        torch.from_numpy(states),
        torch.from_numpy(actions),
        torch.from_numpy(lengths),
        torch.from_numpy(targets),
        programs,
    )


def evaluate_model(
    model: nn.Module,
    states: torch.Tensor,
    actions: torch.Tensor,
    lengths: torch.Tensor,
    targets: torch.Tensor,
    slices: list[slice],
    batch_size: int,
) -> dict[str, float]:
    model.eval()
    predictions: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(states), batch_size):
            predictions.append(
                model(
                    states[start : start + batch_size],
                    actions[start : start + batch_size],
                    lengths[start : start + batch_size],
                )
                .float()
                .cpu()
                .numpy()
            )
    predicted = np.concatenate(predictions)
    actual = targets.cpu().numpy()
    errors = predicted - actual
    return {
        "mae": float(np.abs(errors).mean()),
        "rmse": float(np.sqrt(np.square(errors).mean())),
        "pearson": correlation(predicted, actual),
        "spearman": correlation(rankdata(predicted), rankdata(actual)),
        "same_program_pairwise_accuracy": pairwise_accuracy(
            predicted, actual, slices
        ),
    }


def run_training(config: Mapping[str, Any], output_dir: Path) -> dict[str, Any]:
    training = config["training"]
    representation = config["representation"]
    seed_everything(int(training["seed"]))
    device = torch.device(training["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Frozen training requires CUDA, but CUDA is unavailable")

    train_records = read_records(config["dataset"]["train_path"])
    dev_records = read_records(config["dataset"]["dev_path"])
    mean, std = state_statistics(train_records, int(representation["state_dimension"]))
    tensor_args = {
        "state_mean": mean,
        "state_std": std,
        "action_count": int(representation["action_count"]),
        "start_action_index": int(representation["start_action_index"]),
        "max_sequence_length": int(representation["max_sequence_length"]),
        "target_name": config["dataset"]["target"],
    }
    train_states, train_actions, train_lengths, train_targets, train_programs = build_sequence_tensors(
        train_records, **tensor_args
    )
    dev_states, dev_actions, dev_lengths, dev_targets, dev_programs = build_sequence_tensors(
        dev_records, **tensor_args
    )
    if set(train_programs) & set(dev_programs):
        raise ValueError("Train and dev programs overlap")
    train_slices = program_slices(train_programs)
    dev_slices = program_slices(dev_programs)
    group_size = len(train_records) // len(train_slices)
    if any(group.stop - group.start != group_size for group in train_slices + dev_slices):
        raise ValueError("Every program must have the same trajectory count")

    train_states = train_states.to(device)
    train_actions = train_actions.to(device)
    train_lengths = train_lengths.to(device)
    train_targets = train_targets.to(device)
    dev_states = dev_states.to(device)
    dev_actions = dev_actions.to(device)
    dev_lengths = dev_lengths.to(device)
    dev_targets = dev_targets.to(device)
    model = MambaValueModel(representation, config["model"]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    regression_loss = nn.HuberLoss()
    pairwise_weight = float(config["objective"]["pairwise_weight"])
    programs_per_batch = int(training["programs_per_batch"])
    generator = torch.Generator().manual_seed(int(training["seed"]))
    curve: list[dict[str, Any]] = []

    for epoch in range(1, int(training["epochs"]) + 1):
        model.train()
        order = torch.randperm(len(train_slices), generator=generator).tolist()
        losses: list[float] = []
        for offset in range(0, len(order), programs_per_batch):
            indices: list[int] = []
            for program_index in order[offset : offset + programs_per_batch]:
                group = train_slices[program_index]
                indices.extend(range(group.start, group.stop))
            index = torch.tensor(indices, device=device)
            predictions = model(
                train_states[index], train_actions[index], train_lengths[index]
            )
            targets = train_targets[index]
            loss = regression_loss(predictions, targets) + pairwise_weight * batch_pairwise_loss(
                predictions, targets, group_size
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        dev_metrics = evaluate_model(
            model,
            dev_states,
            dev_actions,
            dev_lengths,
            dev_targets,
            dev_slices,
            int(training["evaluation_batch_size"]),
        )
        epoch_result = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "dev": dev_metrics,
        }
        curve.append(epoch_result)
        print(json.dumps(epoch_result, sort_keys=True), flush=True)

    checkpoint = {
        "model_type": "Mamba",
        "model_state_dict": model.state_dict(),
        "representation": dict(representation),
        "model_config": dict(config["model"]),
        "state_mean": torch.from_numpy(mean),
        "state_std": torch.from_numpy(std),
        "target": config["dataset"]["target"],
    }
    checkpoint_path = output_dir / "model.pt"
    torch.save(checkpoint, checkpoint_path)
    write_json(output_dir / "learning_curve.json", curve)
    final_metrics = curve[-1]["dev"]
    checks = {
        "completed_epochs": len(curve) == config["pass_fail_gate"]["completed_epochs"],
        "all_dev_metrics_finite": all(
            math.isfinite(value) for value in final_metrics.values()
        ),
        "checkpoint_written": checkpoint_path.is_file(),
    }
    return {
        "dataset": {
            "train_trajectories": len(train_records),
            "dev_trajectories": len(dev_records),
            "train_programs": len(train_slices),
            "dev_programs": len(dev_slices),
        },
        "model": {
            "parameter_count": sum(parameter.numel() for parameter in model.parameters())
        },
        "final_dev": final_metrics,
        "checks": checks,
        "decision": "PASS" if all(checks.values()) else "FAIL",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    config = load_config(args.config.resolve())
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    write_json(output_dir / "config.json", config)
    report: dict[str, Any] = {
        "experiment_name": config["experiment_name"],
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "git": git_metadata(repo_root),
        "host": {
            "platform": platform.platform(),
            "python": sys.version,
            "torch": torch.__version__,
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "decision": "FAIL",
    }
    try:
        import mamba_ssm

        report["host"]["mamba_ssm"] = mamba_ssm.__version__
        report.update(run_training(config, output_dir))
    except Exception as error:
        report["error"] = {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        }
    finally:
        report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(output_dir / "experiment_report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["decision"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
