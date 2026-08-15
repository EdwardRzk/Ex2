#!/usr/bin/env python3
"""Train frozen LSTM or Transformer sequence value baselines."""

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
from torch import nn

if __package__:
    from scripts.train_mambapo_value import build_sequence_tensors, state_statistics
    from scripts.train_mlp_value_baseline import (
        batch_pairwise_loss, correlation, git_metadata, pairwise_accuracy,
        program_slices, rankdata, read_records, seed_everything, write_json,
    )
else:
    from train_mambapo_value import build_sequence_tensors, state_statistics
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


class SequenceValueModel(nn.Module):
    def __init__(self, representation: Mapping[str, Any], model: Mapping[str, Any]):
        super().__init__()
        self.model_type = str(model["type"])
        d_model = int(model["d_model"])
        self.state_projection = nn.Linear(int(representation["state_dimension"]), d_model)
        self.action_embedding = nn.Embedding(
            int(representation["action_count"]) + 1, d_model
        )
        self.position_embedding = nn.Embedding(
            int(representation["max_sequence_length"]) + 1, d_model
        )
        self.input_norm = nn.LayerNorm(d_model)
        if self.model_type == "LSTM":
            self.encoder: nn.Module = nn.LSTM(
                input_size=d_model,
                hidden_size=d_model,
                num_layers=int(model["layers"]),
                dropout=float(model["dropout"]),
                batch_first=True,
            )
        elif self.model_type == "Transformer":
            layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=int(model["attention_heads"]),
                dim_feedforward=int(model["feedforward_dimension"]),
                dropout=float(model["dropout"]),
                activation=str(model["activation"]),
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(
                layer, num_layers=int(model["layers"]), norm=nn.LayerNorm(d_model)
            )
        else:
            raise ValueError(f"Unsupported sequence baseline: {self.model_type}")
        self.output_norm = nn.LayerNorm(d_model)
        self.value_head = nn.Linear(d_model, 1)

    def forward(
        self,
        states: torch.Tensor,
        previous_actions: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        positions = torch.arange(states.shape[1], device=states.device)
        hidden = self.input_norm(
            self.state_projection(states)
            + self.action_embedding(previous_actions)
            + self.position_embedding(positions)[None, :, :]
        )
        if self.model_type == "LSTM":
            encoded, _ = self.encoder(hidden)
        else:
            padding_mask = positions[None, :] >= lengths[:, None]
            encoded = self.encoder(hidden, src_key_padding_mask=padding_mask)
        final_hidden = encoded[
            torch.arange(encoded.shape[0], device=encoded.device), lengths - 1
        ]
        return self.value_head(self.output_norm(final_hidden)).squeeze(-1)


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        config = json.load(file)
    required = {
        "experiment_name", "hypothesis", "dataset", "representation", "model",
        "objective", "training", "metrics", "pass_fail_gate",
    }
    missing = required - config.keys()
    if missing:
        raise ValueError(f"Config is missing fields: {sorted(missing)}")
    if config["model"]["type"] not in {"LSTM", "Transformer"}:
        raise ValueError("Sequence baseline type must be LSTM or Transformer")
    if config["model"]["d_model"] != 128 or config["model"]["layers"] != 2:
        raise ValueError("Sequence baselines must use d_model=128 and two layers")
    if config["representation"]["max_sequence_length"] != 32:
        raise ValueError("MambaPO v0 maximum sequence length must remain 32")
    if config["training"]["early_stopping"] is not False:
        raise ValueError("Frozen v0 training must not use early stopping")
    return config


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
                ).cpu().numpy()
            )
    predicted = np.concatenate(predictions)
    actual = targets.cpu().numpy()
    errors = predicted - actual
    return {
        "mae": float(np.abs(errors).mean()),
        "rmse": float(np.sqrt(np.square(errors).mean())),
        "pearson": correlation(predicted, actual),
        "spearman": correlation(rankdata(predicted), rankdata(actual)),
        "same_program_pairwise_accuracy": pairwise_accuracy(predicted, actual, slices),
    }


def run_training(config: Mapping[str, Any], output_dir: Path) -> dict[str, Any]:
    training = config["training"]
    representation = config["representation"]
    seed_everything(int(training["seed"]))
    device = torch.device(training["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Frozen training requires CUDA, but CUDA is unavailable")

    train_records = read_records(Path(config["dataset"]["train_path"]))
    dev_records = read_records(Path(config["dataset"]["dev_path"]))
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
    model = SequenceValueModel(representation, config["model"]).to(device)
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
            model, dev_states, dev_actions, dev_lengths, dev_targets, dev_slices,
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
        "model_type": config["model"]["type"],
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
        "all_dev_metrics_finite": all(math.isfinite(value) for value in final_metrics.values()),
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
            "type": config["model"]["type"],
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
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
