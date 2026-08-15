#!/usr/bin/env python3
"""Train the frozen order-agnostic MambaPO MLP value baseline."""

from __future__ import annotations

import argparse
import gzip
import json
import math
import platform
import random
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn


class MlpValueModel(nn.Module):
    def __init__(self, input_dimension: int, hidden_dimensions: list[int]):
        super().__init__()
        layers: list[nn.Module] = []
        current = input_dimension
        for hidden in hidden_dimensions:
            layers.extend((nn.Linear(current, hidden), nn.ReLU()))
            current = hidden
        layers.append(nn.Linear(current, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(-1)


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
    training = config["training"]
    if training["early_stopping"] is not False:
        raise ValueError("Frozen v0 training must not use early stopping")
    if training["epochs"] < 1 or training["programs_per_batch"] < 1:
        raise ValueError("epochs and programs_per_batch must be positive")
    if config["dataset"]["target"] != "size_reduction_vs_oz":
        raise ValueError("MLP v0 target must remain size_reduction_vs_oz")
    return config


def read_records(paths: str | Path | list[str]) -> list[dict[str, Any]]:
    if isinstance(paths, (str, Path)):
        paths = [paths]
    records: list[dict[str, Any]] = []
    for path in paths:
        with gzip.open(Path(path), "rt", encoding="utf-8") as file:
            records.extend(json.loads(line) for line in file)
    return records


def build_feature(record: Mapping[str, Any], action_count: int, max_length: int) -> np.ndarray:
    final_state = np.asarray(record["states"][-1], dtype=np.float32)
    histogram = np.bincount(
        np.asarray(record["action_indices"], dtype=np.int64), minlength=action_count
    ).astype(np.float32)
    histogram /= max(float(record["sequence_length"]), 1.0)
    length = np.asarray([record["sequence_length"] / max_length], dtype=np.float32)
    return np.concatenate((final_state, histogram, length))


def group_records(
    records: list[dict[str, Any]], action_count: int, max_length: int, target_name: str
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    ordered = sorted(records, key=lambda row: (row["program_id"], row["trajectory_index"]))
    features = np.stack(
        [build_feature(row, action_count, max_length) for row in ordered]
    )
    targets = np.asarray([row[target_name] for row in ordered], dtype=np.float32)
    programs = [str(row["program_id"]) for row in ordered]
    return features, targets, programs


def normalize_features(
    train: np.ndarray, dev: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = train.mean(axis=0)
    std = train.std(axis=0)
    std[std < 1e-6] = 1.0
    return (train - mean) / std, (dev - mean) / std, mean, std


def program_slices(programs: list[str]) -> list[slice]:
    slices: list[slice] = []
    start = 0
    for index in range(1, len(programs) + 1):
        if index == len(programs) or programs[index] != programs[start]:
            slices.append(slice(start, index))
            start = index
    return slices


def pairwise_ranking_loss(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    prediction_difference = predictions[:, None] - predictions[None, :]
    target_difference = targets[:, None] - targets[None, :]
    upper = torch.triu(torch.ones_like(target_difference, dtype=torch.bool), diagonal=1)
    comparable = upper & (target_difference != 0)
    if not comparable.any():
        return predictions.sum() * 0
    signs = target_difference[comparable].sign()
    return torch.nn.functional.softplus(
        -signs * prediction_difference[comparable]
    ).mean()


def batch_pairwise_loss(
    predictions: torch.Tensor, targets: torch.Tensor, group_size: int
) -> torch.Tensor:
    losses = [
        pairwise_ranking_loss(
            predictions[start : start + group_size], targets[start : start + group_size]
        )
        for start in range(0, len(predictions), group_size)
    ]
    return torch.stack(losses).mean()


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2
        start = end
    return ranks


def correlation(left: np.ndarray, right: np.ndarray) -> float:
    if np.std(left) == 0 or np.std(right) == 0:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def pairwise_accuracy(
    predictions: np.ndarray, targets: np.ndarray, slices: list[slice]
) -> float:
    correct = 0
    total = 0
    for group in slices:
        prediction_difference = predictions[group, None] - predictions[None, group]
        target_difference = targets[group, None] - targets[None, group]
        upper = np.triu(np.ones_like(target_difference, dtype=bool), k=1)
        comparable = upper & (target_difference != 0)
        correct += int(
            ((prediction_difference[comparable] * target_difference[comparable]) > 0).sum()
        )
        total += int(comparable.sum())
    return correct / total if total else 0.0


def evaluate(
    model: nn.Module,
    features: torch.Tensor,
    targets: torch.Tensor,
    slices: list[slice],
) -> dict[str, float]:
    model.eval()
    with torch.no_grad():
        predictions = model(features).cpu().numpy()
    actual = targets.cpu().numpy()
    errors = predictions - actual
    return {
        "mae": float(np.abs(errors).mean()),
        "rmse": float(np.sqrt(np.square(errors).mean())),
        "pearson": correlation(predictions, actual),
        "spearman": correlation(rankdata(predictions), rankdata(actual)),
        "same_program_pairwise_accuracy": pairwise_accuracy(
            predictions, actual, slices
        ),
    }


def write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(value, file, indent=2, sort_keys=True)
        file.write("\n")


def git_metadata(repo_root: Path) -> dict[str, Any]:
    def run_git(*args: str) -> str:
        result = subprocess.run(
            ["git", *args], cwd=repo_root, check=True, capture_output=True, text=True
        )
        return result.stdout.strip()

    status = run_git("status", "--short")
    return {
        "commit": run_git("rev-parse", "HEAD"),
        "branch": run_git("branch", "--show-current"),
        "status_short": status.splitlines() if status else [],
    }


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_training(config: Mapping[str, Any], output_dir: Path) -> dict[str, Any]:
    training = config["training"]
    seed_everything(training["seed"])
    device = torch.device(training["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Frozen training requires CUDA, but CUDA is unavailable")

    train_records = read_records(config["dataset"]["train_path"])
    dev_records = read_records(config["dataset"]["dev_path"])
    action_count = int(config["representation"]["action_count"])
    max_sequence_length = int(config["representation"]["max_sequence_length"])
    if any(
        action < 0 or action >= action_count
        for row in train_records + dev_records
        for action in row["action_indices"]
    ):
        raise ValueError("Dataset contains an action outside the frozen action space")
    target_name = config["dataset"]["target"]
    train_x, train_y, train_programs = group_records(
        train_records, action_count, max_sequence_length, target_name
    )
    dev_x, dev_y, dev_programs = group_records(dev_records, action_count, max_sequence_length, target_name)
    if set(train_programs) & set(dev_programs):
        raise ValueError("Train and dev programs overlap")
    train_x, dev_x, feature_mean, feature_std = normalize_features(train_x, dev_x)
    train_slices = program_slices(train_programs)
    dev_slices = program_slices(dev_programs)
    group_size = len(train_records) // len(train_slices)
    if any(group.stop - group.start != group_size for group in train_slices + dev_slices):
        raise ValueError("Every program must have the same trajectory count")

    model = MlpValueModel(train_x.shape[1], config["model"]["hidden_dimensions"]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training["learning_rate"],
        weight_decay=training["weight_decay"],
    )
    regression_loss = nn.HuberLoss()
    pairwise_weight = config["objective"]["pairwise_weight"]
    train_x_tensor = torch.from_numpy(train_x).to(device)
    train_y_tensor = torch.from_numpy(train_y).to(device)
    dev_x_tensor = torch.from_numpy(dev_x).to(device)
    dev_y_tensor = torch.from_numpy(dev_y).to(device)

    generator = torch.Generator().manual_seed(training["seed"])
    curve: list[dict[str, Any]] = []
    programs_per_batch = training["programs_per_batch"]
    for epoch in range(1, training["epochs"] + 1):
        model.train()
        order = torch.randperm(len(train_slices), generator=generator).tolist()
        epoch_losses: list[float] = []
        for offset in range(0, len(order), programs_per_batch):
            indices: list[int] = []
            for program_index in order[offset : offset + programs_per_batch]:
                group = train_slices[program_index]
                indices.extend(range(group.start, group.stop))
            index_tensor = torch.tensor(indices, device=device)
            features = train_x_tensor[index_tensor]
            targets = train_y_tensor[index_tensor]
            predictions = model(features)
            loss = regression_loss(predictions, targets) + pairwise_weight * batch_pairwise_loss(
                predictions, targets, group_size
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))

        dev_metrics = evaluate(model, dev_x_tensor, dev_y_tensor, dev_slices)
        epoch_result = {
            "epoch": epoch,
            "train_loss": float(np.mean(epoch_losses)),
            "dev": dev_metrics,
        }
        curve.append(epoch_result)
        print(json.dumps(epoch_result, sort_keys=True), flush=True)

    constant = float(train_y.mean())
    constant_errors = constant - dev_y
    constant_metrics = {
        "mae": float(np.abs(constant_errors).mean()),
        "rmse": float(np.sqrt(np.square(constant_errors).mean())),
    }
    checkpoint = {
        "model_type": "MLP",
        "model_state_dict": model.state_dict(),
        "input_dimension": train_x.shape[1],
        "hidden_dimensions": config["model"]["hidden_dimensions"],
        "feature_mean": torch.from_numpy(feature_mean),
        "feature_std": torch.from_numpy(feature_std),
        "action_count": action_count,
        "max_sequence_length": max_sequence_length,
        "target": target_name,
    }
    checkpoint_path = output_dir / "model.pt"
    torch.save(checkpoint, checkpoint_path)
    write_json(output_dir / "learning_curve.json", curve)
    final_metrics = curve[-1]["dev"]
    finite = all(math.isfinite(value) for value in final_metrics.values())
    checks = {
        "completed_epochs": len(curve) == config["pass_fail_gate"]["completed_epochs"],
        "all_dev_metrics_finite": finite,
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
            "input_dimension": train_x.shape[1],
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        },
        "constant_predictor_dev": constant_metrics,
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
