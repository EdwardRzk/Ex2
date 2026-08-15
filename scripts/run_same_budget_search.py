#!/usr/bin/env python3
"""Run frozen same-budget ObjectTextSize search on held-out programs."""

from __future__ import annotations

import argparse
import gzip
import json
import math
import platform
import random
import sys
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

if __package__:
    from scripts.train_mambapo_value import MambaValueModel
    from scripts.train_mlp_value_baseline import MlpValueModel, git_metadata, write_json
    from scripts.train_sequence_value_baseline import SequenceValueModel
else:
    from train_mambapo_value import MambaValueModel
    from train_mlp_value_baseline import MlpValueModel, git_metadata, write_json
    from train_sequence_value_baseline import SequenceValueModel


LEARNED_METHODS = ("mlp", "lstm", "transformer", "mamba")
SEARCH_METHODS = ("random",) + LEARNED_METHODS


@dataclass
class SearchEntry:
    actions: tuple[int, ...]
    states: tuple[np.ndarray, ...]
    predicted_score: float | None = None


class MlpScorer:
    def __init__(self, checkpoint_path: Path, device: torch.device):
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        self.action_count = int(checkpoint["action_count"])
        self.max_sequence_length = int(checkpoint["max_sequence_length"])
        self.mean = checkpoint["feature_mean"].numpy()
        self.std = checkpoint["feature_std"].numpy()
        self.device = device
        self.model = MlpValueModel(
            int(checkpoint["input_dimension"]), list(checkpoint["hidden_dimensions"])
        ).to(device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

    def score(self, entries: Sequence[SearchEntry]) -> np.ndarray:
        features: list[np.ndarray] = []
        for entry in entries:
            histogram = np.bincount(
                np.asarray(entry.actions, dtype=np.int64), minlength=self.action_count
            ).astype(np.float32)
            histogram /= max(len(entry.actions), 1)
            feature = np.concatenate(
                (
                    np.asarray(entry.states[-1], dtype=np.float32),
                    histogram,
                    np.asarray(
                        [len(entry.actions) / self.max_sequence_length], dtype=np.float32
                    ),
                )
            )
            features.append((feature - self.mean) / self.std)
        tensor = torch.from_numpy(np.stack(features)).to(self.device)
        with torch.no_grad():
            return self.model(tensor).float().cpu().numpy()


class SequenceScorer:
    def __init__(self, method: str, checkpoint_path: Path, device: torch.device):
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        self.representation = checkpoint["representation"]
        self.mean = checkpoint["state_mean"].numpy()
        self.std = checkpoint["state_std"].numpy()
        self.device = device
        if method == "mamba":
            self.model = MambaValueModel(
                self.representation, checkpoint["model_config"]
            ).to(device)
        else:
            self.model = SequenceValueModel(
                self.representation, checkpoint["model_config"]
            ).to(device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

    def score(self, entries: Sequence[SearchEntry]) -> np.ndarray:
        max_tokens = int(self.representation["max_sequence_length"]) + 1
        state_dimension = int(self.representation["state_dimension"])
        start_action = int(self.representation["start_action_index"])
        states = np.zeros((len(entries), max_tokens, state_dimension), dtype=np.float32)
        actions = np.full((len(entries), max_tokens), start_action, dtype=np.int64)
        lengths = np.zeros(len(entries), dtype=np.int64)
        for index, entry in enumerate(entries):
            length = len(entry.actions) + 1
            if length > max_tokens or len(entry.states) != length:
                raise ValueError("Search entry does not match the frozen token schema")
            states[index, :length] = (
                np.asarray(entry.states, dtype=np.float32) - self.mean
            ) / self.std
            actions[index, 1:length] = entry.actions
            lengths[index] = length
        with torch.no_grad():
            return (
                self.model(
                    torch.from_numpy(states).to(self.device),
                    torch.from_numpy(actions).to(self.device),
                    torch.from_numpy(lengths).to(self.device),
                )
                .float()
                .cpu()
                .numpy()
            )


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        config = json.load(file)
    required = {
        "experiment_name", "hypothesis", "dataset", "environment", "models",
        "search", "objective", "metrics", "pass_fail_gate",
    }
    missing = required - config.keys()
    if missing:
        raise ValueError(f"Config is missing fields: {sorted(missing)}")
    search = config["search"]
    if search["candidate_budget"] != 128:
        raise ValueError("Formal search budget must remain 128")
    if search["maximum_sequence_length"] != 32:
        raise ValueError("MambaPO v0 maximum sequence length must remain 32")
    if search["beam_width"] != 8 or search["top_k_actions"] != 8:
        raise ValueError("MambaPO v0 beam width and top-k must remain 8")
    if config["dataset"]["final_test_accessed"] is not False:
        raise ValueError("Held-out search must not access cBench")
    if set(config["models"]) != set(LEARNED_METHODS):
        raise ValueError("All four frozen model checkpoints are required")
    return config


def execute_sequence(
    environment: Any,
    program_uri: str,
    actions: Sequence[int],
    *,
    collect_states: bool,
    measure_size: bool,
    size_observation: str,
) -> tuple[tuple[np.ndarray, ...], int | None]:
    state = environment.reset(benchmark=program_uri)
    states: list[np.ndarray] = []
    if collect_states:
        states.append(np.asarray(state, dtype=np.float32))
    for action in actions:
        state, _, done, info = environment.step(int(action))
        if done:
            raise RuntimeError(
                f"LLVM episode ended for {program_uri} after action {action}: {info}"
            )
        if collect_states:
            states.append(np.asarray(state, dtype=np.float32))
    size = int(environment.observation[size_observation]) if measure_size else None
    return tuple(states), size


def propose_actions(
    beams: Sequence[SearchEntry],
    scorer: Any,
    *,
    action_count: int,
    top_k_actions: int,
    beam_width: int,
) -> list[tuple[int, ...]]:
    proposals: list[SearchEntry] = []
    owners: list[int] = []
    current_state_entries: list[SearchEntry] = []
    for beam_index, beam in enumerate(beams):
        for action in range(action_count):
            current_state_entries.append(
                SearchEntry(
                    actions=beam.actions + (action,),
                    states=beam.states + (beam.states[-1],),
                )
            )
            owners.append(beam_index)
    scores = scorer.score(current_state_entries)
    for beam_index in range(len(beams)):
        indices = [index for index, owner in enumerate(owners) if owner == beam_index]
        indices.sort(key=lambda index: float(scores[index]), reverse=True)
        for index in indices[:top_k_actions]:
            entry = current_state_entries[index]
            entry.predicted_score = float(scores[index])
            proposals.append(entry)
    proposals.sort(key=lambda entry: float(entry.predicted_score), reverse=True)
    return [entry.actions for entry in proposals[:beam_width]]


def generate_model_candidates(
    environment: Any,
    program_uri: str,
    scorer: Any,
    search: Mapping[str, Any],
    environment_config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    initial_states, _ = execute_sequence(
        environment,
        program_uri,
        (),
        collect_states=True,
        measure_size=False,
        size_observation=environment_config["size_observation"],
    )
    beams = [SearchEntry(actions=(), states=initial_states)]
    pool: list[SearchEntry] = []
    for _ in range(int(search["maximum_sequence_length"])):
        selected_actions = propose_actions(
            beams,
            scorer,
            action_count=int(environment_config["action_count"]),
            top_k_actions=int(search["top_k_actions"]),
            beam_width=int(search["beam_width"]),
        )
        exact_entries: list[SearchEntry] = []
        for actions in selected_actions:
            states, _ = execute_sequence(
                environment,
                program_uri,
                actions,
                collect_states=True,
                measure_size=False,
                size_observation=environment_config["size_observation"],
            )
            exact_entries.append(SearchEntry(actions=actions, states=states))
        exact_scores = scorer.score(exact_entries)
        for entry, score in zip(exact_entries, exact_scores):
            entry.predicted_score = float(score)
        exact_entries.sort(key=lambda entry: float(entry.predicted_score), reverse=True)
        beams = exact_entries[: int(search["beam_width"])]
        pool.extend(beams)

    unique: dict[tuple[int, ...], SearchEntry] = {}
    for entry in pool:
        previous = unique.get(entry.actions)
        if previous is None or float(entry.predicted_score) > float(previous.predicted_score):
            unique[entry.actions] = entry
    ranked = sorted(
        unique.values(), key=lambda entry: float(entry.predicted_score), reverse=True
    )
    budget = int(search["candidate_budget"])
    if len(ranked) < budget:
        raise RuntimeError(f"Beam generated only {len(ranked)} unique candidates")
    return [
        {
            "actions": list(entry.actions),
            "predicted_score": float(entry.predicted_score),
        }
        for entry in ranked[:budget]
    ]


def generate_random_candidates(
    *,
    action_count: int,
    budget: int,
    minimum_length: int,
    maximum_length: int,
    seed: int,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    candidates: list[dict[str, Any]] = []
    seen: set[tuple[int, ...]] = set()
    while len(candidates) < budget:
        length = rng.randint(minimum_length, maximum_length)
        actions = tuple(rng.randrange(action_count) for _ in range(length))
        if actions in seen:
            continue
        seen.add(actions)
        candidates.append({"actions": list(actions), "predicted_score": None})
    return candidates


def evaluate_candidates(
    environment: Any,
    program_uri: str,
    candidates: list[dict[str, Any]],
    *,
    oz_size_bytes: int,
    budget_points: Sequence[int],
    size_observation: str,
) -> dict[str, Any]:
    evaluated: list[dict[str, Any]] = []
    best_size = math.inf
    best_candidate: dict[str, Any] | None = None
    curve: dict[str, dict[str, Any]] = {}
    checkpoints = set(int(value) for value in budget_points)
    for index, candidate in enumerate(candidates, start=1):
        _, size = execute_sequence(
            environment,
            program_uri,
            candidate["actions"],
            collect_states=False,
            measure_size=True,
            size_observation=size_observation,
        )
        assert size is not None
        result = {
            "evaluation_index": index,
            "actions": candidate["actions"],
            "sequence_length": len(candidate["actions"]),
            "predicted_score": candidate["predicted_score"],
            "object_text_size_bytes": size,
            "size_reduction_vs_oz": (oz_size_bytes - size) / oz_size_bytes,
        }
        evaluated.append(result)
        if size < best_size:
            best_size = size
            best_candidate = result
        if index in checkpoints:
            curve[str(index)] = {
                "best_object_text_size_bytes": int(best_size),
                "size_reduction_vs_oz": (oz_size_bytes - best_size) / oz_size_bytes,
            }
    assert best_candidate is not None
    return {
        "candidate_evaluations": evaluated,
        "budget_curve": curve,
        "best_candidate": best_candidate,
    }


def aggregate_program_results(
    program_results: Sequence[Mapping[str, Any]], budget_points: Sequence[int]
) -> dict[str, Any]:
    aggregate: dict[str, Any] = {}
    for method in SEARCH_METHODS:
        curves: dict[str, Any] = {}
        for budget in budget_points:
            reductions = np.asarray(
                [
                    result["methods"][method]["budget_curve"][str(budget)][
                        "size_reduction_vs_oz"
                    ]
                    for result in program_results
                ],
                dtype=np.float64,
            )
            ratios = 1 - reductions
            curves[str(budget)] = {
                "mean_size_reduction_vs_oz": float(reductions.mean()),
                "median_size_reduction_vs_oz": float(np.median(reductions)),
                "geomean_size_ratio_vs_oz": float(np.exp(np.log(ratios).mean())),
                "geomean_size_reduction_vs_oz": float(
                    1 - np.exp(np.log(ratios).mean())
                ),
                "positive_program_count": int((reductions > 0).sum()),
                "best_program_reduction": float(reductions.max()),
                "worst_program_reduction": float(reductions.min()),
            }
        aggregate[method] = {"budget_curve": curves}
    return aggregate


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    import compiler_gym

    repo_root = Path(__file__).resolve().parents[1]
    config = load_config(args.config.resolve())
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    write_json(output_dir / "config.json", config)
    report: dict[str, Any] = {
        "experiment_name": config["experiment_name"],
        "mode": "smoke" if args.smoke else "formal",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "git": git_metadata(repo_root),
        "host": {
            "platform": platform.platform(),
            "python": sys.version,
            "torch": torch.__version__,
            "compiler_gym": compiler_gym.__version__,
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "decision": "FAIL",
    }
    results_file = None
    try:
        with Path(config["dataset"]["program_splits_path"]).open(encoding="utf-8") as file:
            splits = json.load(file)
        programs = list(splits[config["dataset"]["split"]])
        search = dict(config["search"])
        expected_programs = int(config["dataset"]["expected_programs"])
        if args.smoke:
            programs = programs[:1]
            search.update(
                {
                    "candidate_budget": 2,
                    "budget_points": [1, 2],
                    "maximum_sequence_length": 2,
                    "beam_width": 2,
                    "top_k_actions": 2,
                }
            )
            expected_programs = 1
        if len(programs) != expected_programs:
            raise ValueError("Unexpected held-out program count")

        device = torch.device("cuda")
        scorers: dict[str, Any] = {
            "mlp": MlpScorer(Path(config["models"]["mlp"]), device),
            "lstm": SequenceScorer("lstm", Path(config["models"]["lstm"]), device),
            "transformer": SequenceScorer(
                "transformer", Path(config["models"]["transformer"]), device
            ),
            "mamba": SequenceScorer("mamba", Path(config["models"]["mamba"]), device),
        }
        total_evaluations = {method: 0 for method in SEARCH_METHODS}
        program_summaries: list[dict[str, Any]] = []
        results_path = output_dir / "program_results.jsonl.gz"
        results_file = gzip.open(results_path, "wt", encoding="utf-8")
        with compiler_gym.make(
            config["environment"]["id"],
            observation_space=config["environment"]["feature_space"],
        ) as environment:
            if environment.action_space.n != config["environment"]["action_count"]:
                raise ValueError("LLVM action count changed from the frozen dataset")
            for program_index, program_uri in enumerate(programs):
                environment.reset(benchmark=program_uri)
                oz_size_bytes = int(
                    environment.observation[
                        config["environment"]["oz_baseline_observation"]
                    ]
                )
                candidates: dict[str, list[dict[str, Any]]] = {
                    "random": generate_random_candidates(
                        action_count=config["environment"]["action_count"],
                        budget=search["candidate_budget"],
                        minimum_length=search["minimum_sequence_length"],
                        maximum_length=search["maximum_sequence_length"],
                        seed=search["seed"] + program_index * 1_000_003,
                    )
                }
                for method in LEARNED_METHODS:
                    candidates[method] = generate_model_candidates(
                        environment,
                        program_uri,
                        scorers[method],
                        search,
                        config["environment"],
                    )
                methods: dict[str, Any] = {}
                for method in SEARCH_METHODS:
                    methods[method] = evaluate_candidates(
                        environment,
                        program_uri,
                        candidates[method],
                        oz_size_bytes=oz_size_bytes,
                        budget_points=search["budget_points"],
                        size_observation=config["environment"]["size_observation"],
                    )
                    total_evaluations[method] += len(
                        methods[method]["candidate_evaluations"]
                    )
                program_result = {
                    "program_id": program_uri,
                    "oz_object_text_size_bytes": oz_size_bytes,
                    "methods": methods,
                }
                results_file.write(json.dumps(program_result, separators=(",", ":")))
                results_file.write("\n")
                program_summaries.append(program_result)
                print(
                    f"completed_programs={program_index + 1}/{len(programs)}",
                    flush=True,
                )
        results_file.close()
        results_file = None
        aggregate = aggregate_program_results(
            program_summaries, search["budget_points"]
        )
        expected_per_method = int(search["candidate_budget"]) * len(programs)
        checks = {
            "completed_programs": len(program_summaries) == expected_programs,
            "true_candidate_evaluations_per_method": all(
                value == expected_per_method for value in total_evaluations.values()
            ),
            "invalid_episode_count": True,
            "train_dev_program_overlap": not (
                set(splits["train"]) & set(splits["dev"])
            ),
            "final_test_accessed": config["dataset"]["final_test_accessed"] is False,
        }
        report.update(
            {
                "completed_programs": len(program_summaries),
                "true_candidate_evaluations": total_evaluations,
                "budget_points": search["budget_points"],
                "aggregate": aggregate,
                "checks": checks,
                "decision": "PASS" if all(checks.values()) else "FAIL",
            }
        )
    except Exception as error:
        report["error"] = {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        }
    finally:
        if results_file is not None:
            results_file.close()
        report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(output_dir / "experiment_report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["decision"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
