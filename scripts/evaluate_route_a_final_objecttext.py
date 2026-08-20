#!/usr/bin/env python3
"""Execute frozen v6 Step-11 final/OOD ObjectText evaluation without selection."""
from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import json
import os
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

if __package__:
    from scripts.generate_rlcompopt_objecttext_labels import K, environment_metadata, load_candidates, run_split_parallel
    from scripts.train_autophase_nvp_objecttext import AutophaseNVP
    from scripts.train_controlled_nvp_stage_a import ControlledCandidateModel, load_candidates as load_controlled_candidates
else:
    from generate_rlcompopt_objecttext_labels import K, environment_metadata, load_candidates, run_split_parallel
    from train_autophase_nvp_objecttext import AutophaseNVP
    from train_controlled_nvp_stage_a import ControlledCandidateModel, load_candidates as load_controlled_candidates


_ENV: Any = None


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dataset_id(program_id: str) -> str:
    return program_id.split("://", 1)[1].split("/", 1)[0]


def load_final_manifest(path: Path, cfg: Mapping[str, Any]) -> list[str]:
    payload = load_json(path)
    programs = [str(value) for value in payload["benchmarks"]]
    expected = cfg["program_manifest"]
    if file_sha256(path) != expected["sha256"]:
        raise ValueError("official final manifest SHA256 mismatch")
    if len(programs) != expected["expected_program_count"] or len(programs) != len(set(programs)):
        raise ValueError("official final manifest count or uniqueness mismatch")
    counts = collections.Counter(dataset_id(program) for program in programs)
    if dict(sorted(counts.items())) != dict(sorted(expected["expected_dataset_counts"].items())):
        raise ValueError("official final manifest dataset membership mismatch")
    return programs

def assert_split_integrity(programs: Sequence[str], cfg: Mapping[str, Any]) -> None:
    policy = cfg["split_integrity"]
    splits = []
    for source, expected_count in ((policy["train_source"], policy["train_expected_count"]), (policy["validation_source"], policy["validation_expected_count"])):
        payload = load_json(Path(source))
        values = [row["benchmark"] for row in payload["samples"]]
        if len(values) != expected_count or len(values) != len(set(values)):
            raise ValueError(f"invalid frozen split source: {source}")
        splits.append(set(values))
    if splits[0] & splits[1] or (set(programs) & (splits[0] | splits[1])):
        raise ValueError("official train/validation/final split leakage")


def validate_checkpoint_inventory(root: Path, cfg: Mapping[str, Any]) -> None:
    expected = {checkpoint_path(root, model, seed) for model in cfg["models"]["names"] for seed in cfg["models"]["seeds"]}
    actual = set(root.glob("*/seed*/model.pt"))
    if actual != expected:
        raise ValueError("checkpoint inventory must be exactly Stage-B models x seeds 1/2/3")
    for path in sorted(expected):
        payload = torch.load(path, map_location="cpu")
        if payload.get("stage") != "Route-A Stage B" or payload.get("stage_a_checkpoint_reused") is not False:
            raise ValueError(f"checkpoint is not a permitted Stage-B checkpoint: {path}")



def validate_config(cfg: Mapping[str, Any]) -> None:
    if cfg["candidate_space"]["K"] != K or cfg["inference"]["learned_scored_pass_budget"] != 45:
        raise ValueError("frozen candidate or policy budget mismatch")
    if cfg["target"]["temperature"] != 0.05 or cfg["inference"]["sampling"] is not False:
        raise ValueError("frozen target or deterministic inference mismatch")
    if cfg["models"]["names"] != ["NVP", "MLP", "LSTM", "Transformer", "Mamba"] or cfg["models"]["seeds"] != [1, 2, 3]:
        raise ValueError("exact Stage-B model/seed set is required")
    if cfg["models"]["stage_a_checkpoint_allowed"] or cfg["models"]["final_checkpoint_selection"]:
        raise ValueError("Stage-A or final-selected checkpoints are forbidden")
    if cfg["comparison_families"] != {"H1": ["Mamba", "Oz"], "H2a": ["Mamba", "NVP"], "H2b": ["MLP", "LSTM", "Transformer", "Mamba"]}:
        raise ValueError("primary comparison families are frozen")


def prepare_final_output(output_dir: Path, cfg: Mapping[str, Any], manifest: Path, *, resume: bool) -> None:
    frozen = dict(cfg)
    frozen["frozen_program_manifest"] = "final_program_manifest.json"
    frozen["environment_observed_at_freeze"] = environment_metadata()
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
        shutil.copyfile(manifest, output_dir / "final_program_manifest.json")
        (output_dir / "config.json").write_text(json.dumps(frozen, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return
    if not resume:
        raise FileExistsError(f"refusing to overwrite output: {output_dir}")
    existing = load_json(output_dir / "config.json")
    if existing != frozen or file_sha256(output_dir / "final_program_manifest.json") != cfg["program_manifest"]["sha256"]:
        raise ValueError("resume does not match the frozen Step-10 config or manifest")
    if (output_dir / "comparison_report.json").exists():
        raise FileExistsError("refusing to resume completed final evaluation")


def read_label_matrix(shards: Path, programs: Sequence[str]) -> tuple[dict[str, list[dict[str, Any]]], dict[str, dict[str, Any]]]:
    matrix, summaries = {}, {}
    for index, program in enumerate(programs):
        path = shards / f"{index:06d}.json.gz"
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            payload = json.load(handle)
        summary, records = payload["program_summary"], payload["records"]
        if summary["program_id"] != program or len(records) != K:
            raise ValueError(f"invalid frozen final shard: {path}")
        summaries[program] = summary
        if summary["oracle_K50_validity"] == "valid_complete_K50":
            records = sorted(records, key=lambda row: row["candidate_id"])
            if [row["candidate_id"] for row in records] != list(range(K)):
                raise ValueError("candidate K=50 ordering mismatch")
            matrix[program] = records
    return matrix, summaries


def _init_feature_worker() -> None:
    global _ENV
    import compiler_gym
    _ENV = compiler_gym.make("llvm-v0", reward_space=None)


def _feature_one(program: str) -> tuple[str, list[float] | None, str | None]:
    try:
        _ENV.reset(benchmark=program)
        raw = np.asarray(_ENV.observation["Autophase"], dtype=np.float32).reshape(-1)
        if raw.size != 56 or raw[51] <= 0:
            raise ValueError(f"invalid Autophase shape/denominator: {raw.shape}/{raw[51] if raw.size > 51 else None}")
        return program, (raw / raw[51]).tolist(), None
    except Exception as error:
        return program, None, f"{type(error).__name__}: {error}"


def extract_features(programs: Sequence[str], workers: int) -> tuple[dict[str, list[float]], dict[str, str]]:
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_feature_worker) as pool:
        rows = list(pool.map(_feature_one, programs, chunksize=16))
    return ({program: feature for program, feature, error in rows if error is None and feature is not None}, {program: error for program, feature, error in rows if error is not None})


def checkpoint_path(root: Path, model: str, seed: int) -> Path:
    return root / model.lower() / f"seed{seed}" / "model.pt"


def load_model(model_name: str, seed: int, root: Path, controlled_cfg: Mapping[str, Any], device: torch.device) -> torch.nn.Module:
    path = checkpoint_path(root, model_name, seed)
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location=device)
    if payload.get("stage") != "Route-A Stage B" or payload.get("architecture") != model_name or payload.get("seed") != seed or payload.get("stage_a_checkpoint_reused") is not False:
        raise ValueError(f"not an exact Stage-B checkpoint: {path}")
    if model_name == "NVP":
        model: torch.nn.Module = AutophaseNVP()
    else:
        tokens, lengths = load_controlled_candidates(Path(controlled_cfg["candidate_representation"]["candidate_sequences"]), pad_token_id=int(controlled_cfg["candidate_representation"]["pad_token_id"]), padded_length=int(controlled_cfg["candidate_representation"]["padded_length"]))
        model = ControlledCandidateModel(model_name, {**controlled_cfg["candidate_representation"], **controlled_cfg["models"][model_name]}, tokens, lengths)
    model.load_state_dict(payload["state_dict"], strict=True)
    return model.to(device).eval()


def policy45(score: Sequence[float], records: Sequence[Mapping[str, Any]]) -> int:
    budget, observed = 45, []
    for candidate_id in sorted(range(K), key=lambda index: (-score[index], index)):
        prefix = records[candidate_id]["prefix_object_text_size_bytes"]
        take = min(budget, len(prefix))
        observed.extend(prefix[:take])
        budget -= take
        if budget == 0:
            break
    if budget != 0 or not observed:
        raise ValueError("policy-45 did not consume exactly 45 scored passes")
    return min(observed)


def evaluate_model(model_name: str, seed: int, model: torch.nn.Module, programs: Sequence[str], matrix: Mapping[str, Sequence[Mapping[str, Any]]], summaries: Mapping[str, Mapping[str, Any]], features: Mapping[str, Sequence[float]], feature_errors: Mapping[str, str], output: Path, device: torch.device) -> dict[str, Any]:
    eligible = [program for program in programs if program in matrix and summaries[program]["ratio_metric_validity"] == "valid_for_ObjectText_ratio_metric" and program in features]
    logits: dict[str, list[float]] = {}
    with torch.no_grad():
        for start in range(0, len(eligible), 128):
            current = eligible[start:start + 128]
            values = model(torch.tensor([features[program] for program in current], dtype=torch.float32, device=device)).detach().cpu().tolist()
            logits.update(zip(current, values))
    output.parent.mkdir(parents=True, exist_ok=True)
    failure_counts: collections.Counter[str] = collections.Counter()
    dataset_counts: dict[str, collections.Counter[str]] = collections.defaultdict(collections.Counter)
    with gzip.open(output, "wt", encoding="utf-8") as handle:
        for program in programs:
            summary = summaries[program]
            dataset = summary["dataset_id"]
            row: dict[str, Any] = {"program_id": program, "dataset_id": dataset, "model": model_name, "seed": seed, "valid": False}
            if summary["oracle_K50_validity"] != "valid_complete_K50":
                reason = "incomplete_K50"
            elif summary["ratio_metric_validity"] != "valid_for_ObjectText_ratio_metric":
                reason = "invalid_ratio_denominator"
            elif program not in features:
                reason = feature_errors.get(program, "missing_feature")
            else:
                try:
                    policy = policy45(logits[program], matrix[program])
                    oracle = min(row_["best_object_text_size_bytes"] for row_ in matrix[program])
                    oz = int(summary["oz_object_text_size_bytes"])
                    row.update({"valid": True, "policy45_object_text_size_bytes": policy, "oracle_object_text_size_bytes": oracle, "oz_object_text_size_bytes": oz, "mean_over_oz": (oz - policy) / oz, "policy45_regret_bytes": policy - oracle})
                    dataset_counts[dataset]["N_primary_valid"] += 1
                    handle.write(json.dumps(row, separators=(",", ":")) + "\n")
                    continue
                except Exception as error:
                    reason = f"{type(error).__name__}: {error}"
            row["failure_reason"] = reason
            failure_counts[reason] += 1
            dataset_counts[dataset]["N_failed_or_invalid"] += 1
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    return {"model": model_name, "seed": seed, "result_file": str(output), "failure_count_by_reason": dict(failure_counts), "per_dataset_method_validity": {dataset: {"N_primary_valid": values["N_primary_valid"], "N_failed_or_invalid": values["N_failed_or_invalid"]} for dataset, values in sorted(dataset_counts.items())}}


def read_results(path: Path) -> dict[str, dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return {row["program_id"]: row for row in (json.loads(line) for line in handle)}


def aggregate_family(name: str, methods: Sequence[str], datasets: Sequence[str], programs: Sequence[str], results: Mapping[tuple[str, int], Mapping[str, Mapping[str, Any]]], summaries: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    seeds = [1, 2, 3]
    per_dataset: dict[str, Any] = {}
    for dataset in datasets:
        members = [program for program in programs if summaries[program]["dataset_id"] == dataset]
        common = [program for program in members if summaries[program]["ratio_metric_validity"] == "valid_for_ObjectText_ratio_metric" and all(results[(method, seed)][program]["valid"] for method in methods if method != "Oz" for seed in seeds)]
        values: dict[str, Any] = {"N_total": len(members), "N_primary_valid": len(common), "N_failed_or_invalid": len(members) - len(common)}
        for method in methods:
            if method == "Oz":
                values[method] = {"per_seed": {str(seed): 0.0 if common else None for seed in seeds}, "three_seed_mean": 0.0 if common else None}
            else:
                seed_values = {str(seed): (sum(results[(method, seed)][program]["mean_over_oz"] for program in common) / len(common) if common else None) for seed in seeds}
                values[method] = {"per_seed": seed_values, "three_seed_mean": (sum(seed_values.values()) / 3 if common else None)}
        per_dataset[dataset] = values
    macros: dict[str, Any] = {}
    for method in methods:
        if any(per_dataset[dataset][method]["three_seed_mean"] is None for dataset in datasets):
            macros[method] = {"per_seed": {str(seed): None for seed in seeds}, "three_seed_mean": None}
        else:
            seed_values = {str(seed): sum(per_dataset[dataset][method]["per_seed"][str(seed)] for dataset in datasets) / len(datasets) for seed in seeds}
            macros[method] = {"per_seed": seed_values, "three_seed_mean": sum(seed_values.values()) / 3}
    return {"family": name, "methods": list(methods), "per_dataset": per_dataset, "dataset_macro": macros}


def oracle_summary(datasets: Sequence[str], programs: Sequence[str], matrix: Mapping[str, Sequence[Mapping[str, Any]]], summaries: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    values = {}
    for dataset in datasets:
        rows = [program for program in programs if summaries[program]["dataset_id"] == dataset and program in matrix and summaries[program]["ratio_metric_validity"] == "valid_for_ObjectText_ratio_metric"]
        values[dataset] = None if not rows else sum((summaries[program]["oz_object_text_size_bytes"] - min(record["best_object_text_size_bytes"] for record in matrix[program])) / summaries[program]["oz_object_text_size_bytes"] for program in rows) / len(rows)
    return {"per_dataset": values, "dataset_macro": None if any(value is None for value in values.values()) else sum(values.values()) / len(values)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--official-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    cfg = load_json(args.config); validate_config(cfg)
    programs = load_final_manifest(args.official_manifest, cfg)
    if args.workers < 1: raise ValueError("workers must be positive")
    assert_split_integrity(programs, cfg)
    root = Path(cfg["models"]["checkpoint_root"])
    validate_checkpoint_inventory(root, cfg)
    prepare_final_output(args.output_dir, cfg, args.official_manifest, resume=args.resume)
    candidates = load_candidates(Path(cfg["candidate_space"]["source"]))
    labels = run_split_parallel(output_dir=args.output_dir, split_name="final", programs=programs, candidates=candidates, metadata=environment_metadata(), workers=args.workers)
    matrix, summaries = read_label_matrix(args.output_dir / "shards" / "final", programs)
    features, feature_errors = extract_features(programs, args.workers)
    controlled_cfg = load_json(Path("configs/controlled_nvp_stage_a_v6.json"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reports, result_maps = [], {}
    for model_name in cfg["models"]["names"]:
        for seed in cfg["models"]["seeds"]:
            result_path = args.output_dir / "model_results" / model_name.lower() / f"seed{seed}.jsonl.gz"
            if result_path.exists():
                report = {"model": model_name, "seed": seed, "result_file": str(result_path), "resumed": True}
            else:
                model = load_model(model_name, seed, root, controlled_cfg, device)
                report = evaluate_model(model_name, seed, model, programs, matrix, summaries, features, feature_errors, result_path, device)
            reports.append(report); result_maps[(model_name, seed)] = read_results(result_path)
    datasets = sorted({dataset_id(program) for program in programs})
    families = [aggregate_family(name, methods, datasets, programs, result_maps, summaries) for name, methods in cfg["comparison_families"].items()]
    report = {"step_execution": "COMPLETE", "program_manifest_count": len(programs), "datasets": datasets, "final_k50_labels": labels, "feature_failure_count_by_reason": dict(collections.Counter(feature_errors.values())), "model_seed_results": reports, "offline_k50_oracle": oracle_summary(datasets, programs, matrix, summaries), "comparison_families": families, "runtime_accessed": False, "final_checkpoint_selection": False}
    (args.output_dir / "comparison_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
