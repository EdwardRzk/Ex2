#!/usr/bin/env python3
"""Run the frozen MambaNVP-versus-NVP post-hoc runtime comparison.

This program deliberately never initializes CompilerGym and never applies a
candidate sequence.  The selected policy-45 prefixes are reconstructed from
the frozen K=50 labels, then matched byte-for-byte to an already-built frozen
Route-A runtime binary with the same prefix.  The matched binary is copied to
the new isolated result directory and timed under the inherited protocol.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import shutil
import statistics
import subprocess
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

if __package__:
    from scripts.evaluate_mamba_nvp_final_objecttext import (
        load_final_features,
        load_mamba_nvp,
        read_final_artifacts,
    )
    from scripts.train_mamba_nvp_objecttext import load_frozen_nvp, load_json
else:
    from evaluate_mamba_nvp_final_objecttext import load_final_features, load_mamba_nvp, read_final_artifacts
    from train_mamba_nvp_objecttext import load_frozen_nvp, load_json


K = 50
SEEDS = (1, 2, 3)
METHODS = ("oz", *(f"nvp_seed{seed}" for seed in SEEDS), *(f"mambanvp_seed{seed}" for seed in SEEDS))
THREAD_ENV = {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assert_hash(path: Path, expected: str) -> Path:
    if not path.is_file() or sha256(path) != expected:
        raise ValueError(f"artifact hash mismatch: {path}")
    return path


def method_id(model: str, seed: int) -> str:
    return f"{model.lower()}_seed{seed}"


def selected_prefix(scores: Sequence[float], records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Implement the inherited deterministic frozen policy-45 exactly."""
    if len(scores) != K or len(records) != K:
        raise ValueError("policy45 requires complete K=50 logits and records")
    budget, best = 45, None
    ranking = sorted(range(K), key=lambda candidate_id: (-scores[candidate_id], candidate_id))
    for rank, candidate_id in enumerate(ranking):
        record = records[candidate_id]
        actions = record["ordered_pass_sequence"]
        values = record["prefix_object_text_size_bytes"]
        if record["candidate_id"] != candidate_id or len(actions) != len(values):
            raise ValueError("frozen candidate records are not ordered")
        take = min(budget, len(values))
        for prefix_index, value in enumerate(values[:take]):
            if best is None or value < best["policy45_object_text_size_bytes"]:
                best = {
                    "candidate_id": candidate_id,
                    "candidate_rank": rank,
                    "prefix_index": prefix_index,
                    "pass_count": prefix_index + 1,
                    "action_ids": list(actions[: prefix_index + 1]),
                    "policy45_object_text_size_bytes": int(value),
                }
        budget -= take
        if budget == 0:
            break
    if budget != 0 or best is None:
        raise ValueError("policy45 did not consume exactly 45 frozen observations")
    return best


def gmean(values: Sequence[float]) -> float | None:
    if not values or any(value <= 0 for value in values):
        return None
    return math.exp(sum(math.log(value) for value in values) / len(values))


def check_source_inputs(root: Path) -> dict[str, Any]:
    protocol = root / "configs/route_a_posthoc_runtime_v6.json"
    legacy_prefixes = root / "outputs/route_a_posthoc_runtime_v6/policy_prefixes.json"
    legacy_builds = root / "outputs/route_a_posthoc_runtime_v6/build_manifest.json"
    amplification = root / "outputs/route_a_posthoc_runtime_v6/amplification_manifest.json"
    final_cfg = root / "configs/mamba_nvp_final_objecttext_v6.json"
    training_cfg = root / "configs/mamba_nvp_objecttext_v6.json"
    controlled_cfg = root / "configs/controlled_nvp_stage_a_v6.json"
    required = (protocol, legacy_prefixes, legacy_builds, amplification, final_cfg, training_cfg, controlled_cfg)
    if not all(path.is_file() for path in required):
        raise FileNotFoundError("a required frozen source artifact is missing")
    return {
        "route_a_posthoc_runtime_config": {"path": str(protocol), "sha256": sha256(protocol)},
        "legacy_policy_prefixes": {"path": str(legacy_prefixes), "sha256": sha256(legacy_prefixes)},
        "legacy_build_manifest": {"path": str(legacy_builds), "sha256": sha256(legacy_builds)},
        "frozen_oz_amplification": {"path": str(amplification), "sha256": sha256(amplification)},
        "mambanvp_final_config": {"path": str(final_cfg), "sha256": sha256(final_cfg)},
        "mambanvp_training_config": {"path": str(training_cfg), "sha256": sha256(training_cfg)},
        "controlled_config": {"path": str(controlled_cfg), "sha256": sha256(controlled_cfg)},
        "final_autophase_cache": {"path": str(root / "outputs/autophase_feature_cache_v6/final_autophase.jsonl.gz"), "sha256": sha256(root / "outputs/autophase_feature_cache_v6/final_autophase.jsonl.gz")},
        "final_label_shards": str(root / "outputs/route_a_final_objecttext_v6/shards/final"),
        "checkpoints": {
            "NVP": {
                str(seed): {"path": str(root / f"outputs/route_a_stage_b_v6/nvp/seed{seed}/model.pt"), "sha256": sha256(root / f"outputs/route_a_stage_b_v6/nvp/seed{seed}/model.pt")}
                for seed in SEEDS
            },
            "MambaNVP": {
                str(seed): {"path": str(root / f"outputs/mamba_nvp_objecttext_v6/seed{seed}/model.pt"), "sha256": sha256(root / f"outputs/mamba_nvp_objecttext_v6/seed{seed}/model.pt")}
                for seed in SEEDS
            },
        },
    }


def freeze(root: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output directory: {output}")
    inputs = check_source_inputs(root)
    base = load_json(root / "configs/route_a_posthoc_runtime_v6.json")
    if base["protocol_class"] != "post_hoc_exploratory_runtime" or len(base["programs"]) != 9:
        raise ValueError("frozen Route-A runtime protocol mismatch")
    programs = []
    for program in base["programs"]:
        copied = dict(program)
        copied["workdir"] = str(output / "work" / program["program_id"].rsplit("/", 1)[-1])
        programs.append(copied)
    cfg = {
        "experiment_name": "mambanvp_nvp_runtime_v1",
        "protocol_class": "post_hoc_exploratory_runtime",
        "purpose": "Frozen MambaNVP versus NVP runtime comparison only",
        "parent_protocol": inputs["route_a_posthoc_runtime_config"],
        "frozen_inputs": inputs,
        "population": {"included_program_count": 9, "program_ids": [row["program_id"] for row in programs]},
        "methods": {"baseline": "Oz", "learned": [{"model": model, "seed": seed} for model in ("NVP", "MambaNVP") for seed in SEEDS]},
        "inference": {"sampling": False, "ranking": "descending frozen logits; candidate_id ascending ties", "policy": base["methods"]["policy"]},
        "binary_provenance": "Copy byte-identical legacy Route-A runtime binaries only after exact selected-prefix action-id match; no CompilerGym rollout and no LLVM phase application.",
        "measurement": base["measurement"],
        "correctness": base["correctness"],
        "aggregation": {**base["aggregation"], "primary_cohort": "semantic validation passes for Oz, all NVP seeds, and all MambaNVP seeds", "secondary_cohort": "all seven methods execute successfully", "direct_comparison": "GMeanSpeedup_MambaNVP / GMeanSpeedup_NVP; greater than 1 favors MambaNVP"},
        "forbidden": ["CompilerGym candidate rollout", "LLVM phase search", "ObjectText measurement", "label regeneration", "retraining", "checkpoint selection", "sampling", "final/OOD re-evaluation"],
        "programs": programs,
    }
    output.mkdir(parents=True)
    (output / "config.json").write_text(json.dumps(cfg, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_output_config(output: Path) -> dict[str, Any]:
    cfg_path = output / "config.json"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"missing frozen output config: {cfg_path}")
    cfg = load_json(cfg_path)
    if cfg["experiment_name"] != "mambanvp_nvp_runtime_v1" or cfg["inference"]["sampling"] is not False:
        raise ValueError("not the frozen MambaNVP/NVP runtime protocol")
    return cfg


def recover_prefixes(root: Path, output: Path) -> None:
    cfg = load_output_config(output)
    destination = output / "policy_prefixes.json"
    if destination.exists():
        raise FileExistsError(destination)
    source = cfg["frozen_inputs"]
    for entry in (source["mambanvp_final_config"], source["mambanvp_training_config"], source["controlled_config"], source["final_autophase_cache"]):
        assert_hash(Path(entry["path"]), entry["sha256"])
    final_cfg = load_json(Path(source["mambanvp_final_config"]["path"]))
    training_cfg = load_json(Path(source["mambanvp_training_config"]["path"]))
    controlled = load_json(Path(source["controlled_config"]["path"]))
    programs_all, matrix, summaries = read_final_artifacts(Path(source["final_label_shards"]))
    programs = cfg["population"]["program_ids"]
    if not set(programs) <= set(programs_all) or any(program not in matrix for program in programs):
        raise ValueError("runtime program does not have a frozen complete K=50 matrix")
    features = load_final_features(Path(source["final_autophase_cache"]["path"]), list(matrix))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result: dict[str, Any] = {
        "config_sha256": sha256(output / "config.json"),
        "compiler_gym_initialized": False,
        "candidate_rollouts": 0,
        "objecttext_measurements": 0,
        "sampling": False,
        "inference_device": str(device),
        "programs": {program: {} for program in programs},
    }
    for model_name in ("NVP", "MambaNVP"):
        for seed in SEEDS:
            checkpoint = source["checkpoints"][model_name][str(seed)]
            assert_hash(Path(checkpoint["path"]), checkpoint["sha256"])
            if model_name == "NVP":
                model = load_frozen_nvp(Path(checkpoint["path"]), seed).to(device).eval()
            else:
                model = load_mamba_nvp(seed, final_cfg, training_cfg, controlled, device)
            if model.training:
                raise RuntimeError("frozen model must be eval()")
            with torch.no_grad():
                scores = model(torch.tensor([features[program] for program in programs], dtype=torch.float32, device=device)).cpu().tolist()
            for program, values in zip(programs, scores):
                result["programs"][program][method_id(model_name, seed)] = selected_prefix(values, matrix[program])
    destination.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def choose_legacy_source(selected: Mapping[str, Any], legacy: Mapping[str, Any], preferred: str | None) -> tuple[str, Mapping[str, Any]]:
    keys = ([preferred] if preferred else []) + sorted(key for key in legacy if key != preferred)
    for key in keys:
        if key is not None and legacy[key]["action_ids"] == selected["action_ids"]:
            return key, legacy[key]
    raise ValueError("selected frozen prefix has no exact legacy binary provenance")


def copy_checked(source: Path, destination: Path, expected_hash: str) -> str:
    if not source.is_file() or sha256(source) != expected_hash:
        raise ValueError(f"legacy binary missing or corrupt: {source}")
    if destination.exists():
        raise FileExistsError(destination)
    shutil.copy2(source, destination)
    digest = sha256(destination)
    if digest != expected_hash:
        raise RuntimeError(f"copied binary hash mismatch: {destination}")
    return digest


def build(root: Path, output: Path) -> None:
    cfg = load_output_config(output)
    manifest_path, metadata_path = output / "build_manifest.json", output / "binary_metadata.json"
    if manifest_path.exists() or metadata_path.exists():
        raise FileExistsError("refusing to overwrite binary metadata")
    source = cfg["frozen_inputs"]
    assert_hash(Path(source["legacy_policy_prefixes"]["path"]), source["legacy_policy_prefixes"]["sha256"])
    assert_hash(Path(source["legacy_build_manifest"]["path"]), source["legacy_build_manifest"]["sha256"])
    prefixes = load_json(output / "policy_prefixes.json")
    legacy_prefixes = load_json(Path(source["legacy_policy_prefixes"]["path"]))
    legacy_builds = load_json(Path(source["legacy_build_manifest"]["path"]))
    report: dict[str, Any] = {"config_sha256": sha256(output / "config.json"), "execution": "COPY_VERIFIED_FROZEN_BINARY_NO_COMPILERGYM_ROLLOUT_NO_LLVM_PHASE_APPLICATION", "programs": {}}
    for spec in cfg["programs"]:
        program = spec["program_id"]
        legacy_methods = legacy_builds["programs"][program]["methods"]
        legacy_selected = legacy_prefixes["programs"][program]
        binary_dir = output / "work" / program.rsplit("/", 1)[-1] / "binaries"
        binary_dir.mkdir(parents=True)
        entries: dict[str, Any] = {}
        oz_source = legacy_methods["oz"]
        oz_dest = binary_dir / "oz"
        entries["oz"] = {"binary": str(oz_dest), "sha256": copy_checked(Path(oz_source["binary"]), oz_dest, oz_source["sha256"]), "legacy_source_method": "oz", "candidate_id": None, "pass_sequence_id": None, "action_ids": None, "pass_count": None, "prefix_index": None}
        for model_name in ("NVP", "MambaNVP"):
            for seed in SEEDS:
                identifier = method_id(model_name, seed)
                selected = prefixes["programs"][program][identifier]
                preferred = f"NVP_seed{seed}" if model_name == "NVP" else None
                legacy_key, legacy_selected_prefix = choose_legacy_source(selected, legacy_selected, preferred)
                legacy_method = legacy_key.lower()
                legacy_info = legacy_methods[legacy_method]
                if legacy_info["action_ids"] != selected["action_ids"]:
                    raise ValueError(f"legacy binary action mismatch: {program}/{identifier}")
                bitcode_dest, binary_dest = binary_dir / f"{identifier}.bc", binary_dir / identifier
                entries[identifier] = {
                    "binary": str(binary_dest), "sha256": copy_checked(Path(legacy_info["binary"]), binary_dest, legacy_info["sha256"]),
                    "bitcode": str(bitcode_dest), "bitcode_sha256": copy_checked(Path(legacy_info["bitcode"]), bitcode_dest, legacy_info["bitcode_sha256"]),
                    "legacy_source_method": legacy_method,
                    "legacy_prefix_key": legacy_key,
                    "candidate_id": selected["candidate_id"], "candidate_rank": selected["candidate_rank"], "pass_sequence_id": selected["candidate_id"],
                    "action_ids": selected["action_ids"], "pass_count": selected["pass_count"], "prefix_index": selected["prefix_index"],
                    "policy45_object_text_size_bytes": selected["policy45_object_text_size_bytes"],
                }
        report["programs"][program] = {"methods": entries, "all_expected_methods_built": set(entries) == set(METHODS)}
    manifest_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    metadata_path.write_text(json.dumps({"config_sha256": sha256(output / "config.json"), "binary_count": 63, "binary_reuse": "exact_hash_checked_copy", "programs": report["programs"]}, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def invoke(binary: Path, argv: Sequence[str], factor: int, cwd: Path, env: Mapping[str, str], timeout: int) -> float:
    command = ["taskset", "-c", "0", str(binary.resolve()), *argv[1:]]
    start = time.perf_counter()
    for _ in range(factor):
        done = subprocess.run(command, cwd=cwd, env=dict(env), stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout, check=False)
        if done.returncode:
            raise RuntimeError(f"execution failed ({done.returncode}): {' '.join(command)}")
    return time.perf_counter() - start


def timing_stats(values: Sequence[float]) -> dict[str, Any]:
    mean = statistics.mean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    rse = (std / math.sqrt(len(values))) / mean if mean else float("inf")
    return {"n": len(values), "mean_seconds": mean, "median_seconds": statistics.median(values), "sample_std_seconds": std, "rse": rse, "rse_target_reached": rse <= 0.01}


def run_correctness(cfg: Mapping[str, Any], builds: Mapping[str, Any], output: Path) -> None:
    result_path, cohort_path = output / "correctness_results.jsonl", output / "runtime_cohort_manifest.json"
    if result_path.exists() or cohort_path.exists():
        raise FileExistsError("refusing to overwrite correctness results")
    rows, per_program = [], {}
    for spec in cfg["programs"]:
        program = spec["program_id"]
        program_root = output / "work" / program.rsplit("/", 1)[-1]
        with tempfile.TemporaryDirectory(dir=program_root) as ref_dir_text:
            ref_dir = Path(ref_dir_text)
            reference = ref_dir / "reference"
            clang = load_json(Path(cfg["parent_protocol"]["path"]))["environment"]["clang"]
            done = subprocess.run([clang, spec["source_bitcode"], "-O2", "-o", str(reference), *spec["linkopts"]], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
            if done.returncode:
                raise RuntimeError(f"reference compile failed: {program}")
            ref_status, _, ref_stdout = correctness_run(reference, spec, ref_dir)
            if ref_status != "executed":
                raise RuntimeError(f"reference execution failed: {program}/{ref_status}")
            output_name = spec["correctness"].split("_and_", 1)[1] if "_and_" in spec["correctness"] else None
            ref_output = (ref_dir / output_name).read_bytes() if output_name else None
            program_rows = []
            for method in METHODS:
                info = builds["programs"][program]["methods"][method]
                binary = Path(info["binary"])
                if sha256(binary) != info["sha256"]:
                    raise ValueError(f"binary hash mismatch before correctness: {binary}")
                with tempfile.TemporaryDirectory(dir=program_root) as work_text:
                    work = Path(work_text)
                    status, code, stdout = correctness_run(binary, spec, work)
                    available = spec["correctness"] != "execution_only"
                    if status == "executed" and available:
                        output_ok = output_name is None or ((work / output_name).is_file() and (work / output_name).read_bytes() == ref_output)
                        correctness = "semantic_validated_pass" if stdout == ref_stdout and output_ok else "semantic_validated_fail"
                    elif status == "executed":
                        correctness = "execution_only_unverified"
                    else:
                        correctness = status
                    row = {"program_id": program, "method": method, "seed": None if method == "oz" else int(method.rsplit("seed", 1)[1]), "binary_path": str(binary), "binary_sha256": info["sha256"], "candidate_id": info["candidate_id"], "pass_sequence_id": info["pass_sequence_id"], "correctness_status": correctness, "execution_status": status, "exit_code": code, "correctness_check_available": available}
                    rows.append(row); program_rows.append(row)
            per_program[program] = program_rows
    result_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    counts = Counter(row["correctness_status"] for row in rows)
    cohorts = {program: {"primary_runtime_common_valid": {row["correctness_status"] for row in rows_for_program} == {"semantic_validated_pass"}, "execution_runtime_common_valid": {row["correctness_status"] for row in rows_for_program} <= {"semantic_validated_pass", "execution_only_unverified"}, "exclusion_reason": None if {row["correctness_status"] for row in rows_for_program} == {"semantic_validated_pass"} else "; ".join(sorted({row["correctness_status"] for row in rows_for_program}))} for program, rows_for_program in per_program.items()}
    cohort_path.write_text(json.dumps({"config_sha256": sha256(output / "config.json"), "N_runtime_programs": 9, "N_binaries": 63, **{f"N_{key}": counts[key] for key in ("semantic_validated_pass", "semantic_validated_fail", "execution_only_unverified", "execution_failed", "timeout")}, "N_primary_common_programs": sum(row["primary_runtime_common_valid"] for row in cohorts.values()), "N_secondary_execution_common_programs": sum(row["execution_runtime_common_valid"] for row in cohorts.values()), "programs": cohorts}, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def correctness_run(binary: Path, spec: Mapping[str, Any], cwd: Path) -> tuple[str, int | None, str]:
    (cwd / "_finfo_dataset").write_text("1\n", encoding="utf-8")
    command = [str(binary.resolve()) if value == "./a.out" else value for value in spec["argv"]]
    env = {"PATH": os.environ["PATH"], **THREAD_ENV, **spec["env"]}
    try:
        done = subprocess.run(command, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=spec["timeout_seconds"], check=False)
    except subprocess.TimeoutExpired:
        return "timeout", None, ""
    return ("executed" if done.returncode == 0 else "execution_failed"), done.returncode, done.stdout.decode("utf-8", "replace")


def time_binaries(cfg: Mapping[str, Any], builds: Mapping[str, Any], output: Path) -> None:
    raw_path, summary_path = output / "raw_timing_samples.jsonl", output / "timing_summary.json"
    if raw_path.exists() or summary_path.exists():
        raise FileExistsError("refusing to overwrite timing output")
    amplification = load_json(Path(cfg["frozen_inputs"]["frozen_oz_amplification"]["path"]))
    factors = {row["program_id"]: row["amplification_factor"] for row in amplification["programs"]}
    rows, report = [], {"timing_samples_only": True, "warmup_runs_excluded": 3, "programs": {}}
    for spec in cfg["programs"]:
        program = spec["program_id"]
        work = Path(spec["workdir"])
        (work / "_finfo_dataset").write_text("1\n", encoding="utf-8")
        env = {"PATH": os.environ["PATH"], **THREAD_ENV, **spec["env"]}
        factor = factors[program]
        report["programs"][program] = {}
        for method in METHODS:
            info = builds["programs"][program]["methods"][method]
            binary = Path(info["binary"])
            warmups = []
            for sample_index in range(1, 4):
                seconds = invoke(binary, spec["argv"], factor, work, env, spec["timeout_seconds"])
                warmups.append(seconds)
                rows.append(timing_row(program, method, info, "warmup", sample_index, seconds, factor))
            formal = []
            while len(formal) < 5 or (timing_stats(formal)["rse"] > 0.01 and len(formal) < 20):
                seconds = invoke(binary, spec["argv"], factor, work, env, spec["timeout_seconds"])
                formal.append(seconds)
                rows.append(timing_row(program, method, info, "formal", len(formal), seconds, factor))
            report["programs"][program][method] = {**timing_stats(formal), "warmup_seconds": warmups, "binary_sha256": info["sha256"], "candidate_id": info["candidate_id"], "pass_sequence_id": info["pass_sequence_id"]}
    raw_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    summary_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def timing_row(program: str, method: str, info: Mapping[str, Any], sample_type: str, sample_index: int, seconds: float, factor: int) -> dict[str, Any]:
    return {"program_id": program, "method": method, "seed": None if method == "oz" else int(method.rsplit("seed", 1)[1]), "sample_type": sample_type, "sample_index": sample_index, "seconds": seconds, "amplification_factor": factor, "binary_sha256": info["sha256"], "candidate_id": info["candidate_id"], "pass_sequence_id": info["pass_sequence_id"], "action_ids": info["action_ids"]}


def aggregate(output: Path) -> None:
    cfg = load_output_config(output)
    report_path = output / "runtime_report.json"
    if report_path.exists():
        raise FileExistsError(report_path)
    correctness = {row["program_id"]: {} for row in (json.loads(line) for line in (output / "correctness_results.jsonl").read_text(encoding="utf-8").splitlines())}
    for line in (output / "correctness_results.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line); correctness[row["program_id"]][row["method"]] = row
    summary = load_json(output / "timing_summary.json")
    primary = [program for program, rows in correctness.items() if all(rows[method]["correctness_status"] == "semantic_validated_pass" for method in METHODS)]
    secondary = [program for program, rows in correctness.items() if all(rows[method]["correctness_status"] in {"semantic_validated_pass", "execution_only_unverified"} for method in METHODS)]
    def cohort_result(programs: Sequence[str]) -> dict[str, Any]:
        by_seed = {}
        for seed in SEEDS:
            nvp = [summary["programs"][program]["oz"]["median_seconds"] / summary["programs"][program][f"nvp_seed{seed}"]["median_seconds"] for program in programs]
            mambanvp = [summary["programs"][program]["oz"]["median_seconds"] / summary["programs"][program][f"mambanvp_seed{seed}"]["median_seconds"] for program in programs]
            by_seed[str(seed)] = {"NVP_geomean_speedup_vs_Oz": gmean(nvp), "MambaNVP_geomean_speedup_vs_Oz": gmean(mambanvp), "speedup_ratio_MambaNVP_over_NVP": gmean(mambanvp) / gmean(nvp) if gmean(nvp) and gmean(mambanvp) else None, "per_program": {program: {"NVP_speedup_vs_Oz": nvp[index], "MambaNVP_speedup_vs_Oz": mambanvp[index]} for index, program in enumerate(programs)}}
        nvp_three = gmean([by_seed[str(seed)]["NVP_geomean_speedup_vs_Oz"] for seed in SEEDS])
        mambanvp_three = gmean([by_seed[str(seed)]["MambaNVP_geomean_speedup_vs_Oz"] for seed in SEEDS])
        return {"N_programs": len(programs), "program_ids": list(programs), "per_seed": by_seed, "three_seed": {"NVP_geomean_speedup_vs_Oz": nvp_three, "MambaNVP_geomean_speedup_vs_Oz": mambanvp_three, "speedup_ratio_MambaNVP_over_NVP": mambanvp_three / nvp_three if nvp_three and mambanvp_three else None}}
    primary_result, secondary_result = cohort_result(primary), cohort_result(secondary)
    report = {"step_execution": "COMPLETE", "protocol": "route_a_posthoc_runtime_v6 reused unchanged for benchmarks/toolchain/affinity/warmup/repetitions/correctness", "prohibitions_observed": {"compiler_gym_initialized": False, "compiler_gym_candidate_rollouts": 0, "llvm_phase_search": 0, "llvm_phase_application": 0, "objecttext_measurements": 0, "label_regeneration": 0, "model_training": 0, "sampling": False}, "completed_programs": len(secondary), "invalid_or_failure_programs": 9 - len(secondary), "correctness": load_json(output / "runtime_cohort_manifest.json"), "primary_semantic_cohort": primary_result, "secondary_execution_cohort": secondary_result, "objecttext_runtime_correlation": {"status": "SKIPPED", "reason": "Runtime uses policy45 winning prefixes while existing ObjectText reports are aggregate policy outputs; no unambiguous per-program paired metric is introduced."}, "input_integrity_after_run": check_source_inputs(Path.cwd())}
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def validate(output: Path) -> None:
    cfg = load_output_config(output)
    required = ("policy_prefixes.json", "build_manifest.json", "binary_metadata.json", "correctness_results.jsonl", "runtime_cohort_manifest.json", "raw_timing_samples.jsonl", "timing_summary.json", "runtime_report.json")
    if any(not (output / name).is_file() for name in required):
        raise ValueError("runtime output schema is incomplete")
    builds, timing = load_json(output / "build_manifest.json"), load_json(output / "timing_summary.json")
    if set(builds["programs"]) != set(cfg["population"]["program_ids"]):
        raise ValueError("binary metadata program population mismatch")
    for program in cfg["population"]["program_ids"]:
        if set(builds["programs"][program]["methods"]) != set(METHODS) or set(timing["programs"][program]) != set(METHODS):
            raise ValueError("method inventory mismatch")
    raw = [json.loads(line) for line in (output / "raw_timing_samples.jsonl").read_text(encoding="utf-8").splitlines()]
    if not raw or {row["sample_type"] for row in raw} != {"warmup", "formal"}:
        raise ValueError("raw timing sample schema mismatch")
    print(json.dumps({"status": "COMPLETE", "schema_valid": True, "programs": 9, "methods": 7, "raw_samples": len(raw)}, sort_keys=True))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stage", choices=("freeze", "prefixes", "build", "correctness", "timing", "aggregate", "validate", "all"), required=True)
    args = parser.parse_args()
    root, output = Path.cwd(), args.output_dir
    if args.stage == "freeze": freeze(root, output)
    elif args.stage == "prefixes": recover_prefixes(root, output)
    elif args.stage == "build": build(root, output)
    elif args.stage == "correctness": run_correctness(load_output_config(output), load_json(output / "build_manifest.json"), output)
    elif args.stage == "timing": time_binaries(load_output_config(output), load_json(output / "build_manifest.json"), output)
    elif args.stage == "aggregate": aggregate(output)
    elif args.stage == "validate": validate(output)
    else:
        freeze(root, output); recover_prefixes(root, output); build(root, output)
        run_correctness(load_output_config(output), load_json(output / "build_manifest.json"), output)
        time_binaries(load_output_config(output), load_json(output / "build_manifest.json"), output)
        aggregate(output); validate(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
