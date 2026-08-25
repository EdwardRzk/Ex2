#!/usr/bin/env python3
"""Run frozen Gated-Calibrated MambaNVP with the Route-A runtime protocol."""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

if __package__:
    from scripts import run_mambanvp_nvp_runtime as base
    from scripts.evaluate_gated_calibrated_mambanvp_final import load_gated, selected_steps_from_report
    from scripts.evaluate_mamba_nvp_final_objecttext import load_final_features, load_mamba_nvp, read_final_artifacts
    from scripts.train_set_conditioned_mamba_ranker import load_json
else:
    import run_mambanvp_nvp_runtime as base
    from evaluate_gated_calibrated_mambanvp_final import load_gated, selected_steps_from_report
    from evaluate_mamba_nvp_final_objecttext import load_final_features, load_mamba_nvp, read_final_artifacts
    from train_set_conditioned_mamba_ranker import load_json

SEEDS = (1, 2, 3)
MODELS = ("NVP", "MambaNVP", "GatedCalibratedMambaNVP")
METHODS = ("oz", *(f"nvp_seed{s}" for s in SEEDS), *(f"mambanvp_seed{s}" for s in SEEDS), *(f"gatedcalibratedmambanvp_seed{s}" for s in SEEDS))


def method_id(model: str, seed: int) -> str:
    return f"{model.lower()}_seed{seed}"


def validate_config(cfg: Mapping[str, Any]) -> None:
    if cfg["experiment_name"] != "gated_calibrated_mambanvp_runtime_v1":
        raise ValueError("wrong experiment")
    if cfg["protocol_class"] != "post_hoc_exploratory_runtime" or cfg["population"]["included_program_count"] != 9:
        raise ValueError("parent runtime cohort changed")
    if cfg["inference"]["sampling"] or cfg["inference"]["ranking"] != "descending frozen logits; candidate_id ascending ties" or cfg["inference"]["policy"]["scored_pass_budget"] != 45:
        raise ValueError("frozen inference mismatch")
    expected = [{"model": m, "seed": s} for m in MODELS for s in SEEDS]
    if cfg["methods"]["learned"] != expected or cfg["gated_selected_steps"] != {"1": 3400, "2": 500, "3": 3500}:
        raise ValueError("frozen model/checkpoint inventory mismatch")


def inputs(root: Path) -> dict[str, Any]:
    result = base.check_source_inputs(root)
    extra = {
        "gated_final_config": root / "configs/gated_calibrated_mambanvp_final_objecttext_v2.json",
        "gated_training_config": root / "configs/gated_calibrated_mambanvp_v2.json",
        "gated_validation_report": root / "outputs/gated_calibrated_mambanvp_v2/comparison_report.json",
    }
    for key, path in extra.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        result[key] = {"path": str(path), "sha256": base.sha256(path)}
    result["GatedCalibratedMambaNVP"] = {str(s): {"path": str(root / f"outputs/gated_calibrated_mambanvp_v2/checkpoints/seed{s}/model.pt"), "sha256": base.sha256(root / f"outputs/gated_calibrated_mambanvp_v2/checkpoints/seed{s}/model.pt")} for s in SEEDS}
    return result


def check_inputs(cfg: Mapping[str, Any]) -> None:
    def check(entry: Mapping[str, Any]) -> None:
        if "path" in entry:
            base.assert_hash(Path(entry["path"]), entry["sha256"])
        else:
            for value in entry.values():
                if isinstance(value, Mapping):
                    check(value)
    for value in cfg["frozen_inputs"].values():
        if isinstance(value, Mapping):
            check(value)


def load_cfg(output: Path) -> dict[str, Any]:
    cfg = load_json(output / "config.json")
    validate_config(cfg)
    return cfg


def freeze(root: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    parent = load_json(root / "configs/route_a_posthoc_runtime_v6.json")
    frozen = inputs(root)
    selected = selected_steps_from_report(Path(frozen["gated_validation_report"]["path"]))
    if selected != {1: 3400, 2: 500, 3: 3500}:
        raise ValueError("unexpected Gated selections")
    programs = [{**row, "workdir": str(output / "work" / row["program_id"].rsplit("/", 1)[-1])} for row in parent["programs"]]
    cfg = {
        "experiment_name": "gated_calibrated_mambanvp_runtime_v1",
        "protocol_class": "post_hoc_exploratory_runtime",
        "purpose": "Frozen Gated-Calibrated MambaNVP runtime evaluation only",
        "parent_protocol": frozen["route_a_posthoc_runtime_config"],
        "frozen_inputs": frozen,
        "gated_selected_steps": {str(k): v for k, v in selected.items()},
        "population": {"included_program_count": 9, "program_ids": [p["program_id"] for p in programs]},
        "methods": {"baseline": "Oz", "learned": [{"model": m, "seed": s} for m in MODELS for s in SEEDS]},
        "inference": {"sampling": False, "ranking": "descending frozen logits; candidate_id ascending ties", "policy": parent["methods"]["policy"]},
        "binary_provenance": "Exact action-id-matched, hash-verified copies of existing Route-A binaries only; no CompilerGym rollout or LLVM phase application.",
        "measurement": parent["measurement"], "correctness": parent["correctness"],
        "aggregation": {**parent["aggregation"], "primary_cohort": "semantic validation passes for Oz and every listed learned seed", "secondary_cohort": "all listed methods execute successfully", "direct_comparison": "GMeanSpeedup_GatedCalibratedMambaNVP / GMeanSpeedup_NVP"},
        "forbidden": ["CompilerGym candidate rollout", "LLVM phase search", "LLVM phase application", "ObjectText measurement", "label regeneration", "retraining", "checkpoint selection", "sampling", "final/OOD re-evaluation"],
        "programs": programs,
    }
    validate_config(cfg)
    output.mkdir(parents=True)
    (output / "config.json").write_text(json.dumps(cfg, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def recover_prefixes(output: Path) -> None:
    cfg = load_cfg(output); target = output / "policy_prefixes.json"
    if target.exists(): raise FileExistsError(target)
    check_inputs(cfg)
    all_programs, matrix, summaries = read_final_artifacts(Path(cfg["frozen_inputs"]["final_label_shards"]))
    eligible = [p for p in all_programs if p in matrix and summaries[p]["ratio_metric_validity"] == "valid_for_ObjectText_ratio_metric"]
    programs = cfg["population"]["program_ids"]
    if not set(programs) <= set(eligible): raise ValueError("runtime program not in frozen complete K50 cohort")
    features = load_final_features(Path(cfg["frozen_inputs"]["final_autophase_cache"]["path"]), eligible)
    final_cfg = load_json(Path(cfg["frozen_inputs"]["mambanvp_final_config"]["path"])); mnvp_cfg = load_json(Path(cfg["frozen_inputs"]["mambanvp_training_config"]["path"])); controlled = load_json(Path(cfg["frozen_inputs"]["controlled_config"]["path"]))
    gated_final = load_json(Path(cfg["frozen_inputs"]["gated_final_config"]["path"])); gated_train = load_json(Path(cfg["frozen_inputs"]["gated_training_config"]["path"])); selected = selected_steps_from_report(Path(cfg["frozen_inputs"]["gated_validation_report"]["path"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result: dict[str, Any] = {"config_sha256": base.sha256(output / "config.json"), "compiler_gym_initialized": False, "candidate_rollouts": 0, "llvm_phase_application": 0, "objecttext_measurements": 0, "sampling": False, "inference_device": str(device), "programs": {p: {} for p in programs}}
    for model_name in MODELS:
        for seed in SEEDS:
            if model_name == "NVP":
                checkpoint = cfg["frozen_inputs"]["checkpoints"]["NVP"][str(seed)]
                model = base.load_frozen_nvp(Path(checkpoint["path"]), seed).to(device).eval()
            elif model_name == "MambaNVP":
                model = load_mamba_nvp(seed, final_cfg, mnvp_cfg, controlled, device)
            else:
                model = load_gated(seed, gated_final, gated_train, selected, device)
            if model.training: raise RuntimeError("frozen model unexpectedly in train mode")
            with torch.no_grad():
                scores = model(torch.tensor([features[p] for p in programs], dtype=torch.float32, device=device)).cpu().tolist()
            for program, score in zip(programs, scores): result["programs"][program][method_id(model_name, seed)] = base.selected_prefix(score, matrix[program])
    target.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build(output: Path) -> None:
    cfg = load_cfg(output); manifest, metadata = output / "build_manifest.json", output / "binary_metadata.json"
    if manifest.exists() or metadata.exists(): raise FileExistsError("refusing to overwrite builds")
    check_inputs(cfg)
    selected = load_json(output / "policy_prefixes.json")["programs"]
    legacy_prefixes = load_json(Path(cfg["frozen_inputs"]["legacy_policy_prefixes"]["path"]))["programs"]
    legacy_builds = load_json(Path(cfg["frozen_inputs"]["legacy_build_manifest"]["path"]))["programs"]
    report: dict[str, Any] = {"config_sha256": base.sha256(output / "config.json"), "execution": "COPY_VERIFIED_FROZEN_BINARY_NO_COMPILERGYM_ROLLOUT_NO_LLVM_PHASE_APPLICATION", "programs": {}}
    for spec in cfg["programs"]:
        program = spec["program_id"]; binary_dir = output / "work" / program.rsplit("/", 1)[-1] / "binaries"; binary_dir.mkdir(parents=True, exist_ok=True)
        legacy_methods = legacy_builds[program]["methods"]; rows: dict[str, Any] = {}
        oz = legacy_methods["oz"]; destination = binary_dir / "oz"
        rows["oz"] = {"binary": str(destination), "sha256": base.copy_checked(Path(oz["binary"]), destination, oz["sha256"]), "candidate_id": None, "pass_sequence_id": None, "action_ids": None, "pass_count": None, "prefix_index": None, "legacy_source_method": "oz"}
        for model_name in MODELS:
            for seed in SEEDS:
                identifier = method_id(model_name, seed); prefix = selected[program][identifier]
                preferred = f"NVP_seed{seed}" if model_name == "NVP" else None
                legacy_key, _ = base.choose_legacy_source(prefix, legacy_prefixes[program], preferred); source = legacy_methods[legacy_key.lower()]
                if source["action_ids"] != prefix["action_ids"]: raise ValueError(f"action mismatch {program}/{identifier}")
                binary, bitcode = binary_dir / identifier, binary_dir / f"{identifier}.bc"
                rows[identifier] = {"binary": str(binary), "sha256": base.copy_checked(Path(source["binary"]), binary, source["sha256"]), "bitcode": str(bitcode), "bitcode_sha256": base.copy_checked(Path(source["bitcode"]), bitcode, source["bitcode_sha256"]), "legacy_source_method": legacy_key.lower(), "legacy_prefix_key": legacy_key, "candidate_id": prefix["candidate_id"], "candidate_rank": prefix["candidate_rank"], "pass_sequence_id": prefix["candidate_id"], "action_ids": prefix["action_ids"], "pass_count": prefix["pass_count"], "prefix_index": prefix["prefix_index"], "policy45_object_text_size_bytes": prefix["policy45_object_text_size_bytes"]}
        report["programs"][program] = {"methods": rows, "all_expected_methods_built": set(rows) == set(METHODS)}
    manifest.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    metadata.write_text(json.dumps({"config_sha256": base.sha256(output / "config.json"), "binary_count": len(METHODS) * len(cfg["programs"]), "binary_reuse": "exact_hash_checked_copy", "programs": report["programs"]}, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def correctness(cfg: Mapping[str, Any], builds: Mapping[str, Any], output: Path) -> None:
    result_file, cohort_file = output / "correctness_results.jsonl", output / "runtime_cohort_manifest.json"
    if result_file.exists() or cohort_file.exists(): raise FileExistsError("refusing to overwrite correctness")
    clang = load_json(Path(cfg["parent_protocol"]["path"]))["environment"]["clang"]; rows, per_program = [], {}
    for spec in cfg["programs"]:
        program = spec["program_id"]; root = output / "work" / program.rsplit("/", 1)[-1]
        with tempfile.TemporaryDirectory(dir=root) as reference_text:
            reference_dir = Path(reference_text); reference = reference_dir / "reference"
            done = subprocess.run([clang, spec["source_bitcode"], "-O2", "-o", str(reference), *spec["linkopts"]], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
            if done.returncode: raise RuntimeError(f"reference compile failed: {program}")
            status, _, ref_stdout = base.correctness_run(reference, spec, reference_dir)
            if status != "executed": raise RuntimeError(f"reference execution failed: {program}")
            output_name = spec["correctness"].split("_and_", 1)[1] if "_and_" in spec["correctness"] else None; ref_output = (reference_dir / output_name).read_bytes() if output_name else None; current = []
            for method in METHODS:
                info = builds["programs"][program]["methods"][method]; binary = Path(info["binary"])
                if base.sha256(binary) != info["sha256"]: raise ValueError(f"binary hash mismatch: {binary}")
                with tempfile.TemporaryDirectory(dir=root) as work_text:
                    work = Path(work_text); execution, code, stdout = base.correctness_run(binary, spec, work); available = spec["correctness"] != "execution_only"
                    if execution == "executed" and available:
                        output_ok = output_name is None or ((work / output_name).is_file() and (work / output_name).read_bytes() == ref_output); checked = "semantic_validated_pass" if stdout == ref_stdout and output_ok else "semantic_validated_fail"
                    else: checked = "execution_only_unverified" if execution == "executed" else execution
                    row = {"program_id": program, "method": method, "seed": None if method == "oz" else int(method.rsplit("seed", 1)[1]), "binary_path": str(binary), "binary_sha256": info["sha256"], "candidate_id": info["candidate_id"], "pass_sequence_id": info["pass_sequence_id"], "correctness_status": checked, "execution_status": execution, "exit_code": code, "correctness_check_available": available}; rows.append(row); current.append(row)
            per_program[program] = current
    result_file.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    counts = Counter(row["correctness_status"] for row in rows)
    cohorts = {p: {"primary_runtime_common_valid": {r["correctness_status"] for r in group} == {"semantic_validated_pass"}, "execution_runtime_common_valid": {r["correctness_status"] for r in group} <= {"semantic_validated_pass", "execution_only_unverified"}, "exclusion_reason": None if {r["correctness_status"] for r in group} == {"semantic_validated_pass"} else "; ".join(sorted({r["correctness_status"] for r in group}))} for p, group in per_program.items()}
    cohort_file.write_text(json.dumps({"config_sha256": base.sha256(output / "config.json"), "N_runtime_programs": len(cfg["programs"]), "N_binaries": len(rows), **{f"N_{key}": counts[key] for key in ("semantic_validated_pass", "semantic_validated_fail", "execution_only_unverified", "execution_failed", "timeout")}, "N_primary_common_programs": sum(row["primary_runtime_common_valid"] for row in cohorts.values()), "N_secondary_execution_common_programs": sum(row["execution_runtime_common_valid"] for row in cohorts.values()), "programs": cohorts}, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def aggregate(output: Path) -> None:
    cfg = load_cfg(output); runtime_path, comparison_path = output / "runtime_report.json", output / "comparison_report.json"
    if runtime_path.exists() or comparison_path.exists(): raise FileExistsError("refusing to overwrite report")
    status = {p: {} for p in cfg["population"]["program_ids"]}
    for line in (output / "correctness_results.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line); status[row["program_id"]][row["method"]] = row
    timing = load_json(output / "timing_summary.json")
    primary = [p for p, rows in status.items() if all(rows[m]["correctness_status"] == "semantic_validated_pass" for m in METHODS)]
    secondary = [p for p, rows in status.items() if all(rows[m]["correctness_status"] in {"semantic_validated_pass", "execution_only_unverified"} for m in METHODS)]
    def summarize(programs: Sequence[str]) -> dict[str, Any]:
        per_seed: dict[str, Any] = {}
        for seed in SEEDS:
            per_seed[str(seed)] = {}
            for model in MODELS:
                ident = method_id(model, seed); speed = {p: timing["programs"][p]["oz"]["median_seconds"] / timing["programs"][p][ident]["median_seconds"] for p in programs}
                per_seed[str(seed)][model] = {"geomean_speedup_vs_Oz": base.gmean(list(speed.values())), "median_speedup_vs_Oz": statistics.median(speed.values()) if speed else None, "improvement_count": sum(v > 1.0 for v in speed.values()), "regression_count": sum(v < 1.0 for v in speed.values()), "per_program_speedup_vs_Oz": speed}
        three = {m: {"geomean_speedup_vs_Oz": base.gmean([per_seed[str(s)][m]["geomean_speedup_vs_Oz"] for s in SEEDS]), "median_speedup_vs_Oz": statistics.median([per_seed[str(s)][m]["median_speedup_vs_Oz"] for s in SEEDS]), "improvement_count_median_across_seeds": statistics.median([per_seed[str(s)][m]["improvement_count"] for s in SEEDS]), "regression_count_median_across_seeds": statistics.median([per_seed[str(s)][m]["regression_count"] for s in SEEDS])} for m in MODELS}
        return {"N_programs": len(programs), "program_ids": list(programs), "per_seed": per_seed, "three_seed": three, "gated_over_nvp_speedup_ratio": three["GatedCalibratedMambaNVP"]["geomean_speedup_vs_Oz"] / three["NVP"]["geomean_speedup_vs_Oz"], "gated_over_mambanvp_speedup_ratio": three["GatedCalibratedMambaNVP"]["geomean_speedup_vs_Oz"] / three["MambaNVP"]["geomean_speedup_vs_Oz"]}
    primary_result, secondary_result = summarize(primary), summarize(secondary)
    cohort = load_json(output / "runtime_cohort_manifest.json")
    runtime = {"step_execution": "COMPLETE", "protocol": "route_a_posthoc_runtime_v6 reused unchanged for benchmark set, binary provenance, LLVM toolchain, CPU binding, warmups, timed repetitions, and correctness", "prohibitions_observed": {"compiler_gym_initialized": False, "compiler_gym_candidate_rollouts": 0, "llvm_phase_search": 0, "llvm_phase_application": 0, "objecttext_measurements": 0, "label_regeneration": 0, "model_training": 0, "sampling": False}, "completed_programs": len(secondary), "invalid_or_failure_programs": len(cfg["programs"]) - len(secondary), "correctness": cohort, "primary_semantic_cohort": primary_result, "secondary_execution_cohort": secondary_result, "input_integrity_after_run": inputs(Path.cwd())}
    comparison = {"step_execution": "COMPLETE", "comparison_models": ["Oz", *MODELS], "common_primary_semantic_cohort": primary_result, "common_secondary_execution_cohort": secondary_result, "correctness": cohort, "timeout_or_failure_count": runtime["invalid_or_failure_programs"], "cross_candidate_runtime": {"included": False, "reason": "no existing frozen Cross-Candidate runtime artifact"}}
    runtime_path.write_text(json.dumps(runtime, indent=2, sort_keys=True) + "\n", encoding="utf-8"); comparison_path.write_text(json.dumps(comparison, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def validate(output: Path) -> None:
    cfg = load_cfg(output); required = ("policy_prefixes.json", "build_manifest.json", "binary_metadata.json", "correctness_results.jsonl", "runtime_cohort_manifest.json", "raw_timing_samples.jsonl", "timing_summary.json", "runtime_report.json", "comparison_report.json")
    if any(not (output / name).is_file() for name in required): raise ValueError("runtime output schema incomplete")
    builds, timing = load_json(output / "build_manifest.json"), load_json(output / "timing_summary.json")
    for program in cfg["population"]["program_ids"]:
        if set(builds["programs"][program]["methods"]) != set(METHODS) or set(timing["programs"][program]) != set(METHODS): raise ValueError("method inventory mismatch")
    raw = [json.loads(line) for line in (output / "raw_timing_samples.jsonl").read_text(encoding="utf-8").splitlines()]
    if not raw or {r["sample_type"] for r in raw} != {"warmup", "formal"}: raise ValueError("timing sample schema mismatch")
    print(json.dumps({"status": "COMPLETE", "schema_valid": True, "programs": len(cfg["programs"]), "methods": len(METHODS), "raw_samples": len(raw)}, sort_keys=True))


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--output-dir", type=Path, required=True); parser.add_argument("--stage", choices=("freeze", "prefixes", "build", "correctness", "timing", "aggregate", "validate", "all"), required=True); args = parser.parse_args()
    base.METHODS = METHODS
    if args.stage == "freeze": freeze(Path.cwd(), args.output_dir)
    elif args.stage == "prefixes": recover_prefixes(args.output_dir)
    elif args.stage == "build": build(args.output_dir)
    elif args.stage == "correctness": correctness(load_cfg(args.output_dir), load_json(args.output_dir / "build_manifest.json"), args.output_dir)
    elif args.stage == "timing": base.time_binaries(load_cfg(args.output_dir), load_json(args.output_dir / "build_manifest.json"), args.output_dir)
    elif args.stage == "aggregate": aggregate(args.output_dir)
    elif args.stage == "validate": validate(args.output_dir)
    else:
        freeze(Path.cwd(), args.output_dir); recover_prefixes(args.output_dir); build(args.output_dir); correctness(load_cfg(args.output_dir), load_json(args.output_dir / "build_manifest.json"), args.output_dir); base.time_binaries(load_cfg(args.output_dir), load_json(args.output_dir / "build_manifest.json"), args.output_dir); aggregate(args.output_dir); validate(args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
