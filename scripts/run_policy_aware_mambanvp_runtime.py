#!/usr/bin/env python3
"""Supplementary PA-MambaNVP-only runtime measurement under the frozen Route-A protocol."""
from __future__ import annotations

import argparse
import collections
import gzip
import json
import statistics
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

if __package__:
    from scripts import run_mambanvp_nvp_runtime as base
    from scripts.evaluate_mamba_nvp_final_objecttext import load_final_features, read_final_artifacts
    from scripts.train_mamba_nvp_objecttext import load_json
    from scripts.train_policy_aware_mambanvp import METHOD as PA_ARCHITECTURE, load_pa_checkpoint
else:
    import run_mambanvp_nvp_runtime as base
    from evaluate_mamba_nvp_final_objecttext import load_final_features, read_final_artifacts
    from train_mamba_nvp_objecttext import load_json
    from train_policy_aware_mambanvp import METHOD as PA_ARCHITECTURE, load_pa_checkpoint


SEEDS = (1, 2, 3)
METHODS = tuple(f"pa_mambanvp_seed{seed}" for seed in SEEDS)
BASELINE = Path("outputs/gated_calibrated_mambanvp_runtime_v1_retry1")


def checked(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path), "sha256": base.sha256(path)}


def load_cfg(output: Path) -> dict[str, Any]:
    cfg = load_json(output / "config.json")
    if cfg["experiment_name"] != "policy_aware_mambanvp_runtime" or cfg["methods"] != list(METHODS):
        raise ValueError("not the frozen PA-only runtime protocol")
    if cfg["measurement"]["warmup_runs_excluded"] != 3 or cfg["measurement"]["initial_timed_samples"] != 5 or cfg["measurement"]["maximum_timed_samples"] != 20 or cfg["measurement"]["rse_target"] != .01:
        raise ValueError("runtime measurement protocol mismatch")
    return cfg


def freeze(root: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    parent = root / "configs/route_a_posthoc_runtime_v6.json"
    baseline_files = {name: root / BASELINE / name for name in ("config.json", "timing_summary.json", "runtime_cohort_manifest.json", "runtime_report.json", "comparison_report.json")}
    pa_cfg = root / "outputs/policy_aware_mambanvp_v1/config.json"
    controlled = root / "configs/controlled_nvp_stage_a_v6.json"
    final_features = root / "outputs/autophase_feature_cache_v6/final_autophase.jsonl.gz"
    labels = root / "outputs/route_a_final_objecttext_v6/shards/final"
    legacy_prefixes = root / "outputs/route_a_posthoc_runtime_v6/policy_prefixes.json"
    legacy_builds = root / "outputs/route_a_posthoc_runtime_v6/build_manifest.json"
    amplification = root / "outputs/route_a_posthoc_runtime_v6/amplification_manifest.json"
    for path in (parent, pa_cfg, controlled, final_features, legacy_prefixes, legacy_builds, amplification, *baseline_files.values()):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not labels.is_dir():
        raise FileNotFoundError(labels)
    parent_cfg = load_json(parent); baseline_config = load_json(baseline_files["config.json"])
    if len(parent_cfg["programs"]) != 9 or baseline_config["parent_protocol"]["sha256"] != base.sha256(parent):
        raise ValueError("baseline does not match the formal Route-A runtime protocol")
    programs = [{**row, "workdir": str(output / "work" / row["program_id"].rsplit("/", 1)[-1])} for row in parent_cfg["programs"]]
    cfg = {
        "experiment_name": "policy_aware_mambanvp_runtime",
        "purpose": "One PA-MambaNVP-only supplementary runtime measurement; all baseline timings are reused by reference.",
        "parent_protocol": checked(parent),
        "baseline_runtime_source": {name: checked(path) for name, path in baseline_files.items()},
        "frozen_inputs": {"pa_config": checked(pa_cfg), "controlled_config": checked(controlled), "final_autophase_cache": checked(final_features), "final_label_shards": str(labels), "legacy_policy_prefixes": checked(legacy_prefixes), "legacy_build_manifest": checked(legacy_builds), "frozen_oz_amplification": checked(amplification), "pa_checkpoints": {str(seed): checked(root / f"outputs/policy_aware_mambanvp_v1/checkpoints/seed{seed}/model.pt") for seed in SEEDS}},
        "population": {"included_program_count": 9, "program_ids": [row["program_id"] for row in programs]},
        "methods": list(METHODS),
        "inference": {"sampling": False, "ranking": "descending frozen logits; candidate_id ascending ties", "policy": parent_cfg["methods"]["policy"]},
        "binary_provenance": "PA-only exact hash-verified copy of an already-built Route-A binary after action-id prefix equality; missing provenance is a hard stop. No CompilerGym or LLVM phase application.",
        "measurement": parent_cfg["measurement"], "correctness": parent_cfg["correctness"], "aggregation": parent_cfg["aggregation"],
        "forbidden": ["baseline binary timing", "CompilerGym candidate rollout", "LLVM phase search", "LLVM phase application", "ObjectText measurement", "label regeneration", "retraining", "checkpoint selection", "sampling", "runtime-guided sequence selection"],
        "programs": programs,
    }
    output.mkdir(parents=True); (output / "config.json").write_text(json.dumps(cfg, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def verify_frozen(cfg: Mapping[str, Any]) -> None:
    for entry in cfg["frozen_inputs"]["pa_checkpoints"].values():
        base.assert_hash(Path(entry["path"]), entry["sha256"])
    for entry in (cfg["frozen_inputs"]["pa_config"], cfg["frozen_inputs"]["controlled_config"], cfg["frozen_inputs"]["final_autophase_cache"], cfg["frozen_inputs"]["legacy_policy_prefixes"], cfg["frozen_inputs"]["legacy_build_manifest"], cfg["frozen_inputs"]["frozen_oz_amplification"]):
        base.assert_hash(Path(entry["path"]), entry["sha256"])
    for entry in cfg["baseline_runtime_source"].values():
        base.assert_hash(Path(entry["path"]), entry["sha256"])


def recover_prefixes(output: Path) -> None:
    cfg = load_cfg(output); verify_frozen(cfg); destination = output / "policy_prefixes.json"
    if destination.exists():
        raise FileExistsError(destination)
    controlled = load_json(Path(cfg["frozen_inputs"]["controlled_config"]["path"]))
    programs_all, matrix, summaries = read_final_artifacts(Path(cfg["frozen_inputs"]["final_label_shards"]))
    programs = cfg["population"]["program_ids"]
    if not set(programs) <= set(matrix) or any(summaries[p]["ratio_metric_validity"] != "valid_for_ObjectText_ratio_metric" for p in programs):
        raise ValueError("runtime cohort is not within frozen final complete-K50 validity")
    features = load_final_features(Path(cfg["frozen_inputs"]["final_autophase_cache"]["path"]), list(matrix))
    rows: dict[str, Any] = {"config_sha256": base.sha256(output / "config.json"), "compiler_gym_initialized": False, "candidate_rollouts": 0, "llvm_phase_application": 0, "objecttext_measurements": 0, "sampling": False, "programs": {program: {} for program in programs}}
    for seed in SEEDS:
        model = load_pa_checkpoint(Path(cfg["frozen_inputs"]["pa_checkpoints"][str(seed)]["path"]), seed, controlled)
        if model.training or model.nvp.training:
            raise RuntimeError("PA frozen checkpoint is not eval-only")
        with torch.no_grad():
            scores = model(torch.tensor([features[p] for p in programs], dtype=torch.float32, device="cuda")).cpu().tolist()
        for program, score in zip(programs, scores):
            rows["programs"][program][f"pa_mambanvp_seed{seed}"] = base.selected_prefix(score, matrix[program])
    destination.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build(output: Path) -> None:
    cfg = load_cfg(output); verify_frozen(cfg); manifest_path, metadata_path = output / "build_manifest.json", output / "binary_metadata.json"
    if manifest_path.exists() or metadata_path.exists():
        raise FileExistsError("refusing to overwrite PA binary metadata")
    selected = load_json(output / "policy_prefixes.json")["programs"]
    legacy_prefixes = load_json(Path(cfg["frozen_inputs"]["legacy_policy_prefixes"]["path"]))["programs"]
    legacy_builds = load_json(Path(cfg["frozen_inputs"]["legacy_build_manifest"]["path"]))["programs"]
    report: dict[str, Any] = {"config_sha256": base.sha256(output / "config.json"), "execution": "PA_ONLY_COPY_VERIFIED_FROZEN_BINARY_NO_COMPILERGYM_ROLLOUT_NO_LLVM_PHASE_APPLICATION", "programs": {}}
    for spec in cfg["programs"]:
        program = spec["program_id"]; entries = {}; binary_dir = output / "work" / program.rsplit("/", 1)[-1] / "binaries"; binary_dir.mkdir(parents=True)
        for method in METHODS:
            prefix = selected[program][method]
            legacy_key, legacy = base.choose_legacy_source(prefix, legacy_prefixes[program], None)
            source = legacy_builds[program][legacy_key.lower()]
            if source["action_ids"] != prefix["action_ids"]:
                raise ValueError(f"PA selected prefix has no exact legacy action provenance: {program}/{method}")
            binary, bitcode = binary_dir / method, binary_dir / f"{method}.bc"
            entries[method] = {"binary": str(binary), "sha256": base.copy_checked(Path(source["binary"]), binary, source["sha256"]), "bitcode": str(bitcode), "bitcode_sha256": base.copy_checked(Path(source["bitcode"]), bitcode, source["bitcode_sha256"]), "legacy_prefix_key": legacy_key, "candidate_id": prefix["candidate_id"], "candidate_rank": prefix["candidate_rank"], "pass_sequence_id": prefix["candidate_id"], "action_ids": prefix["action_ids"], "pass_count": prefix["pass_count"], "prefix_index": prefix["prefix_index"], "policy45_object_text_size_bytes": prefix["policy45_object_text_size_bytes"]}
        report["programs"][program] = {"methods": entries, "pa_only": set(entries) == set(METHODS)}
    manifest_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    metadata_path.write_text(json.dumps({"config_sha256": base.sha256(output / "config.json"), "binary_count": len(METHODS) * len(cfg["programs"]), "baseline_binaries_newly_timed": 0, "programs": report["programs"]}, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def correctness_run(cfg: Mapping[str, Any], builds: Mapping[str, Any], output: Path) -> None:
    result, cohort = output / "correctness_results.jsonl", output / "runtime_cohort_manifest.json"
    if result.exists() or cohort.exists():
        raise FileExistsError("refusing to overwrite correctness")
    clang = load_json(Path(cfg["parent_protocol"]["path"]))["environment"]["clang"]; rows=[]; by_program={}
    for spec in cfg["programs"]:
        program=spec["program_id"]; root=Path(spec["workdir"]); current=[]
        with tempfile.TemporaryDirectory(dir=root) as text:
            reference=Path(text)/"reference"; done=subprocess.run([clang,spec["source_bitcode"],"-O2","-o",str(reference),*spec["linkopts"]],stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=False)
            if done.returncode: raise RuntimeError(f"reference compile failed: {program}")
            status,_,stdout=base.correctness_run(reference,spec,Path(text))
            if status!="executed": raise RuntimeError(f"reference execution failed: {program}")
            output_name=spec["correctness"].split("_and_",1)[1] if "_and_" in spec["correctness"] else None; expected=(Path(text)/output_name).read_bytes() if output_name else None
            for method in METHODS:
                info=builds["programs"][program]["methods"][method]
                with tempfile.TemporaryDirectory(dir=root) as run_text:
                    run=Path(run_text); execution,code,actual=base.correctness_run(Path(info["binary"]),spec,run); available=spec["correctness"]!="execution_only"
                    if execution=="executed" and available: checked="semantic_validated_pass" if actual==stdout and (output_name is None or ((run/output_name).is_file() and (run/output_name).read_bytes()==expected)) else "semantic_validated_fail"
                    else: checked="execution_only_unverified" if execution=="executed" else execution
                    row={"program_id":program,"method":method,"seed":int(method.rsplit("seed",1)[1]),"candidate_id":info["candidate_id"],"pass_sequence_id":info["pass_sequence_id"],"correctness_status":checked,"execution_status":execution,"exit_code":code,"correctness_check_available":available}; rows.append(row); current.append(row)
        by_program[program]=current
    result.write_text("".join(json.dumps(row,sort_keys=True)+"\n" for row in rows),encoding="utf-8")
    counts=collections.Counter(row["correctness_status"] for row in rows); base_cohort=load_json(Path(cfg["baseline_runtime_source"]["runtime_cohort_manifest.json"]["path"]))["programs"]
    programs={program:{"pa_semantic_valid":all(row["correctness_status"]=="semantic_validated_pass" for row in rows),"pa_execution_valid":all(row["correctness_status"] in {"semantic_validated_pass","execution_only_unverified"} for row in rows),"baseline_primary_valid":base_cohort[program]["primary_runtime_common_valid"],"baseline_secondary_valid":base_cohort[program]["execution_runtime_common_valid"]} for program,rows in by_program.items()}
    cohort.write_text(json.dumps({"config_sha256":base.sha256(output/"config.json"),"N_runtime_programs":9,"N_pa_binaries":len(rows),**{f"N_{key}":counts[key] for key in ("semantic_validated_pass","semantic_validated_fail","execution_only_unverified","execution_failed","timeout")},"N_primary_common_programs":sum(row["pa_semantic_valid"] and row["baseline_primary_valid"] for row in programs.values()),"N_secondary_common_programs":sum(row["pa_execution_valid"] and row["baseline_secondary_valid"] for row in programs.values()),"programs":programs},indent=2,sort_keys=True)+"\n",encoding="utf-8")


def enriched_summary(output: Path) -> None:
    cfg=load_cfg(output); raw=[json.loads(line) for line in (output/"raw_timing_samples.jsonl").read_text(encoding="utf-8").splitlines()]; timing=load_json(output/"timing_summary.json"); builds=load_json(output/"build_manifest.json"); correctness={}
    for line in (output/"correctness_results.jsonl").read_text(encoding="utf-8").splitlines():
        row=json.loads(line); correctness[(row["program_id"],row["method"])]=row["correctness_status"]
    samples=collections.defaultdict(list)
    for row in raw: samples[(row["program_id"],row["method"],row["sample_type"])].append(row["seconds"])
    report={"pa_only":True,"programs":{}}
    for spec in cfg["programs"]:
        program=spec["program_id"]; report["programs"][program]={}
        for method in METHODS:
            info=builds["programs"][program]["methods"][method]; stat=timing["programs"][program][method]; report["programs"][program][method]={"benchmark_id":program,"seed":int(method.rsplit("seed",1)[1]),"selected_candidate_id":info["candidate_id"],"selected_action_ids":info["action_ids"],"cpu_core":0,"thread_count":1,"input_id":spec["argv"],"cache_policy":"inherited_route_a_posthoc_runtime_v6_no_page_cache_flush","warmup_runs":samples[(program,method,"warmup")],"timed_runs_raw":samples[(program,method,"formal")],"mean_runtime":stat["mean_seconds"],"median_runtime":stat["median_seconds"],"sample_std":stat["sample_std_seconds"],"RSE":stat["rse"],"timed_run_count":stat["n"],"stopping_reason":"rse_target" if stat["rse_target_reached"] else "maximum_20_runs_noisy","correctness_result":correctness[(program,method)]}
    (output/"per_benchmark_summary.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n",encoding="utf-8")


def gmean(values: Sequence[float]) -> float | None:
    return base.gmean(values)


def aggregate(output: Path) -> None:
    cfg=load_cfg(output); current=load_json(output/"timing_summary.json"); saved=load_json(Path(cfg["baseline_runtime_source"]["timing_summary.json"]["path"])); cohorts=load_json(output/"runtime_cohort_manifest.json")["programs"]
    primary=[p for p,row in cohorts.items() if row["pa_semantic_valid"] and row["baseline_primary_valid"]]; secondary=[p for p,row in cohorts.items() if row["pa_execution_valid"] and row["baseline_secondary_valid"]]
    def summary(programs: Sequence[str]) -> dict[str, Any]:
        per_seed={}
        for seed in SEEDS:
            method=f"pa_mambanvp_seed{seed}"; rows={}
            for program in programs:
                pa=current["programs"][program][method]["median_seconds"]; old=saved["programs"][program]; rows[program]={"speedup_vs_oz":old["oz"]["median_seconds"]/pa,"speedup_vs_nvp":old[f"nvp_seed{seed}"]["median_seconds"]/pa,"direct_over_pa":old[f"mambanvp_seed{seed}"]["median_seconds"]/pa,"anchored_over_pa":old[f"gatedcalibratedmambanvp_seed{seed}"]["median_seconds"]/pa}
            per_seed[str(seed)]={"pa_geomean_speedup_vs_oz":gmean([row["speedup_vs_oz"] for row in rows.values()]),"pa_geomean_speedup_vs_nvp":gmean([row["speedup_vs_nvp"] for row in rows.values()]),"direct_over_pa":gmean([row["direct_over_pa"] for row in rows.values()]),"anchored_over_pa":gmean([row["anchored_over_pa"] for row in rows.values()]),"per_benchmark":rows}
        return {"N_programs":len(programs),"program_ids":list(programs),"per_seed":per_seed,"three_seed":{"pa_geomean_speedup_vs_oz":gmean([per_seed[str(s)]["pa_geomean_speedup_vs_oz"] for s in SEEDS]),"pa_geomean_speedup_vs_nvp":gmean([per_seed[str(s)]["pa_geomean_speedup_vs_nvp"] for s in SEEDS]),"direct_over_pa":gmean([per_seed[str(s)]["direct_over_pa"] for s in SEEDS]),"anchored_over_pa":gmean([per_seed[str(s)]["anchored_over_pa"] for s in SEEDS])}}
    primary_summary,secondary_summary=summary(primary),summary(secondary); enriched_summary(output); details=load_json(output/"per_benchmark_summary.json"); all_stats=[row for methods in details["programs"].values() for row in methods.values()]
    report={"step_execution":"COMPLETE","protocol":"exact route_a_posthoc_runtime_v6 cohort/CPU/thread/env/warmup/timing/correctness reused; only PA binaries newly timed","baseline_measurements_reused_only":True,"baseline_binary_executions":0,"pa_newly_timed_binaries":len(METHODS)*9,"compiler_gym_initialized":False,"candidate_rollouts":0,"llvm_phase_search":0,"llvm_phase_application":0,"objecttext_measurements":0,"primary_semantic_cohort":primary_summary,"secondary_execution_cohort":secondary_summary,"speedup_vs_o3":{"status":"unavailable","reason":"no same-protocol saved O3 timing result"},"rse":{"target_reached":sum(row["RSE"]<=.01 for row in all_stats),"twenty_run_cap":sum(row["timed_run_count"]==20 and row["RSE"]>.01 for row in all_stats)},"correctness":load_json(output/"runtime_cohort_manifest.json")}
    (output/"comparison_report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n",encoding="utf-8")


def validate(output: Path) -> None:
    cfg=load_cfg(output); required=("policy_prefixes.json","build_manifest.json","binary_metadata.json","correctness_results.jsonl","runtime_cohort_manifest.json","raw_timing_samples.jsonl","timing_summary.json","per_benchmark_summary.json","comparison_report.json")
    if any(not (output/name).is_file() for name in required): raise ValueError("supplementary PA runtime schema incomplete")
    builds=load_json(output/"build_manifest.json"); timing=load_json(output/"timing_summary.json")
    if any(set(builds["programs"][p]["methods"])!=set(METHODS) or set(timing["programs"][p])!=set(METHODS) for p in cfg["population"]["program_ids"]): raise ValueError("only PA method inventory permitted")
    raw=[json.loads(line) for line in (output/"raw_timing_samples.jsonl").read_text(encoding="utf-8").splitlines()]
    if not raw or {row["method"] for row in raw}!=set(METHODS) or {row["sample_type"] for row in raw}!={"warmup","formal"}: raise ValueError("PA-only raw timing inventory mismatch")
    print(json.dumps({"status":"COMPLETE","schema_valid":True,"programs":9,"newly_timed_methods":3,"raw_samples":len(raw)},sort_keys=True))


def main() -> int:
    parser=argparse.ArgumentParser(); parser.add_argument("--output-dir",type=Path,required=True); parser.add_argument("--stage",choices=("freeze","prefixes","build","correctness","timing","aggregate","validate","all"),required=True); args=parser.parse_args(); output=args.output_dir; base.METHODS=METHODS
    if args.stage=="freeze": freeze(Path.cwd(),output)
    elif args.stage=="prefixes": recover_prefixes(output)
    elif args.stage=="build": build(output)
    elif args.stage=="correctness": correctness_run(load_cfg(output),load_json(output/"build_manifest.json"),output)
    elif args.stage=="timing": base.time_binaries(load_cfg(output),load_json(output/"build_manifest.json"),output)
    elif args.stage=="aggregate": aggregate(output)
    elif args.stage=="validate": validate(output)
    else:
        freeze(Path.cwd(),output); recover_prefixes(output); build(output); correctness_run(load_cfg(output),load_json(output/"build_manifest.json"),output); base.time_binaries(load_cfg(output),load_json(output/"build_manifest.json"),output); aggregate(output); validate(output)
    return 0


if __name__=="__main__": raise SystemExit(main())
