#!/usr/bin/env python3
"""Run one non-timing correctness execution for each frozen runtime binary."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def method_status(records: list[dict]) -> tuple[bool, bool, str | None]:
    statuses = {r["correctness_status"] for r in records}
    primary = statuses == {"semantic_validated_pass"}
    secondary = statuses <= {"semantic_validated_pass", "execution_only_unverified"} and len(records) == 7
    reason = None if primary else "; ".join(sorted(statuses))
    return primary, secondary, reason


def run(binary: Path, program: dict, cwd: Path) -> tuple[str, int | None, str, dict[str, str]]:
    (cwd / "_finfo_dataset").write_text("1\n", encoding="utf-8")
    command = [str(binary) if value == "./a.out" else value for value in program["argv"]]
    env = {"PATH": os.environ["PATH"], "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", **program["env"]}
    try:
        done = subprocess.run(command, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=program["timeout_seconds"], check=False)
    except subprocess.TimeoutExpired as error:
        return "timeout", None, "", {"stdout": (error.stdout or b"").decode("utf-8", "replace"), "stderr": (error.stderr or b"").decode("utf-8", "replace")}
    return ("executed" if done.returncode == 0 else "execution_failed"), done.returncode, done.stdout.decode("utf-8", "replace"), {"stdout": done.stdout.decode("utf-8", "replace"), "stderr": done.stderr.decode("utf-8", "replace")}


def main() -> int:
    p = argparse.ArgumentParser(); p.add_argument("--config", type=Path, required=True); p.add_argument("--build-manifest", type=Path, required=True); p.add_argument("--output-dir", type=Path, required=True); a = p.parse_args()
    results_path, cohort_path = a.output_dir / "correctness_results.jsonl", a.output_dir / "runtime_cohort_manifest.json"
    if results_path.exists() or cohort_path.exists(): raise FileExistsError("refusing to overwrite correctness artifacts")
    cfg, builds = json.loads(a.config.read_text()), json.loads(a.build_manifest.read_text())
    if builds["execution"] != "BUILD_ONLY_NO_BENCHMARK_EXECUTION": raise ValueError("build manifest is not build-only")
    rows, by_program = [], {}
    for program in cfg["programs"]:
        pid, key = program["program_id"], program["program_id"].rsplit("/", 1)[-1]
        methods = builds["programs"][pid]["methods"]
        if set(methods) != {"oz", "nvp_seed1", "nvp_seed2", "nvp_seed3", "mamba_seed1", "mamba_seed2", "mamba_seed3"}: raise ValueError(f"method inventory mismatch: {pid}")
        with tempfile.TemporaryDirectory(dir=a.output_dir / "work" / key) as reference_dir:
            reference = Path(reference_dir) / "reference"
            compile_ref = subprocess.run([cfg["environment"]["clang"], program["source_bitcode"], "-O2", "-o", str(reference), *program["linkopts"]], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
            if compile_ref.returncode: raise RuntimeError(f"reference compile failed: {pid}")
            ref_status, _, ref_stdout, _ = run(reference, program, Path(reference_dir))
            if ref_status != "executed": raise RuntimeError(f"reference execution failed: {pid}/{ref_status}")
            output_name = program["correctness"].split("_and_", 1)[1] if "_and_" in program["correctness"] else None
            ref_output = (Path(reference_dir) / output_name).read_bytes() if output_name else None
            program_rows = []
            for method, info in sorted(methods.items()):
                binary = Path(info["binary"])
                if not binary.is_file() or digest(binary) != info["sha256"]: raise ValueError(f"binary missing/corrupt: {binary}")
                with tempfile.TemporaryDirectory(dir=a.output_dir / "work" / key) as work:
                    status, code, stdout, diagnostic = run(binary, program, Path(work))
                    available = program["correctness"] != "execution_only"
                    if status == "executed" and available:
                        output_ok = output_name is None or ((Path(work) / output_name).is_file() and (Path(work) / output_name).read_bytes() == ref_output)
                        correctness = "semantic_validated_pass" if stdout == ref_stdout and output_ok else "semantic_validated_fail"
                    elif status == "executed": correctness = "execution_only_unverified"
                    else: correctness = "execution_failed" if status == "execution_failed" else "timeout"
                    row = {"program_id": pid, "method": method, "seed": None if method == "oz" else int(method.rsplit("seed", 1)[1]), "binary_path": str(binary), "execution_status": status, "exit_code": code, "timeout_status": status == "timeout", "correctness_check_available": available, "correctness_check_type": "cbench_o2_stdout_and_declared_output_difftest" if available else "none_upstream_sha_execution_only", "correctness_check_result": None if not available else correctness != "semantic_validated_fail", "correctness_status": correctness, "diagnostic": diagnostic}
                    rows.append(row); program_rows.append(row)
            by_program[pid] = program_rows
    with results_path.open("w", encoding="utf-8") as f:
        for row in rows: f.write(json.dumps(row, sort_keys=True) + "\n")
    cohorts, counts = {}, {key: sum(r["correctness_status"] == key for r in rows) for key in ("semantic_validated_pass", "semantic_validated_fail", "execution_only_unverified", "execution_failed", "timeout")}
    for pid, records in by_program.items():
        primary, secondary, reason = method_status(records); cohorts[pid] = {"primary_runtime_common_valid": primary, "execution_runtime_common_valid": secondary, "exclusion_reason": reason}
    manifest = {"protocol_config_sha256": digest(a.config), "build_manifest_sha256": digest(a.build_manifest), "timing_samples_collected": 0, "N_runtime_programs": 9, "N_binaries": 63, **{f"N_{k}": v for k, v in counts.items()}, "N_primary_common_programs": sum(v["primary_runtime_common_valid"] for v in cohorts.values()), "N_secondary_execution_common_programs": sum(v["execution_runtime_common_valid"] for v in cohorts.values()), "programs": cohorts}
    cohort_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "COMPLETE", **counts, "primary": manifest["N_primary_common_programs"], "secondary": manifest["N_secondary_execution_common_programs"]}, sort_keys=True))


if __name__ == "__main__": main()
