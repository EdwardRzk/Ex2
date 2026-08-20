#!/usr/bin/env python3
"""Build frozen Route-A runtime binaries without executing them."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Mapping


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def method_id(method: Mapping[str, Any]) -> str:
    return f"{method['model'].lower()}_seed{int(method['seed'])}"

def prefix_key(method: Mapping[str, Any]) -> str:
    return f"{method['model']}_seed{int(method['seed'])}"



def compile_binary(command: list[str]) -> dict[str, Any]:
    completed = subprocess.run(command, check=False, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if completed.returncode:
        raise RuntimeError(f"compile failed: {command}\n{completed.stdout}")
    return {"command": command, "stdout": completed.stdout}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--prefixes", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest_path = args.output_dir / "build_manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite build manifest: {manifest_path}")
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    prefixes = json.loads(args.prefixes.read_text(encoding="utf-8"))
    if prefixes["protocol_config_sha256"] != sha256(args.config):
        raise ValueError("prefixes do not match frozen protocol config")
    clang = Path(cfg["environment"]["clang"])
    if not clang.is_file():
        raise FileNotFoundError(clang)
    methods = [{"id": "oz", "model": None, "seed": None}] + [
        {"id": method_id(method), **method} for method in cfg["methods"]["learned"]
    ]
    expected = {method["id"] for method in methods}
    report: dict[str, Any] = {"protocol_config_sha256": sha256(args.config), "prefixes_sha256": sha256(args.prefixes), "execution": "BUILD_ONLY_NO_BENCHMARK_EXECUTION", "programs": {}}
    import compiler_gym

    for program in cfg["programs"]:
        program_id = program["program_id"]
        source = Path(program["source_bitcode"])
        if not source.is_file() or sha256(source) != program["source_sha256"]:
            raise ValueError(f"source bitcode hash mismatch: {source}")
        key = program_id.rsplit("/", 1)[-1]
        frozen = prefixes["programs"].get(program_id, {})
        if set(frozen) != expected - {"oz"}:
            raise ValueError(f"missing frozen prefixes for {program_id}")
        program_root = args.output_dir / "work" / key
        binaries = program_root / "binaries"
        binaries.mkdir(parents=True, exist_ok=False)
        built: dict[str, Any] = {}
        oz = binaries / "oz"
        built["oz"] = compile_binary([str(clang), str(source), "-Oz", "-o", str(oz)] + list(program["linkopts"]))
        if not oz.is_file():
            raise RuntimeError(f"missing Oz binary: {oz}")
        built["oz"].update({"binary": str(oz), "sha256": sha256(oz), "pass_count": None, "action_ids": None})
        for method in methods[1:]:
            selected = frozen[prefix_key(method)]
            # The prefix JSON keys intentionally retain model capitalization.
            env = compiler_gym.make("llvm-v0", reward_space=None)
            try:
                env.reset(benchmark=program_id)
                for action in selected["action_ids"]:
                    _, _, done, _ = env.step(int(action))
                    if done:
                        raise RuntimeError(f"episode ended during frozen prefix: {program_id}/{method['id']}")
                bitcode = binaries / f"{method['id']}.bc"
                env.write_bitcode(bitcode)
            finally:
                env.close()
            binary = binaries / method["id"]
            build = compile_binary([str(clang), str(bitcode), "-o", str(binary)] + list(program["linkopts"]))
            if not binary.is_file():
                raise RuntimeError(f"missing learned binary: {binary}")
            build.update({"binary": str(binary), "bitcode": str(bitcode), "bitcode_sha256": sha256(bitcode), "sha256": sha256(binary), "pass_count": selected["pass_count"], "action_ids": selected["action_ids"], "candidate_id": selected["candidate_id"], "prefix_index": selected["prefix_index"], "policy45_object_text_size_bytes": selected["object_text_size_bytes"]})
            built[method["id"]] = build
        report["programs"][program_id] = {"methods": built, "all_expected_methods_built": set(built) == expected}
    manifest_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "COMPLETE", "program_count": len(report["programs"]), "output": str(manifest_path)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
