#!/usr/bin/env python3
"""Recover frozen Route-A policy-45 prefixes for post-hoc runtime evaluation."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

if __package__:
    from scripts.evaluate_route_a_final_objecttext import (extract_features, load_final_manifest, load_json, load_model, read_label_matrix)
    from scripts.generate_rlcompopt_objecttext_labels import K
else:
    from evaluate_route_a_final_objecttext import (extract_features, load_final_manifest, load_json, load_model, read_label_matrix)
    from generate_rlcompopt_objecttext_labels import K


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assert_artifact(entry: Mapping[str, str]) -> Path:
    path = Path(entry["path"])
    if not path.is_file() or sha256(path) != entry["sha256"]:
        raise ValueError(f"artifact hash mismatch: {path}")
    return path


def selected_prefix(scores: Sequence[float], records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if len(scores) != K or len(records) != K:
        raise ValueError("policy45 requires exactly K=50 scores and records")
    budget, best = 45, None
    for candidate_id in sorted(range(K), key=lambda index: (-scores[index], index)):
        record = records[candidate_id]
        prefix = record["prefix_object_text_size_bytes"]
        actions = record["ordered_pass_sequence"]
        if record["candidate_id"] != candidate_id or len(prefix) != len(actions):
            raise ValueError("candidate record does not preserve frozen order")
        take = min(budget, len(prefix))
        for prefix_index, value in enumerate(prefix[:take]):
            if best is None or value < best["object_text_size_bytes"]:
                best = {
                    "candidate_id": candidate_id,
                    "candidate_rank": sorted(range(K), key=lambda index: (-scores[index], index)).index(candidate_id),
                    "prefix_index": prefix_index,
                    "pass_count": prefix_index + 1,
                    "action_ids": actions[: prefix_index + 1],
                    "object_text_size_bytes": value,
                }
        budget -= take
        if budget == 0:
            break
    if budget != 0 or best is None:
        raise ValueError("policy45 failed to consume exactly 45 pass observations")
    return best


def final_result_value(path: Path, program_id: str) -> int:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["program_id"] == program_id:
                if not row["valid"]:
                    raise ValueError(f"final result invalid for {program_id}: {path}")
                return int(row["policy45_object_text_size_bytes"])
    raise ValueError(f"program missing from final result: {program_id}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output: {args.output_dir}")

    cfg = load_json(args.config)
    if cfg["protocol_class"] != "post_hoc_exploratory_runtime" or not cfg["outcomes_not_seen_at_protocol_freeze"]:
        raise ValueError("not an untouched post-hoc runtime protocol")
    parent = cfg["parent_objecttext_artifacts"]
    final_config = assert_artifact(parent["final_config"])
    final_cfg = load_json(final_config)
    manifest = assert_artifact(parent["final_manifest"])
    final_programs = load_final_manifest(manifest, final_cfg)
    programs = [row["program_id"] for row in cfg["programs"]]
    if not set(programs) <= set(final_programs) or len(programs) != 9:
        raise ValueError("runtime population is not the frozen final-study subset")
    assert_artifact(parent["candidate_file"])
    definitions = cfg["methods"]["inference_definitions"]
    controlled_cfg = load_json(assert_artifact(definitions["controlled_config"]))
    for key in ("nvp_config", "nvp_implementation", "controlled_implementation"):
        assert_artifact(definitions[key])
    matrix, summaries = read_label_matrix(Path(parent["final_label_shards"].rsplit("/", 1)[0]), final_programs)
    if any(program not in matrix or summaries[program]["oracle_K50_validity"] != "valid_complete_K50" for program in programs):
        raise ValueError("runtime program lacks complete frozen K=50 labels")
    features, failures = extract_features(programs, workers=1)
    if failures or set(features) != set(programs):
        raise ValueError(f"Autophase feature recovery failed: {failures}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output: dict[str, Any] = {
        "protocol_config_sha256": sha256(args.config),
        "inference_device": str(device),
        "policy": cfg["methods"]["policy"],
        "programs": {},
    }
    with torch.no_grad():
        for method in cfg["methods"]["learned"]:
            model_name, seed = method["model"], int(method["seed"])
            checkpoint = Path(method["checkpoint"])
            if not checkpoint.is_file() or sha256(checkpoint) != method["sha256"]:
                raise ValueError(f"checkpoint hash mismatch: {checkpoint}")
            model = load_model(model_name, seed, checkpoint.parents[2], controlled_cfg, device)
            values = model(torch.tensor([features[program] for program in programs], dtype=torch.float32, device=device)).detach().cpu().tolist()
            for program, scores in zip(programs, values):
                prefix = selected_prefix(scores, matrix[program])
                expected = final_result_value(Path("outputs/route_a_final_objecttext_v6/model_results") / model_name.lower() / f"seed{seed}.jsonl.gz", program)
                if prefix["object_text_size_bytes"] != expected:
                    raise ValueError(f"final policy45 value mismatch for {model_name}/seed{seed}/{program}")
                output["programs"].setdefault(program, {})[f"{model_name}_seed{seed}"] = prefix
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "policy_prefixes.json").write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "COMPLETE", "program_count": len(programs), "method_count": len(cfg["methods"]["learned"]), "output": str(args.output_dir / "policy_prefixes.json")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
