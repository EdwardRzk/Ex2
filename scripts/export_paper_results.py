#!/usr/bin/env python3
"""Recover frozen formal reports into one canonical paper-results source."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Mapping


MAIN = ("NVP", "MLP", "LSTM", "Transformer", "Mamba", "Direct MambaNVP", "Anchored MambaNVP", "PA-MambaNVP")


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def row(method: str, split: str, record_type: str, source: str, *, seed: int | None = None, dataset: str | None = None, total: int | None = None, valid: int | None = None, mean: float | None = None, delta: float | None = None, regret: float | None = None, recovery: float | None = None, top1: float | None = None, top5: float | None = None, median_delta: float | None = None, positive: int | None = None, negative: int | None = None, leave_llvm: float | None = None) -> dict[str, Any]:
    return {"method": method, "split": split, "record_type": record_type, "seed": seed, "dataset_id": dataset, "N_total": total, "N_valid": valid, "MeanOverOz": mean, "delta_vs_NVP": delta, "policy45_regret_bytes": regret, "oracle_recovery": recovery, "top1_oracle_hit": top1, "top5_oracle_hit": top5, "median_dataset_delta_vs_NVP": median_delta, "positive_dataset_count": positive, "negative_dataset_count": negative, "leave_LLVM_Stress_out_macro_delta_vs_NVP": leave_llvm, "source_artifact_path": source}


def add_validation(records: list[dict[str, Any]], method: str, source: str, summary: Mapping[str, Any], nvp: float, oracle: float) -> None:
    macro = float(summary["ValidationFinalMeanOverOz_3seed"])
    seeds = summary.get("seed_results", [])
    records.append(row(method, "validation", "dataset_macro_3seed", source, total=4488, valid=4488, mean=macro, delta=macro-nvp, regret=summary.get("policy45_regret_mean_bytes_3seed"), recovery=summary.get("oracle_opportunity_recovered", summary.get("oracle_recovery_3seed", macro/oracle)), top1=summary.get("top1_accuracy_3seed"), top5=summary.get("top5_oracle_coverage_3seed")))
    for item in seeds:
        chosen=item["selected"]; records.append(row(method,"validation","dataset_macro_seed",source,seed=int(item["seed"]),total=chosen.get("N_total"),valid=chosen.get("N_primary_valid"),mean=chosen["ValidationFinalMeanOverOz"],delta=chosen["ValidationFinalMeanOverOz"]-nvp,regret=chosen.get("policy45_regret_mean_bytes"),top1=chosen.get("top1_accuracy"),top5=chosen.get("top5_oracle_coverage")))
    per_dataset=summary.get("per_dataset_3seed",{})
    for dataset,value in sorted(per_dataset.items()): records.append(row(method,"validation","dataset_3seed",source,dataset=dataset,mean=value,delta=None))


def add_final(records: list[dict[str, Any]], method: str, source: str, macro: Mapping[str, Any], per_dataset: Mapping[str, Any], nvp_macro: float, nvp_datasets: Mapping[str, Any], *, regret: float | None = None, recovery: float | None = None, top1: float | None = None, top5: float | None = None, median_delta: float | None = None, positive: int | None = None, negative: int | None = None, leave_llvm: float | None = None) -> None:
    mean=float(macro["three_seed_mean"]); records.append(row(method,"final/OOD","dataset_macro_3seed",source,total=4683,valid=4679,mean=mean,delta=mean-nvp_macro,regret=regret,recovery=recovery,top1=top1,top5=top5,median_delta=median_delta,positive=positive,negative=negative,leave_llvm=leave_llvm))
    for seed,value in sorted(macro["per_seed"].items()): records.append(row(method,"final/OOD","dataset_macro_seed",source,seed=int(seed),total=4683,valid=4679,mean=value,delta=value-float(nvp_datasets["__seed_macro__"][seed])))
    for dataset, values in sorted(per_dataset.items()):
        method_values=values[method] if method in values else values
        dataset_mean=float(method_values["three_seed_mean"]); nvp_mean=float(nvp_datasets[dataset]["NVP"]["three_seed_mean"])
        records.append(row(method,"final/OOD","dataset_3seed",source,dataset=dataset,total=values.get("N_total"),valid=values.get("N_primary_valid"),mean=dataset_mean,delta=dataset_mean-nvp_mean))


def export(root: Path, output: Path) -> None:
    if output.exists(): raise FileExistsError(output)
    stage=load(root/"outputs/route_a_stage_b_v6/comparison_report.json"); final=load(root/"outputs/route_a_final_objecttext_v6/comparison_report.json")
    direct_v=load(root/"outputs/mamba_nvp_objecttext_v6/comparison_report.json"); direct_f=load(root/"outputs/mamba_nvp_final_objecttext_v6/comparison_report.json")
    anchored_v=load(root/"outputs/gated_calibrated_mambanvp_v2/comparison_report.json"); anchored_f=load(root/"outputs/gated_calibrated_mambanvp_final_objecttext_v2/comparison_report.json")
    pa=load(root/"outputs/policy_aware_mambanvp_v1/comparison_report.json")
    sources={"Stage-B": "outputs/route_a_stage_b_v6/comparison_report.json", "Route-A-final": "outputs/route_a_final_objecttext_v6/comparison_report.json", "Direct-validation": "outputs/mamba_nvp_objecttext_v6/comparison_report.json", "Direct-final": "outputs/mamba_nvp_final_objecttext_v6/comparison_report.json", "Anchored-validation": "outputs/gated_calibrated_mambanvp_v2/comparison_report.json", "Anchored-final": "outputs/gated_calibrated_mambanvp_final_objecttext_v2/comparison_report.json", "PA": "outputs/policy_aware_mambanvp_v1/comparison_report.json"}
    records=[]; stage_models={item["architecture"]:item for item in stage["models"]}; nvp_v=float(stage_models["NVP"]["ValidationFinalMeanOverOz_3seed"]); oracle_v=float(stage["fixed_route_a_oracle"])
    for name in ("NVP","MLP","LSTM","Transformer","Mamba"): add_validation(records,name,sources["Stage-B"],stage_models[name],nvp_v,oracle_v)
    direct_summary=direct_v["comparison"]; add_validation(records,"Direct MambaNVP",sources["Direct-validation"],direct_summary,nvp_v,oracle_v)
    anchored_summary=anchored_v["gated_calibrated_mambanvp"]; add_validation(records,"Anchored MambaNVP",sources["Anchored-validation"],anchored_summary,nvp_v,oracle_v)
    pa_v={"ValidationFinalMeanOverOz_3seed":pa["validation"]["three_seed_mean_over_oz"],"seed_results":[{"seed":int(seed),"selected":{"N_total":4488,"N_primary_valid":4488,"ValidationFinalMeanOverOz":value}} for seed,value in pa["validation"]["per_seed_mean_over_oz"].items()],"per_dataset_3seed":pa["validation"]["per_dataset"],"policy45_regret_mean_bytes_3seed":pa["validation"]["policy45_regret_mean_bytes"],"top1_accuracy_3seed":pa["validation"]["top1_oracle_tie_accuracy"],"top5_oracle_coverage_3seed":pa["validation"]["top5_oracle_coverage"]}; add_validation(records,"PA-MambaNVP",sources["PA"],pa_v,nvp_v,oracle_v)
    families={item["family"]:item for item in final["comparison_families"]}; h2a,h2b=families["H2a"],families["H2b"]; nvp_f=float(h2a["dataset_macro"]["NVP"]["three_seed_mean"]); nvp_datasets=dict(h2a["per_dataset"]); nvp_datasets["__seed_macro__"]=h2a["dataset_macro"]["NVP"]["per_seed"]
    baseline_sources={"NVP":h2a,"Mamba":h2a,"MLP":h2b,"LSTM":h2b,"Transformer":h2b}
    for name,family in baseline_sources.items(): add_final(records,name,sources["Route-A-final"],family["dataset_macro"][name],family["per_dataset"],nvp_f,nvp_datasets,recovery=float(family["dataset_macro"][name]["three_seed_mean"])/float(final["offline_k50_oracle"]["dataset_macro"]))
    direct_combined=direct_f["combined_comparison"]; add_final(records,"Direct MambaNVP",sources["Direct-final"],direct_combined["dataset_macro"]["MambaNVP"],direct_combined["per_dataset"],nvp_f,nvp_datasets,regret=direct_f["policy45_regret"]["MambaNVP"]["mean_3seed"],recovery=direct_f["mamba_nvp_oracle_recovery"])
    anchored_combined=anchored_f["combined_comparison"]; a=anchored_f["gated_calibrated_mambanvp"]; add_final(records,"Anchored MambaNVP",sources["Anchored-final"],anchored_combined["dataset_macro"]["GatedCalibratedMambaNVP"],anchored_combined["per_dataset"],nvp_f,nvp_datasets,regret=a["policy45_regret"]["mean_3seed"],recovery=a["oracle_recovery"],top1=a["top1_accuracy_3seed"])
    pa_macro={"three_seed_mean":pa["final"]["three_seed_mean_over_oz"],"per_seed":pa["final"]["per_seed_mean_over_oz"]}; pa_datasets={key:{"N_total":None,"N_primary_valid":None,"PA-MambaNVP":value} for key,value in pa["final"]["per_dataset"].items()}; add_final(records,"PA-MambaNVP",sources["PA"],pa_macro,pa_datasets,nvp_f,nvp_datasets,regret=pa["final"]["policy45_regret_mean_bytes"],recovery=pa["final"]["oracle_recovery"],top1=pa["final"]["top1_oracle_tie_accuracy"],top5=pa["final"]["top5_oracle_coverage"],median_delta=pa["final"]["median_dataset_delta_vs_nvp"],positive=pa["final"]["positive_dataset_count_vs_nvp"],negative=pa["final"]["negative_dataset_count_vs_nvp"],leave_llvm=pa["final"]["leave_llvm_stress_out_13dataset_delta_vs_nvp"])
    keys={(x["method"],x["split"],x["record_type"],x["seed"],x["dataset_id"]) for x in records}
    if len(keys)!=len(records): raise ValueError("duplicate canonical entries")
    headline={method:{split:next(x["MeanOverOz"] for x in records if x["method"]==method and x["split"]==split and x["record_type"]=="dataset_macro_3seed") for split in ("validation","final/OOD")} for method in MAIN}
    output.mkdir(parents=True); metadata={"task":"frozen report recovery only","offline_evaluator_rerun_required":False,"compiler_gym_initialized":False,"llvm_execution":False,"objecttext_observation":False,"methods_fully_recoverable":list(MAIN),"methods_partial_metrics":{"MLP":"final secondary metrics unavailable in formal aggregate", "LSTM":"final secondary metrics unavailable in formal aggregate", "Transformer":"final secondary metrics unavailable in formal aggregate"},"optional_cross_candidate":{"status":"not exported to main canonical table","reason":"target main table is restricted to the eight requested methods"},"sources":sources,"records":records,"headline_values":headline}
    (output/"paper_results.json").write_text(json.dumps(metadata,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    fields=list(records[0]);
    with (output/"paper_results.csv").open("w",newline="",encoding="utf-8") as handle: writer=csv.DictWriter(handle,fieldnames=fields); writer.writeheader(); writer.writerows(records)
    with (output/"main_results_table.csv").open("w",newline="",encoding="utf-8") as handle:
        fields2=["method","validation_mean_over_oz","final_ood_mean_over_oz","validation_delta_vs_nvp","final_ood_delta_vs_nvp","paper_display"]; writer=csv.DictWriter(handle,fieldnames=fields2); writer.writeheader()
        for method in MAIN:
            v,f=headline[method]["validation"],headline[method]["final/OOD"]; writer.writerow({"method":method,"validation_mean_over_oz":v,"final_ood_mean_over_oz":f,"validation_delta_vs_nvp":v-nvp_v,"final_ood_delta_vs_nvp":f-nvp_f,"paper_display":f"{v:.5f} / {f:.5f}"})


def main() -> int:
    parser=argparse.ArgumentParser(); parser.add_argument("--output-dir",type=Path,required=True); args=parser.parse_args(); export(Path.cwd(),args.output_dir); return 0


if __name__=="__main__": raise SystemExit(main())
