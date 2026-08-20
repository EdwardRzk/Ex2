#!/usr/bin/env python3
"""Oz-only pre-formal calibration for the frozen Route-A runtime study."""
from __future__ import annotations
import argparse, json, math, os, subprocess, time
from pathlib import Path

ENV={"OMP_NUM_THREADS":"1","MKL_NUM_THREADS":"1","OPENBLAS_NUM_THREADS":"1","NUMEXPR_NUM_THREADS":"1"}

def command(binary: str, argv: list[str]) -> list[str]:
    return [binary if x == "./a.out" else x for x in argv]

def main() -> int:
    p=argparse.ArgumentParser(); p.add_argument("--config",type=Path,required=True);p.add_argument("--build-manifest",type=Path,required=True);p.add_argument("--output",type=Path,required=True);a=p.parse_args()
    if a.output.exists(): raise FileExistsError(a.output)
    cfg=json.loads(a.config.read_text()); build=json.loads(a.build_manifest.read_text()); cpu="0"; programs=[]
    for spec in cfg["programs"]:
        info=build["programs"][spec["program_id"]]["methods"]["oz"]; work=Path(spec["workdir"]); env={"PATH":os.environ["PATH"],**ENV,**spec["env"]}; (work/"_finfo_dataset").write_text("1\n")
        start=time.perf_counter(); done=subprocess.run(["taskset","-c",cpu,*command(str(Path(info["binary"]).resolve()),spec["argv"])],cwd=work,env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=spec["timeout_seconds"],check=False); elapsed=time.perf_counter()-start
        if done.returncode: raise RuntimeError(f"Oz calibration failure: {spec['program_id']}:{done.returncode}")
        factor=max(1,math.ceil(1.0/elapsed)); programs.append({"program_id":spec["program_id"],"canonical_oz_calibration_seconds":elapsed,"amplification_factor":factor,"estimated_amplified_seconds":elapsed*factor,"cpu":{"model":"Intel(R) Xeon(R) Gold 6330 CPU @ 2.00GHz","socket":0,"physical_core":0,"logical_cpu":0,"affinity_command":"taskset -c 0"},"thread_environment":ENV,"applies_to_methods":["oz","nvp_seed1","nvp_seed2","nvp_seed3","mamba_seed1","mamba_seed2","mamba_seed3"]})
    a.output.write_text(json.dumps({"protocol":"POST-HOC / EXPLORATORY","calibration":"Oz-only; not formal timing samples","timing_samples_collected":0,"programs":programs},indent=2,sort_keys=True)+"\n")
    print(json.dumps({"status":"COMPLETE","program_count":len(programs)},sort_keys=True))
if __name__=="__main__": main()
