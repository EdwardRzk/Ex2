#!/usr/bin/env python3
"""Formal frozen single-core timing for Route-A post-hoc runtime study."""
from __future__ import annotations
import argparse,json,math,os,statistics,subprocess,time
from pathlib import Path
ENV={"OMP_NUM_THREADS":"1","MKL_NUM_THREADS":"1","OPENBLAS_NUM_THREADS":"1","NUMEXPR_NUM_THREADS":"1"}
METHODS=["oz","nvp_seed1","nvp_seed2","nvp_seed3","mamba_seed1","mamba_seed2","mamba_seed3"]
def invoke(binary,argv,factor,cwd,env,timeout):
    start=time.perf_counter()
    for _ in range(factor):
        done=subprocess.run(["taskset","-c","0",str(Path(binary).resolve()),*[str(Path(binary).resolve()) if x=="./a.out" else x for x in argv]],cwd=cwd,env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=timeout,check=False)
        if done.returncode: raise RuntimeError(f"execution failed: {done.returncode}")
    return time.perf_counter()-start
def stats(values):
    n=len(values); mean=statistics.mean(values); std=statistics.stdev(values) if n>1 else 0.; rse=(std/math.sqrt(n))/mean if mean else float('inf')
    return {"n":n,"mean_seconds":mean,"median_seconds":statistics.median(values),"sample_std_seconds":std,"rse":rse,"rse_target_reached":rse<=.01}
def main():
 p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--build-manifest',type=Path,required=True);p.add_argument('--amplification',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args(); raw=a.output_dir/'raw_timing_samples.jsonl'; summary=a.output_dir/'timing_summary.json'
 if raw.exists() or summary.exists():raise FileExistsError('refusing to overwrite timing')
 cfg=json.loads(a.config.read_text());build=json.loads(a.build_manifest.read_text()); amps={x['program_id']:x for x in json.loads(a.amplification.read_text())['programs']}; rows=[]; out={"timing_samples_only":True,"warmups_excluded":3,"programs":{}}
 for spec in cfg['programs']:
  pid=spec['program_id'];work=Path(spec['workdir']); env={"PATH":os.environ['PATH'],**ENV,**spec['env']};factor=amps[pid]['amplification_factor'];out['programs'][pid]={}
  for method in METHODS:
   binary=build['programs'][pid]['methods'][method]['binary']
   for _ in range(3):invoke(binary,spec['argv'],factor,work,env,spec['timeout_seconds'])
   vals=[]
   while len(vals)<5 or (stats(vals)['rse']>.01 and len(vals)<20):
    value=invoke(binary,spec['argv'],factor,work,env,spec['timeout_seconds']);vals.append(value);rows.append({"program_id":pid,"method":method,"sample_index":len(vals),"seconds":value,"amplification_factor":factor})
   out['programs'][pid][method]=stats(vals)
 raw.write_text(''.join(json.dumps(x,sort_keys=True)+'\n' for x in rows));summary.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n');print(json.dumps({"status":"COMPLETE","samples":len(rows)},sort_keys=True))
if __name__=='__main__':main()
