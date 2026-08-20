#!/usr/bin/env python3
"""Train frozen-data Cross-Candidate MambaNVP on validation only."""
from __future__ import annotations
import argparse, copy, json, random
from pathlib import Path
from typing import Any, Mapping
import numpy as np
import torch
from mamba_ssm import Mamba
from torch import nn

if __package__:
    from scripts.train_controlled_nvp_stage_a import K, _CommonCandidateInterface, evaluate, learning_rate, load_candidates, read_jsonl, read_label_matrix, soft_cross_entropy
    from scripts.train_mamba_nvp_objecttext import load_feature_cache, load_frozen_nvp
else:
    from train_controlled_nvp_stage_a import K, _CommonCandidateInterface, evaluate, learning_rate, load_candidates, read_jsonl, read_label_matrix, soft_cross_entropy
    from train_mamba_nvp_objecttext import load_feature_cache, load_frozen_nvp

def load_json(path: Path) -> dict[str, Any]: return json.loads(path.read_text(encoding="utf-8"))

class CrossCandidateMambaNVP(nn.Module):
    def __init__(self, nvp: nn.Module, cfg: Mapping[str, Any], tokens: torch.Tensor, lengths: torch.Tensor) -> None:
        super().__init__(); self.nvp=nvp; self.encoder=_CommonCandidateInterface(cfg,tokens,lengths); d=int(cfg["d_model"])
        self.block_norms=nn.ModuleList([nn.LayerNorm(d) for _ in range(int(cfg["layers"]))])
        self.blocks=nn.ModuleList([Mamba(d_model=d,d_state=int(cfg["d_state"]),d_conv=int(cfg["d_conv"]),expand=int(cfg["expand"]),use_fast_path=bool(cfg["use_fast_path"]),layer_idx=i) for i in range(int(cfg["layers"]))])
        interaction=cfg["candidate_interaction"]
        self.attention=nn.ModuleList([nn.MultiheadAttention(d,int(interaction["num_heads"]),dropout=float(interaction["dropout"]),batch_first=True) for _ in range(int(interaction["layers"]))])
        self.attention_norms=nn.ModuleList([nn.LayerNorm(d) for _ in range(int(interaction["layers"]))])
        self.output_norm=nn.LayerNorm(d); self.value_head=nn.Linear(d,1)
        nn.init.zeros_(self.value_head.weight); nn.init.zeros_(self.value_head.bias)
        for p in self.nvp.parameters(): p.requires_grad_(False)
        self.nvp.eval()
    def train(self, mode: bool=True): super().train(mode); self.nvp.eval(); return self
    def residual_logits(self, program: torch.Tensor) -> torch.Tensor:
        hidden,lengths,_=self.encoder.candidate_inputs(program)
        for norm,block in zip(self.block_norms,self.blocks): hidden=hidden+block(norm(hidden))
        rows=torch.arange(len(hidden),device=hidden.device); h=hidden[rows,lengths-1].reshape(program.shape[0],K,-1)
        for attn,norm in zip(self.attention,self.attention_norms):
            update,_=attn(norm(h),norm(h),norm(h),need_weights=False); h=h+update
        return self.value_head(self.output_norm(h)).squeeze(-1)
    def forward(self, program: torch.Tensor) -> torch.Tensor:
        with torch.no_grad(): base=self.nvp(program)
        return base+self.residual_logits(program)
    def trainable_parameter_count(self) -> int: return sum(p.numel() for p in self.parameters() if p.requires_grad)

def seed_everything(seed:int)->None: random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
def validate(cfg:Mapping[str,Any], controlled:Mapping[str,Any])->None:
    assert cfg["final_seed_set"]==[1,2,3] and cfg["target_and_objective"]["target_temperature"]==0.05
    assert cfg["training"]["total_steps"]==10000 and not cfg["training"]["early_stopping"]
    assert cfg["architecture"]["candidate_interaction"]=={"layers":2,"num_heads":4,"dropout":0.0}
    assert controlled["candidate_representation"]["K"]==K==50

def train_seed(cfg, controlled, seed, tokens, lengths, train_x, train_y, val_x, val_y, records, matrix, out):
    seed_everything(seed); model_cfg={**controlled["candidate_representation"],**controlled["models"]["Mamba"],"candidate_interaction":cfg["architecture"]["candidate_interaction"]}
    model=CrossCandidateMambaNVP(load_frozen_nvp(Path(cfg["nvp_checkpoint_root"])/f"seed{seed}"/"model.pt",seed),model_cfg,tokens,lengths).cuda()
    opt=torch.optim.Adam((p for p in model.parameters() if p.requires_grad),lr=float(cfg["training"]["learning_rate"]),weight_decay=float(cfg["training"]["weight_decay"])); rng=torch.Generator().manual_seed(seed); out.mkdir(parents=True)
    curve=[]; best=None; step=0
    while step<int(cfg["training"]["total_steps"]):
        order=torch.randperm(len(train_x),generator=rng)
        for begin in range(0,len(train_x),int(cfg["training"]["batch_size"])):
            step+=1; idx=order[begin:begin+int(cfg["training"]["batch_size"])].cuda(non_blocking=True)
            lr=learning_rate(cfg["training"],step)
            for group in opt.param_groups: group["lr"]=lr
            model.train(); loss=soft_cross_entropy(model(train_x[idx]),train_y[idx]); opt.zero_grad(set_to_none=True); loss.backward()
            if any(p.grad is not None for p in model.nvp.parameters()): raise RuntimeError("frozen NVP received gradients")
            opt.step()
            if step%100==0 or step==int(cfg["training"]["total_steps"]):
                metric=evaluate(model,val_x,val_y,records,dict(matrix),int(cfg["training"]["evaluation_batch_size"])); metric.update({"step":step,"train_loss":float(loss.detach().cpu()),"lr":lr}); curve.append(metric)
                print(json.dumps({"architecture":"CrossCandidateMambaNVP","seed":seed,**metric},sort_keys=True),flush=True)
                if best is None or metric["ValidationFinalMeanOverOz"]>best["ValidationFinalMeanOverOz"]:
                    best=metric; torch.save({"stage":"Route-A Cross-Candidate MambaNVP v1","architecture":"CrossCandidateMambaNVP","seed":seed,"step":step,"metrics":metric,"state_dict":model.state_dict(),"nvp_frozen":True,"fusion":cfg["fusion"],"config":copy.deepcopy(dict(cfg))},out/"model.pt")
            if step==int(cfg["training"]["total_steps"]): break
    (out/"learning_curve.json").write_text(json.dumps(curve,indent=2,sort_keys=True)+"\n")
    report={"architecture":"CrossCandidateMambaNVP","seed":seed,"step_execution":"COMPLETE","trainable_parameters":model.trainable_parameter_count(),"selection_metric":"ValidationFinalMeanOverOz policy-45 dataset macro mean","selected":best}
    (out/"experiment_report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n"); return report

def main()->int:
 p=argparse.ArgumentParser(); p.add_argument("--config",type=Path,required=True);p.add_argument("--output-dir",type=Path,required=True);a=p.parse_args()
 if a.output_dir.exists(): raise FileExistsError(a.output_dir)
 if not torch.cuda.is_available(): raise RuntimeError("formal training requires CUDA")
 cfg=load_json(a.config); controlled=load_json(Path(cfg["candidate_representation_source"])); validate(cfg,controlled)
 train,validation=read_jsonl(Path(cfg["target_files"]["train"])),read_jsonl(Path(cfg["target_files"]["validation"])); assert len(train)==28159 and len(validation)==4488
 tx=load_feature_cache(Path(cfg["autophase_feature_cache"]["train"]),"train",[x["program_id"] for x in train]); vx=load_feature_cache(Path(cfg["autophase_feature_cache"]["validation"]),"validation",[x["program_id"] for x in validation])
 tokens,lengths=load_candidates(Path(controlled["candidate_representation"]["candidate_sequences"]),pad_token_id=124,padded_length=20); matrix=read_label_matrix(Path(cfg["validation_label_shards"])); assert set(matrix)=={x["program_id"] for x in validation}
 train_x=torch.tensor([tx[x["program_id"]] for x in train],dtype=torch.float32,device="cuda");train_y=torch.tensor([x["normalized_target"] for x in train],dtype=torch.float32,device="cuda");val_x=torch.tensor([vx[x["program_id"]] for x in validation],dtype=torch.float32,device="cuda");val_y=torch.tensor([x["normalized_target"] for x in validation],dtype=torch.float32,device="cuda")
 a.output_dir.mkdir(parents=True); (a.output_dir/"config.json").write_text(json.dumps(cfg,indent=2,sort_keys=True)+"\n")
 reports=[train_seed(cfg,controlled,s,tokens.cuda(),lengths.cuda(),train_x,train_y,val_x,val_y,validation,matrix,a.output_dir/f"seed{s}") for s in cfg["final_seed_set"]]
 selected=[x["selected"] for x in reports]; cross={"ValidationFinalMeanOverOz_3seed":sum(x["ValidationFinalMeanOverOz"] for x in selected)/3,"oracle_recovery_3seed":sum(x["ValidationFinalMeanOverOz"] for x in selected)/3/0.07743661591867755,"policy45_regret_mean_bytes_3seed":sum(x["policy45_regret_mean_bytes"] for x in selected)/3,"validation_ce_3seed":sum(x["validation_ce"] for x in selected)/3,"seed_results":reports}
 prior=load_json(Path("outputs/route_a_stage_b_v6/comparison_report.json")); mnvp=load_json(Path("outputs/mamba_nvp_objecttext_v6/comparison_report.json"))["comparison"]; report={"step_execution":"COMPLETE","training_only":True,"trajectory_state_available":False,"compiler_gym_initialized":False,"llvm_execution":False,"objecttext_measurements":0,"final_test_accessed":False,"ood_accessed":False,"cross_candidate_mambanvp":cross,"frozen_references":{"NVP":next(x for x in prior["models"] if x["architecture"]=="NVP")["ValidationFinalMeanOverOz_3seed"],"Mamba":next(x for x in prior["models"] if x["architecture"]=="Mamba")["ValidationFinalMeanOverOz_3seed"],"MambaNVP_v1":mnvp["ValidationFinalMeanOverOz_3seed"]}}
 report["differences"]={k:cross["ValidationFinalMeanOverOz_3seed"]-v for k,v in report["frozen_references"].items()}; (a.output_dir/"comparison_report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n"); print(json.dumps(report,indent=2,sort_keys=True));return 0
if __name__=="__main__": raise SystemExit(main())
