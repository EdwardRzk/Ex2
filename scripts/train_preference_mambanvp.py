#!/usr/bin/env python3
"""Train PreferenceAwareMambaNVP using frozen offline targets only."""
from __future__ import annotations
import argparse, copy, json, random
from pathlib import Path
from typing import Any, Mapping, Sequence
import numpy as np
import torch
from torch import nn
from mamba_ssm import Mamba

if __package__:
    from scripts.train_set_conditioned_mamba_ranker import (AUTOPHASE_DIM, K, CandidateInterface, learning_rate, load_candidates, load_feature_cache, load_json, policy_metrics, read_jsonl, read_label_matrix, seed_everything)
    from scripts.train_controlled_nvp_stage_a import soft_cross_entropy
else:
    from train_set_conditioned_mamba_ranker import (AUTOPHASE_DIM, K, CandidateInterface, learning_rate, load_candidates, load_feature_cache, load_json, policy_metrics, read_jsonl, read_label_matrix, seed_everything)
    from train_controlled_nvp_stage_a import soft_cross_entropy

METHOD = "PreferenceAwareMambaNVP"

class PreferenceAwareMambaNVP(CandidateInterface):
    def __init__(self, cfg: Mapping[str, Any], tokens: torch.Tensor, lengths: torch.Tensor) -> None:
        super().__init__(cfg, tokens, lengths); d=self.d_model
        self.block_norms=nn.ModuleList([nn.LayerNorm(d) for _ in range(int(cfg['layers']))])
        self.blocks=nn.ModuleList([Mamba(d_model=d,d_state=int(cfg['d_state']),d_conv=int(cfg['d_conv']),expand=int(cfg['expand']),use_fast_path=bool(cfg['use_fast_path']),layer_idx=i) for i in range(int(cfg['layers']))])
        self.output_norm=nn.LayerNorm(d); self.value_head=nn.Linear(d,1)
        self.preference_head=nn.Sequential(nn.Linear(d,d),nn.ReLU(),nn.Linear(d,1))
    def embeddings(self, program: torch.Tensor) -> torch.Tensor:
        hidden,lengths=self.candidate_inputs(program)
        for norm,block in zip(self.block_norms,self.blocks): hidden=hidden+block(norm(hidden))
        rows=torch.arange(len(hidden),device=hidden.device)
        return self.output_norm(hidden[rows,lengths-1]).reshape(program.shape[0],K,-1)
    def forward(self, program: torch.Tensor) -> torch.Tensor:
        return self.value_head(self.embeddings(program)).squeeze(-1)
    def preference_logits(self, embeddings: torch.Tensor, first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
        batch=torch.arange(len(embeddings),device=embeddings.device)[:,None]
        return self.preference_head(embeddings[batch,first]-embeddings[batch,second]).squeeze(-1)
    def trainable_parameter_count(self) -> int: return sum(p.numel() for p in self.parameters() if p.requires_grad)

def strict_eligible(values: torch.Tensor) -> torch.Tensor:
    return (values.max(dim=1).values != values.min(dim=1).values)

def sample_balanced_pairs(values: torch.Tensor, eligible: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Five strict pairs become canonical positive plus reversed negative samples."""
    batch=values.shape[0]; first=torch.zeros(batch,10,dtype=torch.long,device=values.device); second=first.clone(); active=eligible.clone()
    for slot in range(5):
        left=torch.randint(K,(batch,),device=values.device); right=torch.randint(K,(batch,),device=values.device)
        bad=active & ((left==right) | (values[torch.arange(batch,device=values.device),left] == values[torch.arange(batch,device=values.device),right]))
        while bool(bad.any()):
            count=int(bad.sum()); left[bad]=torch.randint(K,(count,),device=values.device); right[bad]=torch.randint(K,(count,),device=values.device)
            bad=active & ((left==right) | (values[torch.arange(batch,device=values.device),left] == values[torch.arange(batch,device=values.device),right]))
        winner=torch.where(values[torch.arange(batch,device=values.device),left] > values[torch.arange(batch,device=values.device),right],left,right); loser=torch.where(winner==left,right,left)
        first[:,slot]=winner; second[:,slot]=loser; first[:,slot+5]=loser; second[:,slot+5]=winner
    labels=torch.cat([torch.ones(batch,5,device=values.device),torch.zeros(batch,5,device=values.device)],dim=1)
    return first,second,labels,active

def preference_loss_and_stats(model: PreferenceAwareMambaNVP, embeddings: torch.Tensor, values: torch.Tensor) -> tuple[torch.Tensor, dict[str,int]]:
    eligible=strict_eligible(values)
    if not bool(eligible.any()): return embeddings.new_zeros(()), {'total_pairs':0,'positive_pairs':0,'negative_pairs':0,'eligible_program_count':0,'skipped_tie_program_count':len(values)}
    first,second,labels,active=sample_balanced_pairs(values,eligible)
    logits=model.preference_logits(embeddings,first,second)
    loss=nn.functional.binary_cross_entropy_with_logits(logits[active],labels[active])
    n=int(active.sum()); return loss, {'total_pairs':n*10,'positive_pairs':n*5,'negative_pairs':n*5,'eligible_program_count':n,'skipped_tie_program_count':len(values)-n}

def diagnostics(model: PreferenceAwareMambaNVP, features: torch.Tensor, values: torch.Tensor, batch_size: int) -> dict[str,Any]:
    total_ce=correct_pref=total_pref=correct_value=total_value=0; logits_parts=[]
    model.eval()
    with torch.no_grad():
        for start in range(0,len(features),batch_size):
            x,v=features[start:start+batch_size],values[start:start+batch_size]; h=model.embeddings(x); scores=model.value_head(h).squeeze(-1); logits_parts.append(scores.cpu()); total_ce+=float(soft_cross_entropy(scores, torch.softmax(v/0.05,dim=1)).cpu())*len(x)
            for i in range(K):
                for j in range(i+1,K):
                    strict=v[:,i]!=v[:,j]
                    if bool(strict.any()):
                        winner=torch.where(v[:,i]>v[:,j],torch.full_like(v[:,i],i,dtype=torch.long),torch.full_like(v[:,i],j,dtype=torch.long)); loser=torch.where(winner==i,torch.full_like(winner,j),torch.full_like(winner,i)); idx=torch.arange(len(x),device=x.device); pref=model.preference_head(h[idx,winner]-h[idx,loser]).squeeze(-1)
                        correct_pref+=int((pref[strict]>0).sum()); correct_value+=int((scores[idx,winner][strict]>scores[idx,loser][strict]).sum()); total_pref+=int(strict.sum()); total_value+=int(strict.sum())
    return {'logits':torch.cat(logits_parts),'validation_nvp_ce':total_ce/len(features),'preference_accuracy':correct_pref/total_pref if total_pref else None,'pairwise_accuracy':correct_value/total_value if total_value else None,'validation_strict_pair_count':total_pref}

def validate_config(cfg, controlled):
    assert cfg['final_seed_set']==[1,2,3] and cfg['frozen_data_population']=={'train_complete_k50':28159,'validation_complete_k50':4488}
    assert cfg['target_and_objective']['target_temperature']==0.05 and cfg['target_and_objective']['lambda_preference']==0.1
    assert cfg['pair_sampling']['pairs_per_eligible_program_per_epoch']==10
    assert cfg['training']['total_steps']==10000 and not cfg['training']['early_stopping'] and cfg['training']['checkpoint_evaluation_cadence_steps']==100
    assert controlled['candidate_representation']['K']==K and cfg['validation']['sampling'] is False and cfg['validation']['scored_pass_budget']==45

def evaluate(model, val_x, val_y, val_values, records, matrix, batch_size):
    d=diagnostics(model,val_x,val_values,batch_size); metrics=policy_metrics(d.pop('logits'),records,matrix); metrics.update(d); return metrics

def references(cfg):
    stage=load_json(Path(cfg['frozen_reference_reports']['stage_b'])); models={x['architecture']:x for x in stage['models']}; mnvp=load_json(Path(cfg['frozen_reference_reports']['mamba_nvp_v1']))['comparison']; cross=load_json(Path(cfg['frozen_reference_reports']['cross_candidate_mambanvp']))['cross_candidate_mambanvp']
    return {'NVP':models['NVP']['ValidationFinalMeanOverOz_3seed'],'Mamba':models['Mamba']['ValidationFinalMeanOverOz_3seed'],'MambaNVP_v1':mnvp['ValidationFinalMeanOverOz_3seed'],'CrossCandidateMambaNVP':cross['ValidationFinalMeanOverOz_3seed']}

def train_seed(cfg, controlled, seed, tokens, lengths, train_x, train_y, train_values, val_x, val_y, val_values, validation, matrix, out):
    seed_everything(seed); model_cfg={**controlled['candidate_representation'],**controlled['models']['Mamba']}; model=PreferenceAwareMambaNVP(model_cfg,tokens,lengths).cuda(); opt=torch.optim.Adam(model.parameters(),lr=float(cfg['training']['learning_rate']),weight_decay=float(cfg['training']['weight_decay'])); rng=torch.Generator().manual_seed(seed); out.mkdir(parents=True); curve=[]; pair_total=collections.Counter(); best=None; step=0
    while step<int(cfg['training']['total_steps']):
        order=torch.randperm(len(train_x),generator=rng)
        for begin in range(0,len(train_x),int(cfg['training']['batch_size'])):
            step+=1; idx=order[begin:begin+int(cfg['training']['batch_size'])].cuda(non_blocking=True); lr=learning_rate(cfg['training'],step)
            for group in opt.param_groups: group['lr']=lr
            model.train(); embeddings=model.embeddings(train_x[idx]); scores=model.value_head(embeddings).squeeze(-1); value_loss=soft_cross_entropy(scores,train_y[idx]); pref_loss,stats=preference_loss_and_stats(model,embeddings,train_values[idx]); loss=value_loss+0.1*pref_loss; opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); pair_total.update(stats)
            if step%int(cfg['training']['checkpoint_evaluation_cadence_steps'])==0 or step==int(cfg['training']['total_steps']):
                metric=evaluate(model,val_x,val_y,val_values,validation,matrix,int(cfg['training']['evaluation_batch_size'])); metric.update({'step':step,'train_total_loss':float(loss.detach().cpu()),'train_value_loss':float(value_loss.detach().cpu()),'train_preference_loss':float(pref_loss.detach().cpu()),'lr':lr}); curve.append(metric); print(json.dumps({'architecture':METHOD,'seed':seed,**metric},sort_keys=True),flush=True)
                if best is None or metric['ValidationFinalMeanOverOz']>best['ValidationFinalMeanOverOz']:
                    best=metric; torch.save({'stage':'Route-A Preference-aware MambaNVP v1','architecture':METHOD,'seed':seed,'step':step,'metrics':metric,'state_dict':model.state_dict(),'model_config':model_cfg,'pair_sampling':copy.deepcopy(dict(cfg['pair_sampling'])),'lambda_preference':0.1},out/'model.pt')
            if step==int(cfg['training']['total_steps']): break
    report={'architecture':METHOD,'seed':seed,'step_execution':'COMPLETE','trainable_parameters':model.trainable_parameter_count(),'selection_metric':'ValidationFinalMeanOverOz policy-45 dataset macro mean','selected':best}; return report,curve,dict(pair_total)

def main():
    import collections
    globals()['collections']=collections
    p=argparse.ArgumentParser(); p.add_argument('--config',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--seeds',type=int,nargs='+',default=None);a=p.parse_args()
    if a.output_dir.exists(): raise FileExistsError(a.output_dir)
    if not torch.cuda.is_available(): raise RuntimeError('formal training requires CUDA')
    cfg=load_json(a.config); controlled=load_json(Path(cfg['candidate_representation_source'])); validate_config(cfg,controlled)
    train,validation=read_jsonl(Path(cfg['target_files']['train'])),read_jsonl(Path(cfg['target_files']['validation'])); assert len(train)==28159 and len(validation)==4488
    if any(len(x['raw_candidate_value'])!=K or len(x['normalized_target'])!=K for x in train+validation): raise ValueError('missing frozen target fields')
    tokens,lengths=load_candidates(Path(controlled['candidate_representation']['candidate_sequences']),pad_token_id=124,padded_length=20); matrix=read_label_matrix(Path(cfg['validation_label_shards'])); assert set(matrix)=={x['program_id'] for x in validation}
    tx=load_feature_cache(Path(cfg['autophase_feature_cache']['train']),'train',[x['program_id'] for x in train]); vx=load_feature_cache(Path(cfg['autophase_feature_cache']['validation']),'validation',[x['program_id'] for x in validation])
    train_x=torch.tensor([tx[x['program_id']] for x in train],dtype=torch.float32,device='cuda'); train_y=torch.tensor([x['normalized_target'] for x in train],dtype=torch.float32,device='cuda'); train_values=torch.tensor([x['raw_candidate_value'] for x in train],dtype=torch.float32,device='cuda'); val_x=torch.tensor([vx[x['program_id']] for x in validation],dtype=torch.float32,device='cuda'); val_y=torch.tensor([x['normalized_target'] for x in validation],dtype=torch.float32,device='cuda'); val_values=torch.tensor([x['raw_candidate_value'] for x in validation],dtype=torch.float32,device='cuda')
    selected_seeds=cfg["final_seed_set"] if a.seeds is None else a.seeds
    if not selected_seeds or any(s not in cfg["final_seed_set"] for s in selected_seeds) or len(set(selected_seeds)) != len(selected_seeds):
        raise ValueError("--seeds must be a non-empty, duplicate-free subset of final_seed_set")
    a.output_dir.mkdir(parents=True); (a.output_dir/"config.json").write_text(json.dumps(cfg,indent=2,sort_keys=True)+"\n"); reports=[];curves={};pairs={}
    for s in selected_seeds:
        r,c,ps=train_seed(cfg,controlled,s,tokens.cuda(),lengths.cuda(),train_x,train_y,train_values,val_x,val_y,val_values,validation,matrix,a.output_dir/"checkpoints"/f"seed{s}"); reports.append(r);curves[str(s)]=c;pairs[str(s)]=ps
    (a.output_dir/"training_curve.json").write_text(json.dumps(curves,indent=2,sort_keys=True)+"\n");(a.output_dir/"pair_statistics.json").write_text(json.dumps({"pair_sampling":cfg["pair_sampling"],"per_seed":pairs},indent=2,sort_keys=True)+"\n")
    mean=sum(r["selected"]["ValidationFinalMeanOverOz"] for r in reports)/len(reports); oracle=load_json(Path(cfg["frozen_reference_reports"]["stage_b"]))["fixed_route_a_oracle"]; pref={"architecture":METHOD,"ValidationFinalMeanOverOz_mean_executed_seeds":mean,"oracle_recovery_mean_executed_seeds":mean/oracle,"policy45_regret_mean_bytes_executed_seeds":sum(r["selected"]["policy45_regret_mean_bytes"] for r in reports)/len(reports),"preference_accuracy_mean_executed_seeds":sum(r["selected"]["preference_accuracy"] for r in reports)/len(reports),"pairwise_accuracy_mean_executed_seeds":sum(r["selected"]["pairwise_accuracy"] for r in reports)/len(reports),"validation_nvp_ce_mean_executed_seeds":sum(r["selected"]["validation_nvp_ce"] for r in reports)/len(reports),"trainable_parameters":reports[0]["trainable_parameters"],"seed_results":reports}; refs=references(cfg); report={"step_execution":"COMPLETE","training_only":True,"executed_seeds":selected_seeds,"full_three_seed_aggregate":selected_seeds==cfg["final_seed_set"],"compiler_gym_initialized":False,"llvm_execution":False,"candidate_rollouts":0,"objecttext_measurements":0,"label_regeneration":False,"final_test_accessed":False,"ood_accessed":False,"runtime_accessed":False,"validation_cohort":{"N_total":4488,"N_primary_valid":4488,"N_failed_or_invalid":0},"tie_handling":"all-tie programs retain L_value and contribute L_preference=0; they remain in train and validation","preference_aware_mambanvp":pref,"frozen_references":refs,"differences":{k:mean-v for k,v in refs.items()}}; (a.output_dir/"comparison_report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n");(a.output_dir/"experiment_report.json").write_text(json.dumps({"step_execution":"COMPLETE","executed_seeds":selected_seeds,"seeds":reports,"pair_statistics_file":"pair_statistics.json"},indent=2,sort_keys=True)+"\n");print(json.dumps(report,indent=2,sort_keys=True))
if __name__=='__main__': main()
