"""Fresh calibration, paired learned-actor qualification, and analytic audits.

No final seed allocation or final training is performed by this entry point.
Every output directory is new and immutable on rerun.
"""
import argparse
import hashlib
import json
import math
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import pickle
import shutil
import sys
import traceback
from pathlib import Path
import numpy as np
from analytic_cbvf_batch import evaluate,TRACE_HEADER


def dump(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False,default=lambda x:x.item())+'\n')


class FrozenResidual:
    def __init__(self,path):
        self.path=Path(path);self.sha256=hashlib.sha256(self.path.read_bytes()).hexdigest()
        params=pickle.loads(self.path.read_bytes())['agent']['actor']['params']
        self.layers=[{k:np.asarray(v) for k,v in params['MLP_0'][name].items()} for name in sorted(params['MLP_0'],key=lambda x:int(x.split('_')[-1]))]
        self.output={k:np.asarray(v) for k,v in params['OutputDenseMean'].items()}

    def __call__(self,s):
        x=np.c_[s[:,:2],np.cos(s[:,2]),np.sin(s[:,2]),np.full(len(s),0.),np.full(len(s),1.8)].astype(np.float32)
        for layer in self.layers:x=np.maximum(x@layer['kernel']+layer['bias'],0)
        return np.tanh(x@self.output['kernel']+self.output['bias'])


def audit(cfg,buffers):
    xy=np.linspace(-4,4,31);theta=np.linspace(-math.pi,math.pi,25,endpoint=False)
    x,y,th=np.meshgrid(xy,xy,theta,indexing='ij');r=np.hypot(x,y);h=r-.7
    use=(h>0)&(h<=cfg.get('activation_threshold',1.));x=x[use];y=y[use];th=th[use];r=r[use];h=h[use]
    c=(x*np.cos(th)+y*np.sin(th))/r;L=np.hypot(cfg['gamma']+.6/(r-.6*cfg['dt']),.6);eps=2*L*cfg['dt']
    regions=np.c_[h-cfg['dt']<=.75,h+cfg['dt']>=.75];xi=np.max(np.where(regions,np.array(buffers)[None,:],-np.inf),axis=1)
    jitter=cfg['reset_jitter'];initial_min=math.hypot(.9-jitter,3-jitter)-.7
    first=np.maximum(0,np.ceil(np.log(initial_min/h)/(cfg['gamma']*cfg['dt'])).astype(int))
    multiplicity=np.maximum(0,round(cfg['duration']/cfg['dt'])-first)
    valid=multiplicity>0;out={}
    for method,b in [('cbvf',0),('cp',xi)]:
        margin=cfg['gamma']*h+.2*c+.4*np.abs(c)-eps-b
        out[method]=dict(unique_nodes=int(valid.sum()),node_interval_checks=int(multiplicity.sum()),infeasible_nodes=int(np.sum(valid&(margin<0))),infeasible_node_intervals=int(np.sum(multiplicity[margin<0])),minimum_margin=float(np.min(margin[valid])))
    return dict(domain='31 x 31 x 25 Cartesian/periodic grid; 0<B<=activation and B>=h_initial_min exp(-gamma*t), every control interval; stationary margins evaluated once per unique node and exact interval multiplicities counted',continuous_state_feasibility_claim=False,initial_min_clearance=initial_min,intervals=round(cfg['duration']/cfg['dt']),Cx=1.,L_min=float(L.min()),L_max=float(L.max()),epsilon_min=float(eps.min()),epsilon_max=float(eps.max()),methods=out,passed=all(v['infeasible_nodes']==0 and v['minimum_margin']>0 for v in out.values()))


def run(out,cfg,seed,condition,actor_path,episodes=100):
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    for filename in ['analytic_cbvf_batch.py','analytic_cbvf_mechanism.py','qualify_analytic_cbvf.py']:
        shutil.copyfile(Path(__file__).parent/filename,out/filename)
    frozen=FrozenResidual(actor_path)
    effective=dict(cfg);effective.update(residual_scale=.02,activation_threshold=1.)
    effective.update({'hard':dict(beta=.43,vmin=-.15),'easy':dict(beta=.625,vmin=-.175),'aligned':dict(beta=.8,vmin=-.2)}[condition])
    ncal=int(effective.get('n_calibration',78));delta=float(effective.get('delta',.05));order=math.ceil((ncal+1)*(1-delta/2))
    if not 1<=order<=ncal:raise ValueError('Finite split-conformal quantile unavailable.')
    metadata=dict(configuration=effective,seed=seed,condition=condition,actor=str(Path(actor_path).resolve()),actor_sha256=frozen.sha256,primary_comparison='same deterministic frozen actor and same initial states across all methods',design_only=bool(effective.get('design_only',True)),n_calibration=ncal,delta=delta,regional_delta=delta/2,regional_quantile_order=order,performance_episodes=episodes,coverage_episodes=100,reset_rng_seeds=dict(calibration=seed,performance=seed+10000,coverage=seed+20000))
    dump(out/'configuration.json',metadata)
    center=np.array([-.9,-3,effective['initial_theta']]);jitter=effective['reset_jitter']
    def starts(offset,n):return center+np.random.default_rng(seed+offset).uniform(-jitter,jitter,(n,3))
    allstarts={'calibration':starts(0,ncal),'performance':starts(10000,episodes),'coverage':starts(20000,100)}
    combined=np.concatenate(list(allstarts.values()))
    if len(np.unique(combined,axis=0))!=len(combined):raise RuntimeError('Calibration/evaluation overlap.')
    np.savez(out/'initial_states.npz',**allstarts)
    def stage(name):dump(out/'status.json',dict(stage=name,accepted=False));print(condition,seed,name,flush=True)
    def save_eval(name,result):
        trace=result.pop('trace');np.savetxt(out/(name+'_intervals.csv'),trace,delimiter=',',header=TRACE_HEADER,comments='');dump(out/(name+'.json'),result)
        return result
    try:
        stage('calibration')
        cal=save_eval('calibration',evaluate(allstarts['calibration'],effective,'cbvf',residual=frozen))
        if any(min(r['region_visits'])==0 for r in cal['rollouts']):raise RuntimeError('A calibration region lacks a visited rollout; no silent zero score substitution.')
        buffers=np.sort([r['scores'] for r in cal['rollouts']],axis=0)[order-1]
        dump(out/'buffers.json',dict(buffers=buffers.tolist(),regions=['h<0.75','h>=0.75'],n=[ncal,ncal],delta=[delta/2,delta/2],order=[order,order]))
        stage('fresh_exhaustive_audit');feas=audit(effective,buffers);dump(out/'feasibility_audit.json',feas)
        if not feas['passed']:raise RuntimeError('Fresh exhaustive grid audit failed.')
        performance={}
        for method in ['nominal','cbvf','cp']:
            stage('paired_performance_'+method)
            performance[method]=save_eval('performance_'+method,evaluate(allstarts['performance'],effective,method,buffers,frozen))
        stage('fresh_heldout_reference_coverage')
        heldout=save_eval('heldout_reference',evaluate(allstarts['coverage'],effective,'cbvf',buffers,frozen))
        scores=np.array([r['scores'] for r in heldout['rollouts']]);covered=np.all(scores<=buffers+1e-12,axis=1)
        stage('heldout_deployment_diagnostic')
        deployment=save_eval('heldout_deployment',evaluate(allstarts['coverage'],effective,'cp',buffers,frozen))
        deployment_covered=np.all(np.array([r['scores'] for r in deployment['rollouts']])<=buffers+1e-12,axis=1)
        coverage=dict(reference_covered=int(covered.sum()),reference_total=100,target=1-delta,empirical_acceptance_threshold=.95,deployment_covered=int(deployment_covered.sum()),deployment_total=100,guarantee_population='fresh uncalibrated-filter rollouts of the same frozen actor; deployment coverage separately empirical')
        dump(out/'coverage.json',coverage)
        criteria=dict(exhaustive_feasibility=feas['passed'],positive_runtime_margins=all(r['min_qp']>0 for p in [cal,heldout,deployment,performance['cbvf'],performance['cp']] for r in p['rollouts']),heldout_coverage=int(covered.sum())>=95,nontrivial_buffers=(condition=='aligned' and np.all(buffers==0)) or (condition!='aligned' and np.all(buffers>1e-4)),cp_safety=performance['cp']['unsafe_rate']==0 and deployment['unsafe_rate']==0,cp_goal=performance['cp']['goal_rate']>=.9 and deployment['goal_rate']>=.9,cp_true_interval_certificate=performance['cp']['min_true_interval_margin']>=-1e-10 and deployment['min_true_interval_margin']>=-1e-10,matched_policy=True,exact_integration=True)
        if condition=='hard':criteria['safety_separation']=performance['cbvf']['unsafe_rate']>=.1 and performance['cp']['unsafe_rate']<=.2*performance['cbvf']['unsafe_rate']
        else:criteria['matching_behavior']=all(p['unsafe_rate']==0 and p['goal_rate']>=.9 for p in performance.values())
        result=dict(accepted=all(criteria.values()),criteria=criteria,condition=condition,seed=seed,buffers=buffers.tolist(),coverage=coverage,performance={m:{k:p[k] for k in ['unsafe_rate','goal_rate','min_true_interval_margin']} for m,p in performance.items()})
        dump(out/'qualification.json',result);dump(out/'status.json',dict(stage='completed',**result));print(json.dumps(result,default=lambda x:x.item()),flush=True)
        return result
    except Exception as exc:
        dump(out/'status.json',dict(stage='failed',accepted=False,error=str(exc),traceback=traceback.format_exc()));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--configuration',type=Path,required=True);p.add_argument('--out',required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--condition',choices=['hard','easy','aligned'],required=True);p.add_argument('--actor',required=True);p.add_argument('--episodes',type=int,default=100);a=p.parse_args()
    run(a.out,json.loads(a.configuration.read_text())['configuration'],a.seed,a.condition,a.actor,a.episodes)
