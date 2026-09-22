"""Persistent, qualification-gated matched-actor study runner.

Every failed frozen block is retired in full. No individual final run is
retuned. Only predeclared calibration designs are advanced automatically.
"""
import concurrent.futures
import datetime
import hashlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results_mechanism_v6'
PYTHON=sys.executable


def dump(path,value):
    temporary=path.with_suffix(path.suffix+'.tmp');temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)


def status(stage,**kw):
    row=dict(stage=stage,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),pid=os.getpid(),**kw)
    dump(OUT/'persistent_status.json',row);print(json.dumps(row),flush=True)


def execute(command,log):
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MPLBACKEND='Agg',MPLCONFIGDIR='/tmp/mechanism-v6-matplotlib')
    with Path(log).open('w') as stream:return subprocess.call([PYTHON,*map(str,command)],cwd=str(ROOT),env=env,stdout=stream,stderr=subprocess.STDOUT)


def qualify_job(script,configuration,out,seed,condition,actor):
    out=Path(out)
    if not out.exists():execute([script,'--configuration',configuration,'--out',out,'--seed',seed,'--condition',condition,'--actor',actor],out.with_suffix('.log'))
    result=json.loads((out/'qualification.json').read_text()) if (out/'qualification.json').exists() else json.loads((out/'status.json').read_text())
    if result.get('accepted'):
        audit_script=Path(script).with_name('audit_analytic_intervals.py')
        code=execute([audit_script,out],out.with_suffix('.interval_audit.log'))
        result['independent_interval_audit_passed']=code==0
        result['accepted']=bool(result['accepted'] and code==0)
    return result


def freeze(study,cfg,entry,qualification_dirs):
    from train_analytic_actor import DEFAULTS
    source=study/'source_snapshot';(source/'experiments').mkdir(parents=True)
    scripts=['analytic_cbvf_mechanism.py','analytic_cbvf_batch.py','qualify_analytic_cbvf.py','audit_analytic_intervals.py','train_analytic_actor.py','test_analytic_cbvf_mechanism.py','build_analytic_report.py','run_analytic_study.py']
    for name in scripts:shutil.copyfile(ROOT/'experiments'/name,source/'experiments'/name)
    shutil.copytree(ROOT/'jaxrl5',source/'jaxrl5',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    frozen_cfg=dict(cfg,design_only=False,activation_threshold=1.,residual_scale=.02)
    manifest=dict(configuration=frozen_cfg,training=dict(DEFAULTS),conditions={'aligned':{'vmin':-.2,'beta':.8},'easy':{'vmin':-.175,'beta':.625},'hard':{'vmin':-.15,'beta':.43}},model={'vmax':.6,'vmin':-.2,'beta':.8},geometry={'obstacle_center':[0,0],'obstacle_radius':.5,'robot_radius':.2,'goal':[0,1.8],'goal_radius':.5,'audit_world':[-4,4,-4,4]},reset={'center':[-.9,-3,cfg['initial_theta']],'independent_uniform_half_width':cfg['reset_jitter']},certificate={'B':'hypot(x,y)-0.7, exact discounted safety value on B>=0 for reversible nominal model','Cx':1.,'L':'hypot(gamma+0.6/(r-0.6*Delta),0.6)','epsilon':'L*(Cx+1)*Delta','activation':'B<=1; no release hysteresis','qp':'min ||u-u_nom||^2, u in [-1,1]^2; Psi_model>=epsilon+(xi if CP else 0)','true_dynamics_in_uncal_controller':False,'integration':'exact constant-control unicycle flow and exact arc radial minimum'},qualification_directories=[str(p.resolve()) for p in qualification_dirs],entry=entry,training_runs=5,evaluation_runs=45,episodes_per_evaluation=100,heldout_reference_rollouts_per_condition_seed=100,heldout_deployment_rollouts_per_condition_seed=100,representative='first rollout only, all intervals',standard_deviation='sample standard deviation across five training seeds, ddof=1',limitations=['Finite-grid feasibility plus strict runtime checks is not a global continuous-state feasibility theorem.','Split-conformal coverage refers to uncalibrated-filter reference rollouts; deployment coverage is separately empirical.','Learned policy is a bounded residual on a specified shared reference; this is not a learning-efficiency comparison.'])
    dump(study/'effective_configuration.json',manifest)
    environment=dict(python=sys.version,executable=sys.executable,packages={dist.metadata['Name']:dist.version for dist in importlib.metadata.distributions() if dist.metadata.get('Name')},thread_environment={'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'})
    dump(study/'environment.json',environment)
    hashes={str(p.relative_to(study)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(source.rglob('*')) if p.is_file()}
    hashes['effective_configuration.json']=hashlib.sha256((study/'effective_configuration.json').read_bytes()).hexdigest()
    hashes['environment.json']=hashlib.sha256((study/'environment.json').read_bytes()).hexdigest()
    dump(study/'FREEZE.json',dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),final_data_used=False,final_seeds=entry['final_training_seeds'],sha256=hashes,qualification_passed=True))
    return manifest


def verify_freeze(study):
    values=json.loads((study/'FREEZE.json').read_text())['sha256']
    for name,wanted in values.items():
        if hashlib.sha256((study/name).read_bytes()).hexdigest()!=wanted:raise RuntimeError('Frozen file changed: '+name)


def main():
    protocol=json.loads((OUT/'design_protocol.json').read_text())['persistent_queue']
    actor=OUT/'design_actor830001'/'params_20000.pkl'
    status('waiting_for_design_actor')
    while not (actor.parent/'training_complete.json').exists():time.sleep(5)
    base=json.loads((OUT/'analytic_three_point810005'/'configuration.json').read_text())['configuration']
    for entry in protocol['entries']:
        index=entry['id'];design=OUT/('queue_design_%03d'%index);design.mkdir(exist_ok=True)
        cfg=dict(base,n_calibration=entry['n_calibration'],delta=entry['delta'],design_only=True)
        config=design/'configuration.json'
        if not config.exists():dump(config,dict(configuration=cfg))
        status('qualifying_fresh_design_actor',design=index)
        specs=[('hard',0),('hard',1),('aligned',2),('easy',3)]
        qdirs=[design/('%s_seed%d'%(c,entry['qualification_seed_base']+offset)) for c,offset in specs]
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            futures=[pool.submit(qualify_job,ROOT/'experiments/qualify_analytic_cbvf.py',config,path,entry['qualification_seed_base']+offset,c,actor) for (c,offset),path in zip(specs,qdirs)]
            qualification=[f.result() for f in futures]
        dump(design/'qualification_gate.json',dict(passed=all(r.get('accepted',False) for r in qualification),results=qualification))
        if not all(r.get('accepted',False) for r in qualification):
            status('design_rejected_preserved_advancing',design=index);continue
        study=OUT/('study_%03d'%index);study.mkdir(exist_ok=True)
        if not (study/'FREEZE.json').exists():freeze(study,cfg,entry,qdirs)
        verify_freeze(study);script=study/'source_snapshot/experiments';frozen_config=study/'effective_configuration.json'
        status('frozen_training_five_shared_actors',design=index,study=str(study))
        actors=study/'actors';actors.mkdir(exist_ok=True)
        def train_one(seed):
            dest=actors/('seed_%d'%seed)
            if not (dest/'training_complete.json').exists():
                if dest.exists():raise RuntimeError('Incomplete final training retained; cannot overwrite '+str(dest))
                code=execute([script/'train_analytic_actor.py','--configuration',frozen_config,'--out',dest,'--seed',seed],actors/('seed_%d.log'%seed))
                if code:raise RuntimeError('Final training failed; preserve and diagnose '+str(dest))
            return dest/'params_20000.pkl'
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:checkpoints=list(pool.map(train_one,entry['final_training_seeds']))
        verify_freeze(study);status('running_45_matched_final_evaluations',design=index,study=str(study))
        runs=study/'runs';runs.mkdir(exist_ok=True);jobs=[]
        for seed,checkpoint in zip(entry['final_training_seeds'],checkpoints):
            for condition_index,condition in enumerate(['aligned','easy','hard']):
                evaluation_seed=seed+200000+condition_index*100000
                jobs.append((runs/('%s_seed%d'%(condition,seed)),evaluation_seed,condition,checkpoint))
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            futures=[pool.submit(qualify_job,script/'qualify_analytic_cbvf.py',frozen_config,path,seed,condition,checkpoint) for path,seed,condition,checkpoint in jobs]
            results=[f.result() for f in futures]
        verify_freeze(study);passed=all(r.get('accepted',False) for r in results)
        dump(study/'final_gate.json',dict(passed=passed,condition_seed_blocks=15,evaluation_runs=45,results=results))
        if not passed:
            dump(study/'RETIRED.json',dict(reason='Frozen final acceptance failed; entire seed block retired, no per-run tuning or replacement',seeds=entry['final_training_seeds']))
            status('frozen_study_rejected_preserved_advancing',design=index,study=str(study));continue
        status('building_paper_artifacts_and_scientific_acceptance',design=index,study=str(study))
        code=execute([script/'build_analytic_report.py',study],study/'build_report.log')
        verify_freeze(study)
        if code:raise RuntimeError('Paper artifact or acceptance checks failed; diagnose preserved '+str(study))
        acceptance=json.loads((study/'SCIENTIFIC_ACCEPTANCE.json').read_text())
        if not acceptance['passed']:raise RuntimeError('Scientific acceptance did not pass.')
        dump(OUT/'ACCEPTED_STUDY.json',dict(study=str(study.resolve()),report=str((study/'PAPER_HANDOFF.md').resolve()),passed=True))
        status('accepted_paper_ready',design=index,study=str(study));return
    status('predeclared_queue_exhausted_requires_new_mechanism_diagnosis')


if __name__=='__main__':
    try:main()
    except Exception as exc:status('requires_diagnosis',error=str(exc));raise
