#!/usr/bin/env python3
"""Record and run one immutable design attempt with an exclusive output lock."""
import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import shutil
import time

ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser()
    p.add_argument('name')
    p.add_argument('--seed',type=int,required=True)
    p.add_argument('--gamma',type=float,default=.25)
    p.add_argument('--beta',type=float,default=.6)
    p.add_argument('--speed-min',type=float,default=0.)
    p.add_argument('--activation',type=float,default=.03)
    p.add_argument('--initial-x',type=float,default=-1.45)
    p.add_argument('--duration',type=float,default=14.)
    p.add_argument('--dt',type=float,default=.001)
    p.add_argument('--waypoint-x',type=float,default=-.65)
    p.add_argument('--waypoint-y',type=float,default=.65)
    p.add_argument('--orbit-trigger-radius',type=float,default=0.)
    p.add_argument('--orbit-radius',type=float,default=1.)
    p.add_argument('--orbit-gain',type=float,default=2.)
    p.add_argument('--goal-x',type=float,default=2.)
    p.add_argument('--goal-y',type=float,default=2.)
    p.add_argument('--pilot',action='store_true')
    p.add_argument('--steps',type=int,default=20)
    p.add_argument('--episodes',type=int,default=100)
    p.add_argument('--method',choices=['nominal','cbvf','cp'],default='cp')
    p.add_argument('--initial-theta',type=float,default=0.7853981633974483)
    p.add_argument('--clearance-edges',default='0.75')
    p.add_argument('--n-calib',type=int,default=40)
    p.add_argument('--results-root',default='results_intersample_v5')
    a=p.parse_args()
    root=ROOT/a.results_root
    root.mkdir(exist_ok=True)
    out=root/a.name
    out.mkdir(exist_ok=False)
    lock=(out/'run.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    base=json.loads((ROOT/'experiments/full_suite_intersample.json').read_text())
    cfg=dict(base['common'])
    cfg.update(gamma=a.gamma,seed=a.seed,speed_min=a.speed_min,speed=.6,beta_u=a.beta,
        cbvf_model_speed_min=-.2,cbvf_model_speed=.6,cbvf_model_beta_u=.8,
        dt=a.dt,horizon=round(a.duration/a.dt),initial_x=a.initial_x,
        goal_x=a.goal_x,goal_y=a.goal_y,reference_waypoint_x=a.waypoint_x,reference_waypoint_y=a.waypoint_y,
        reference_orbit_trigger_radius=a.orbit_trigger_radius,
        reference_orbit_radius=a.orbit_radius,reference_orbit_gain=a.orbit_gain,
        initial_theta=a.initial_theta,
        mondrian_clearance_edges=a.clearance_edges,n_calib=a.n_calib,
        cbvf_activate_margin=a.activation,cp_activate_margin=a.activation,
        max_steps=a.steps,start_training=min(a.steps,1000),
        reference_proposal_steps=min(a.steps,1000),
        eval_interval=a.steps,save_interval=a.steps,periodic_eval_episodes=5,
        eval_episodes=a.episodes,training_shield_method=a.method,
        cache_dir=str(out/'cbvf_cache'),
        reuse_cbvf=False,reuse_zero_offline_audit=False,recompute_cbvf=True,
        calibration_deterministic_policy=True,feasibility_only=not a.pilot,
        checkpoint_dir=str(out),results_json=str(out/'training_and_evaluation_results.json'),
        eval_jsonl=str(out/'training_eval_metrics.jsonl'),
        action_log_jsonl=str(out/'training_actions.jsonl'),
        qp_diagnostics_jsonl=str(out/'qp_feasibility_diagnostics.jsonl'),
        plot_dir=str(out/'plots'),odp_root=str(ROOT),solver_file=str(ROOT/'solver_cbvf.py'))
    command=[base['python'],'train/train_sac_lag.py']
    for k,v in cfg.items():
        flag='--'+k.replace('_','-')
        if isinstance(v,bool):
            if v: command.append(flag)
        else: command += [flag,str(v)]
    record={'configuration':cfg,'command':command,'source_hashes':{
        str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest()
        for f in [ROOT/'conformal_shield.py',ROOT/'solver_cbvf.py',ROOT/'train/train_sac_lag.py',ROOT/'redexp/envs/conformal_dubins_3d_env.py',Path(__file__).resolve()]},
        'started':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    (out/'batch_configuration.json').write_text(json.dumps(record,indent=2)+'\n')
    for relative in record['source_hashes']:
        destination=out/'source_snapshot'/relative
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/relative,destination)
    env=dict(os.environ,MPLCONFIGDIR='/tmp/mondrian_matplotlib',PYTHONUNBUFFERED='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    with (out/'run.log').open('w') as log:
        process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        while process.poll() is None:
            if (out/'STOP_REQUESTED.json').exists():
                process.terminate()
                try: process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill(); process.wait()
                break
            time.sleep(1)
        returncode=process.returncode
    (out/'exit_status.json').write_text(json.dumps({'returncode':returncode,'finished':datetime.datetime.now(datetime.timezone.utc).isoformat()},indent=2)+'\n')
    return returncode

if __name__=='__main__':
    raise SystemExit(main())
