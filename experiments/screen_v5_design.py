#!/usr/bin/env python3
"""Coarse design-only baseline screen; never a certification or final result."""
import concurrent.futures
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import conformal_shield as cs
SCREEN_DT=.005
DURATION=12.
TABLE_DIR=ROOT/'results_intersample_v5/solver_reproducibility/0'
TABLE_PATTERN='*.npz'
GOAL=(2.,2.)
GAMMA=.25
WAYPOINT=(-.65,.65)


def reference(state, *unused):
    target = WAYPOINT if state[1] < 0.45 else GOAL
    desired = math.atan2(target[1]-state[1], target[0]-state[0])
    error = (desired-state[2]+math.pi) % (2*math.pi)-math.pi
    return np.array([1., np.clip(1.5*error/0.8, -1., 1.)])


def screen(job):
    global WAYPOINT
    if len(job)>5: WAYPOINT=(job[5],.65)
    beta, speed_min, activation, initial_x = job[:4]
    initial_theta=job[4] if len(job)>4 else math.pi/4
    table_path = next(TABLE_DIR.glob(TABLE_PATTERN))
    table = cs.CBVFTable.from_npz(table_path, max_abs_table=10., max_abs_grad=200.)
    table.enable_continuous_time_interpolation(True)
    variant = cs.DynamicsVariant('design', cs.DynamicsSpec(.6,.8,-.2), cs.DynamicsSpec(.6,beta,speed_min))
    rng = np.random.default_rng(610001)
    starts = [np.array([initial_x,-3.,initial_theta])+rng.uniform(-.1,.1,3) for _ in range(10)]
    rows = {}
    for method in ('nominal','cbvf'):
        logs = []
        for start in starts:
            horizon=round(DURATION/SCREEN_DT)
            env = cs.DubinsCBVFEnv(cs.World(goal=GOAL),cs.Interval(-1.,1.),variant,SCREEN_DT,horizon)
            env.set_state(start)
            minimum = env.safety_margin_value(start)
            infeasible = 0
            for step in range(horizon):
                s=env.state.copy(); u=reference(s); t=-(DURATION+5.)+step*SCREEN_DT
                if method == 'cbvf' and table.value_grad_dt(s,t)[0] <= activation:
                    projection=cs.solve_cbvf_projection_exact(s,t,u,table,env,GAMMA,0.)
                    u=projection.u; infeasible += int(not projection.feasible)
                s,done,info=env.step(u)
                minimum=min(minimum,env.safety_margin_value(s))
                if done:
                    break
            logs.append([info['goal'],info['unsafe'],minimum,infeasible])
        rows[method] = dict(zip(('goal_rate','unsafe_rate','mean_min_l','mean_infeasible_steps'),np.mean(logs,axis=0)))
    return {'waypoint':WAYPOINT,'gamma':GAMMA,'truth_beta': beta, 'truth_speed_min': speed_min, 'activation': activation,
            'initial_x': initial_x, 'release_margin': 0., 'screen_dt': SCREEN_DT,
            'initial_theta':initial_theta,'goal':GOAL,
            'duration':DURATION, 'table':str(table_path), 'episodes': 10,
            'seed': 610001, 'certification': False, 'methods': rows}


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--round',type=int,default=2)
    args=parser.parse_args()
    out = ROOT/('results_intersample_v5/screen_round_'+str(args.round))
    out.mkdir(parents=True, exist_ok=False)
    jobs = [(b,0.,.03,-1.45) for b in [.60,.59,.58,.56,.54]] if args.round==1 else [
        (.6,v,.03,-1.45) for v in [.05,.10,.15,.20]] + [(.6,0.,.023,-1.45),(.56,0.,.023,-1.45)]
    if args.round in (3,4):
        SCREEN_DT=.002
        DURATION=16.
        TABLE_DIR=ROOT/'cbvf_cache_intersample_design/v3'
        TABLE_PATTERN='*lb21.0*.npz'
        jobs=[(b,0.,.03,x) for b in (.54,.56) for x in (-1.45,-1.4,-1.35)]
        if args.round==4:
            jobs=[(b,0.,.03,x) for b in (.58,.60) for x in (-1.4,-1.35,-1.3)]
    if args.round==5:
        SCREEN_DT=.002
        GOAL=(1.5,1.5)
        jobs=[(.6,0.,.04,x,h) for x in (-1.35,-1.3) for h in (math.pi/4,.9,1.)]
    if args.round==6:
        SCREEN_DT=.002
        GOAL=(1.5,1.5)
        jobs=[(b,0.,1.,x) for b in (.54,.6) for x in (-1.45,-1.35,-1.25)]
    if args.round==7:
        SCREEN_DT=.005
        GOAL=(1.5,1.5)
        GAMMA=2.
        TABLE_DIR=ROOT/'results_intersample_v5/equation_correct_gamma2_seed610502/cbvf_cache'
        jobs=[(b,v,1.,-1.35) for b in (.54,.6) for v in (-.1,-.05,0.)]
    if args.round==8:
        SCREEN_DT=.005
        GOAL=(1.5,1.5)
        GAMMA=2.
        TABLE_DIR=ROOT/'results_intersample_v5/equation_correct_gamma2_seed610502/cbvf_cache'
        jobs=[(b,-.1,1.,x) for b in (.50,.54) for x in (-1.35,-1.30,-1.25)]
    if args.round==9:
        SCREEN_DT=.005
        GOAL=(1.5,1.5)
        GAMMA=1.5
        TABLE_DIR=ROOT/'results_intersample_v5/gamma15_beta054_reverse010_dt001_x130_seed610508/cbvf_cache'
        jobs=[(b,-.1,1.,x,math.pi/4,wp) for b in (.45,.5,.54) for x in (-1.3,-1.1) for wp in (-.85,-.95)]
    (out/'manifest.json').write_text(json.dumps({'jobs':jobs,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},indent=2))
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as pool:
        for i,row in enumerate(pool.map(screen, jobs)):
            path = out/('candidate_'+str(i)+'.json')
            path.write_text(json.dumps(row,indent=2,allow_nan=False)+'\n')
            print(json.dumps(row), flush=True)
