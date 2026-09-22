#!/usr/bin/env python3
"""Recompute and export every interval of each primary CP representative rollout.

Representatives are the first evaluation episode for every fixed seed, not
selected for their outcome. This supplements, and does not replace, exhaustive
grid audits and all-episode aggregate/runtime checks.
"""
import argparse
import csv
import json
import math
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import conformal_shield as cs


def audit_run(folder):
    folder = Path(folder)
    cfg = json.loads((folder/'batch_configuration.json').read_text())['configuration']
    data = json.loads((folder/'training_and_evaluation_results.json').read_text())
    trajectory = data['final_evaluation']['cp']['representative_trajectory']
    table = cs.CBVFTable.from_npz(data['cbvf']['path'], max_abs_table=10., max_abs_grad=200.)
    table.enable_continuous_time_interpolation(True)
    saved = json.loads((folder/'final_frozen_policy_mondrian_calibration.json').read_text())['mondrian_calibration']
    regions = saved['regions']
    partition = cs.MondrianPartition(tuple(saved['clearance_edges']), (0.,0.), .7)
    cal = cs.MondrianCalibration(partition, tuple(tuple(r['base_regions']) for r in regions),
        tuple(saved['base_to_effective']), tuple(r['xi_hat_off'] for r in regions),
        tuple(r['delta_m'] for r in regions), tuple(r['n_scores'] for r in regions),
        saved['delta_traj'], saved['epsilon_grid'], saved['promotion_rounds'])
    variant = cs.DynamicsVariant('recorded',
        cs.DynamicsSpec(cfg['cbvf_model_speed'], cfg['cbvf_model_beta_u'], cfg['cbvf_model_speed_min']),
        cs.DynamicsSpec(cfg['speed'], cfg['beta_u'], cfg['speed_min']))
    dt = cfg['dt']; speed = max(abs(cfg['speed']),abs(cfg['speed_min'])); omega = abs(cfg['beta_u'])
    cx = math.hypot(speed, omega)
    env = cs.DubinsCBVFEnv(cs.World(goal=(cfg['goal_x'],cfg['goal_y'])), cs.Interval(-1.,1.), variant, dt, cfg['horizon'])
    tau = np.linspace(-(cfg['horizon']*dt + cfg['cbvf_terminal_guard']), -cfg['cbvf_terminal_guard'], cfg['horizon']+1)
    rows, errors = [], []
    for j, state in enumerate(trajectory['states']):
        s = np.asarray(state)
        bound = table.certified_local_psi_lipschitz(s, tau[j], tau[j+1], speed*dt, omega*dt, variant.model, cfg['gamma'])
        lip = bound['L_Psi_j']; epsilon = lip*(cx+1.)*dt
        # Runtime uses the full state-speed envelope for region selection;
        # the derivative-bound tube separately uses its tighter position bound.
        buffer, _ = cal.applied_buffer(s, cx*dt)
        terms = cs.model_constraint_terms(s, tau[j], table, env, cfg['gamma'])
        margin = terms.base_without_control + float(np.abs(terms.a_u).sum()) - buffer - epsilon
        recorded = trajectory['feasibility_margin'][j]
        if trajectory['active'][j]:
            error = abs(margin-recorded)
            errors.append(error)
        else:
            error = None
        rows.append(dict(interval=j, t_start=tau[j], Delta_j=dt, x=s[0], y=s[1], theta=s[2],
                         active=bool(trajectory['active'][j]), C_x=cx, L_Psi_j=lip,
                         epsilon_inter_j=epsilon, xi_hat_app=buffer,
                         reconstructed_feasibility_margin=margin, recorded_feasibility_margin=recorded,
                         reconstruction_error=error))
    output = folder/'final_representative_interval_certificates.csv'
    with output.open('w',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    result = dict(passed=bool(errors) and max(errors)<=1e-6, intervals=len(rows), active_intervals=len(errors),
                  max_margin_reconstruction_error=max(errors) if errors else None,
                  C_x=cx, L_Psi_min=min(r['L_Psi_j'] for r in rows), L_Psi_max=max(r['L_Psi_j'] for r in rows),
                  epsilon_inter_min=min(r['epsilon_inter_j'] for r in rows), epsilon_inter_max=max(r['epsilon_inter_j'] for r in rows),
                  csv=str(output), scope='Every interval of the first held-out CP episode, separately for every final seed; supplemental to exhaustive grid/all-episode checks.')
    (folder/'runtime_interval_audit.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return result


def main():
    parser=argparse.ArgumentParser()
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--run-dir',type=Path); group.add_argument('--results-dir',type=Path)
    args=parser.parse_args()
    folders=[args.run_dir] if args.run_dir else [Path(r['result_paths']['result']).parent for r in
        json.loads((args.results_dir/'run_state.json').read_text())['runs'].values()
        if r['status']=='completed' and r['configuration']['training_shield_method']=='cp']
    results=[]
    for folder in folders:
        result=audit_run(folder);results.append(result);print(json.dumps(result),flush=True)
    passed=bool(results) and all(r['passed'] for r in results) and (args.run_dir is not None or len(results)==15)
    if args.results_dir:
        output=args.results_dir/'aggregate/runtime_interval_audit.json';output.parent.mkdir(exist_ok=True)
        output.write_text(json.dumps(dict(passed=passed,runs=results),indent=2)+'\n')
    return int(not passed)


if __name__=='__main__':raise SystemExit(main())
