#!/usr/bin/env python3
"""Summarize every full design attempt without selecting away failures."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
base=ROOT/'results_intersample_v4'
rows=[]
for batch in sorted(base.glob('*/batch_configuration.json')):
    folder=batch.parent
    config=json.loads(batch.read_text())['configuration']
    row={'attempt':folder.name,'seed':config['seed'],'beta':config['beta_u'],
         'initial_x':config['initial_x'],'duration':config['horizon']*config['dt'],
         'activation':config['cp_activate_margin'],'dt':config['dt'],
         'goal':[config.get('goal_x',2.),config.get('goal_y',2.)],
         'clearance_edges':config['mondrian_clearance_edges'],
         'status':'pending result'}
    status=folder/'exit_status.json'
    if status.exists():
        row['exit_code']=json.loads(status.read_text())['returncode']
        row['status']='completed' if row['exit_code']==0 else 'failed'
    for phase,name in [('pretraining','pretraining_mondrian_calibration.json'),('final','final_frozen_policy_mondrian_calibration.json')]:
        path=folder/name
        if path.exists():
            calibration=json.loads(path.read_text())
            audit=calibration['offline_cp_qp_feasibility_audit']
            row[phase+'_audit']={key:audit[key] for key in ['n_active_certified_grid_nodes','n_infeasible_grid_nodes','min_feasibility_margin','min_uncalibrated_feasibility_margin']}
            row[phase+'_regions']=calibration['mondrian_calibration']['regions']
    result=folder/'training_and_evaluation_results.json'
    if result.exists():
        d=json.loads(result.read_text())
        row['evaluation']={m:{k:v[k] for k in ['goal_rate','unsafe_rate','infeasible_total','min_active_feasibility_margin']} for m,v in d['final_evaluation'].items()}
    rows.append(row)
(base/'DESIGN_STATUS.json').write_text(json.dumps(rows,indent=2)+'\n')
lines=['# V3 design attempts','','All percentages from completed pilots are design evidence, not held-out final-suite results.','','| Attempt | Status | Pretraining infeasible nodes | Worst pretraining margin |','|---|---|---:|---:|']
for r in rows:
    a=r.get('pretraining_audit',{})
    lines.append(f"| {r['attempt']} | {r['status']} | {a.get('n_infeasible_grid_nodes','pending')} | {a.get('min_feasibility_margin','pending')} |")
(base/'DESIGN_STATUS.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(rows,indent=2))
