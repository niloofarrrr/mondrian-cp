"""Accepted-only paper extraction; never reads design or retired results."""
import argparse
import csv
import json
import math
from pathlib import Path
import numpy as np


def readcsv(path):
    with path.open() as f:return list(csv.DictReader(f))
def table(rows,keys):return '| '+' | '.join(keys)+' |\n| '+' | '.join(['---']*len(keys))+' |\n'+'\n'.join('| '+' | '.join(str(r[k]) for k in keys)+' |' for r in rows)


def extract(study):
    study=Path(study).resolve();accept=json.loads((study/'SCIENTIFIC_ACCEPTANCE.json').read_text())
    if not accept['passed'] or (study/'RETIRED.json').exists():raise RuntimeError('Only a final accepted study can be extracted.')
    manifest=json.loads((study/'effective_configuration.json').read_text());cfg=manifest['configuration'];seeds=manifest['entry']['final_training_seeds']
    for seed in seeds:
        aligned=study/'runs'/('aligned_seed%d'%seed)
        uncal=json.loads((aligned/'performance_cbvf.json').read_text());calibrated=json.loads((aligned/'performance_cp.json').read_text())
        for left,right in zip(uncal['rollouts'],calibrated['rollouts']):
            for key in ['unsafe','goal','steps','interventions','active_steps','true_derivative_violations']:
                if left[key]!=right[key]:raise RuntimeError('Aligned zero-buffer paired outcome differs: '+key)
            for key in ['minimum','episode_return','min_qp']:
                if abs(left[key]-right[key])>1e-10:raise RuntimeError('Aligned paired numeric outcome differs: '+key)
        left=np.loadtxt(aligned/'performance_cbvf_intervals.csv',delimiter=',',skiprows=1,ndmin=2)
        right=np.loadtxt(aligned/'performance_cp_intervals.csv',delimiter=',',skiprows=1,ndmin=2)
        if left.shape!=right.shape or not np.allclose(left,right,atol=1e-10,rtol=0):
            raise RuntimeError('Aligned zero-buffer representative traces differ beyond roundoff tolerance.')
    summary=readcsv(study/'aggregate/summary.csv');coverage=readcsv(study/'tables/coverage_and_buffers.csv');feas=readcsv(study/'tables/feasibility.csv');intersample=readcsv(study/'tables/intersample_summary.csv');bufferstats=readcsv(study/'tables/buffer_summary.csv');artifacts=json.loads((study/'ARTIFACT_INDEX.json').read_text())
    mechanism=[]
    for seed in seeds:
        folder=study/'runs'/('hard_seed%d'%seed);trace=np.loadtxt(folder/'performance_cbvf_intervals.csv',delimiter=',',skiprows=1,ndmin=2);truth=json.loads((folder/'configuration.json').read_text())['configuration']
        for row in trace[trace[:,12]<=.35]:
            x,y,theta=row[2:5];h=row[12];c=(x*math.cos(theta)+y*math.sin(theta))/(h+.7);a=.4*c;base=cfg['gamma']*h+.2*c;rhs=row[16]+row[18];available=base+abs(a)-rhs;counterfactual=float(row[8])
            if available>=0 and base+a*counterfactual<rhs:counterfactual=(rhs-base)/a
            cp_true=None if available<0 else cfg['gamma']*h+c*(truth['vmin']+(counterfactual+1)*(.6-truth['vmin'])/2)
            mechanism.append(dict(seed=seed,step=int(row[0]),time=row[1],x=x,y=y,theta=theta,h=h,nominal_throttle=row[8],uncal_throttle=row[10],uncal_psi_model=row[14],uncal_psi_true=row[15],xi=row[16],epsilon=row[18],uncal_available_margin=row[19],cp_counterfactual_available_margin=available,cp_counterfactual_throttle=None if available<0 else counterfactual,cp_counterfactual_psi_true=cp_true))
    mechanism_path=study/'aggregate/hard_same_state_mechanism.csv'
    with mechanism_path.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(mechanism[0]));w.writeheader();w.writerows(mechanism)
    totals=[]
    for condition in ['aligned','easy','hard']:
        v=[r for r in feas if r['condition']==condition]
        totals.append(dict(Condition=condition,Audits=len(v),Node_interval_checks=sum(int(r['node_interval_checks']) for r in v),Uncalibrated_min_margin=min(float(r['minimum_margin']) for r in v if r['method']=='cbvf'),CP_min_margin=min(float(r['minimum_margin']) for r in v if r['method']=='cp'),Infeasible=sum(int(r['infeasible_node_intervals']) for r in v)))
    clearances=[{'Condition':r['condition'],'Method':r['method'],'Per-seed minimum mean':r['minimum_safety_margin_mean'],'Per-seed minimum SD':r['minimum_safety_margin_std'],'Pooled minimum':r['pooled_minimum_margin']} for r in summary]
    hard={r['method']:r for r in summary if r['condition']=='hard'}
    text=['# Accepted study: paper numbers','',str(study),'','## Frozen case-study parameters','', 'With normalized controls (a,b) in [-1,1]^2, p_dot=v(a)(cos(theta),sin(theta)):','', '- Aligned/model: v=0.2+0.4a; theta_dot=0.8b.','- Easy: v=0.2125+0.3875a; theta_dot=0.625b.','- Hard: v=0.225+0.375a; theta_dot=0.43b.','- Reset: independent uniforms centered at (-0.9,-3,0.7), half-width 0.02 in x, y, and heading.','- Obstacle: origin, radius 0.5 m; robot radius 0.2 m; inflated radius 0.7 m. Goal: (0,1.8), radius 0.5 m. Audit box: [-4,4]^2.','- Exact analytic CBVF B=h=hypot(x,y)-0.7 on the safe domain; gamma=1.5. Delta=0.002 s; horizon=25 s (12,500 intervals); activation B<=1 for both filters; no release hysteresis.','- Public bounds: |v|<=0.6, |omega|<=0.8, C_x=1. Exact held-control flow and exact minimum clearance over every continuous arc.','- Reference before 8 s: waypoint (-0.9,0.65) until y>=0.45, then goal. From 8 s: waypoint (-1.4,-1.2) while x>-1.2 and y<0.45; otherwise (-1.4,0.65) while y<0.45, then goal. Throttle is -1 during [8,10) s and +1 otherwise; yaw=clip(1.5*wrapped_heading_error/0.8,-1,1). No orbit branch.','- Nominal actor: this shared reference plus 0.02 times a deterministic frozen learned residual, with actuator clipping before either QP. The QP constraint itself is never clipped.','- Five independently initialized SAC-Lag residual actors trained for 20,000 steps on Aligned/no-filter dynamics; hidden layers (256,256), ReLU; tanh mean at evaluation. Warmup/reference proposals 1,000 steps; batch 256; update ratio 1; all learning rates 0.0003; discount 0.99; tau 0.005; initial temperature 0.1; cost limit 1. Full defaults are in effective_configuration.json.','- Reward per step: -Delta*distance_to_goal(next state)+100*goal-100*collision; exact-arc collision cost=1; goal/collision terminate.','- Calibration n=%d, delta=%g, two regions h<0.75 and h>=0.75 with delta/2 each. Quantile order ceil((n+1)(1-delta/2)).'%(cfg['n_calibration'],cfg['delta']),'- Final actor seeds: '+', '.join(map(str,seeds))+'. Each is reused across all three conditions and all three filters.','- 45 evaluations x 100 episodes = 4,500 primary episodes. Fresh reference coverage: 15 x 100 rollouts, plus a separate CP deployment diagnostic on those held-out starts.','- Calibration seed = actor seed + 200000 + condition_index*100000 (Aligned=0, Easy=1, Hard=2); primary starts +10000; coverage starts +20000. Training reset RNG = actor seed +100000. No final seeds were used in design.','','## Main numerical comparison','','Mean ± sample SD across five actor seeds; unsafe/goal counts pooled over 500 episodes. Intervention rate is episode-averaged.','',(study/'tables/main_comparison.md').read_text(),'','### Minimum physical safety margins','',table(clearances,list(clearances[0])),'','## Feasibility','', '%d exhaustive audits; %d node/interval combinations; zero calibrated and uncalibrated infeasible combinations. There are 2,700 grid nodes per method-specific audit; the stationary margin is reused across its exact eligible interval multiplicity. These are finite-grid and executed-trajectory checks, not a continuous-state global feasibility theorem.'%(accept['exhaustive_audits'],accept['node_interval_checks']),'',table(totals,list(totals[0])),'','## All 15 held-out coverage results and regional buffers','',table(coverage,list(coverage[0])),'', 'Coverage range: %d/100 to %d/100; formal target %.4g; empirical acceptance threshold 95/100. Aligned zero buffers are exact. Reference-rollout coverage is the conformal population; deployment coverage is reported separately.'%(accept['coverage_min'],accept['coverage_max'],1-cfg['delta']),'',table(bufferstats,list(bufferstats[0])),'','## Inter-sample quantities','', 'L_Psi=hypot(gamma+0.6/(r-0.6*Delta),0.6); epsilon=L_Psi(C_x+1)Delta=0.004 L_Psi. The ranges below cover the active intervals of all primary representative trajectories.','',table(intersample,list(intersample[0])),'', '%d predeclared representative traces; %d reconstructed intervals; %d interior/end checks; all passed.'%(accept['representative_traces'],accept['representative_intervals'],accept['representative_subinterval_checks']),'','## Paper-ready sentences','', '1. We evaluated five frozen residual actors under three dynamics conditions and three filters, using 100 paired episodes per condition–actor block (4,500 primary episodes).','2. Under Hard Mismatch, Nominal and uncalibrated CBVF had %s/500 and %s/500 unsafe episodes, respectively, whereas conformal-CBVF had %s/500.'%(hard['nominal']['unsafe_episodes'],hard['cbvf']['unsafe_episodes'],hard['cp']['unsafe_episodes']),'3. Hard-Mismatch goal counts were %s/500, %s/500, and %s/500 for Nominal, uncalibrated CBVF, and conformal-CBVF, respectively.'%(hard['nominal']['goal_episodes'],hard['cbvf']['goal_episodes'],hard['cp']['goal_episodes']),'4. Both filters used identical actors, activation, gamma, actuator bounds, and inter-sample corrections; the calibrated regional mismatch buffer was the only difference in their QP constraints.','5. All %d exhaustive method-specific grid audits passed with zero infeasible node/interval combinations, and all %d reconstructed representative intervals passed verification.'%(accept['exhaustive_audits'],accept['representative_intervals']),'6. Fresh held-out reference coverage ranged from %d/100 to %d/100 across the 15 condition–actor blocks; deployment coverage was assessed separately, without claiming automatic transfer of the reference-population guarantee.'%(accept['coverage_min'],accept['coverage_max']),'','## Final figures','']
    for figure in artifacts['figures']:text.append('- `'+figure['path']+'`: '+figure['description'])
    text+=['','## Final tables','']+['- `'+s+'`' for s in artifacts['tables']]+['','## Acceptance','',table([{'Criterion':k,'Result':'PASS' if v else 'FAIL'} for k,v in accept['criteria'].items()],['Criterion','Result']),'','## FILES TO OPEN FOR WRITING THE CASE STUDY','', '- `'+str(study/'PAPER_NUMBERS.md')+'`','- `'+str(study/'PAPER_HANDOFF.md')+'`','- `'+str(study/'effective_configuration.json')+'`']+['- `'+r['path']+'`' for r in artifacts['figures']]+['- `'+p+'`' for p in artifacts['tables']+artifacts['aggregate']]+['- `'+str(mechanism_path)+'`']
    output=study/'PAPER_NUMBERS.md';output.write_text('\n'.join(text)+'\n');return output


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('study',type=Path);a=p.parse_args();print(extract(a.study))
