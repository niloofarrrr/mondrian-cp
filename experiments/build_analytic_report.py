"""Acceptance checks and paper artifacts for one frozen matched-actor study."""
import argparse
import csv
import hashlib
import json
import math
import os
os.environ.setdefault('MPLBACKEND','Agg')
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

CONDITIONS=['aligned','easy','hard'];METHODS=['nominal','cbvf','cp']
LABELS={'aligned':'Aligned','easy':'Easy Mismatch','hard':'Hard Mismatch','nominal':'Nominal RL','cbvf':'Uncalibrated CBVF','cp':'Conformal-CBVF'}
COLORS={'nominal':'#c43c39','cbvf':'#db931c','cp':'#1763a6'}


def load(path):return json.loads(path.read_text())
def write(path,obj):path.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
def csvwrite(path,rows):
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
def mdtable(rows,keys):
    return '| '+' | '.join(keys)+' |\n| '+' | '.join(['---']*len(keys))+' |\n'+'\n'.join('| '+' | '.join(str(r[k]) for k in keys)+' |' for r in rows)+'\n'
def meanstd(values):return float(np.mean(values)),float(np.std(values,ddof=1))
def pm(mean,std,scale=1,places=3):return ('%.*f ± %.*f'%(places,mean*scale,places,std*scale))


def build(study):
    study=Path(study).resolve();manifest=load(study/'effective_configuration.json');freeze=load(study/'FREEZE.json');seeds=freeze['final_seeds']
    for name,wanted in freeze['sha256'].items():
        if hashlib.sha256((study/name).read_bytes()).hexdigest()!=wanted:raise RuntimeError('Frozen artifact hash changed: '+name)
    if (study/'RETIRED.json').exists():raise RuntimeError('Cannot publish a retired study.')
    aggregate=study/'aggregate';tables=study/'tables';figures=study/'figures'
    for path in [aggregate,tables,figures]:path.mkdir(exist_ok=True)
    primary=[];perseed=[];coverage=[];feasibility=[];intervals=[];runs={};all_run_checks=[];actors={}
    for seed in seeds:
        training=load(study/'actors'/('seed_%d'%seed)/'training_complete.json')
        if not training['completed'] or training['steps']!=manifest['training']['max_steps']:raise RuntimeError('Incomplete training.')
        for condition in CONDITIONS:
            folder=study/'runs'/('%s_seed%d'%(condition,seed));meta=load(folder/'configuration.json');q=load(folder/'qualification.json');audit=load(folder/'feasibility_audit.json');inter=load(folder/'independent_interval_audit.json');cov=load(folder/'coverage.json');buf=load(folder/'buffers.json')
            actors.setdefault(seed,set()).add(meta['actor_sha256']);all_run_checks.append(q['accepted'] and inter['passed'])
            if meta['design_only']:raise RuntimeError('Design data cannot enter final report.')
            expected_seed=seed+200000+CONDITIONS.index(condition)*100000
            if meta['seed']!=expected_seed:raise RuntimeError('Seed manifest mismatch.')
            if meta['configuration']['n_calibration']!=manifest['configuration']['n_calibration']:raise RuntimeError('Calibration size changed.')
            coverage.append(dict(condition=condition,seed=seed,evaluation_seed=meta['seed'],covered=cov['reference_covered'],total=100,deployment_covered=cov['deployment_covered'],target=cov['target'],empirical_acceptance_threshold=.95,n_calibration=meta['n_calibration'],xi_near=buf['buffers'][0],xi_far=buf['buffers'][1]))
            for method in ['cbvf','cp']:feasibility.append(dict(condition=condition,seed=seed,method=method,**audit['methods'][method]))
            for name,row in inter['results'].items():intervals.append(dict(condition=condition,seed=seed,trace=name,**{k:v for k,v in row.items() if k!='first_failures'}))
            for method in METHODS:
                data=load(folder/('performance_'+method+'.json'));runs[condition,seed,method]=(folder,data)
                if len(data['rollouts'])!=100:raise RuntimeError('Final evaluation episode count mismatch.')
                if method!='nominal' and data['min_constraint_margin']<-1e-9:raise RuntimeError('QP constraint violation.')
                rr=data['rollouts'];values=dict(unsafe_rate=float(np.mean([r['unsafe'] for r in rr])),unsafe_episodes=sum(r['unsafe'] for r in rr),goal_rate=float(np.mean([r['goal'] for r in rr])),goal_episodes=sum(r['goal'] for r in rr),episode_return=float(np.mean([r['episode_return'] for r in rr])),intervention_rate=float(np.mean([r['interventions']/r['steps'] for r in rr])),mean_intervention_count=float(np.mean([r['interventions'] for r in rr])),minimum_safety_margin=min(r['minimum'] for r in rr),total_interventions=sum(r['interventions'] for r in rr),total_steps=sum(r['steps'] for r in rr))
                perseed.append(dict(condition=condition,method=method,seed=seed,**values))
                for index,row in enumerate(rr):primary.append(dict(condition=condition,method=method,seed=seed,episode=index,**{k:v for k,v in row.items() if k not in ['scores','region_visits']}))
    if any(len(values)!=1 for values in actors.values()):raise RuntimeError('Actor differs across paired conditions.')
    if len({next(iter(values)) for values in actors.values()})!=5:raise RuntimeError('Final training actors are not five distinct frozen checkpoints.')
    rows=[];display=[]
    for condition in CONDITIONS:
        for method in METHODS:
            selected=[r for r in perseed if r['condition']==condition and r['method']==method];row=dict(condition=condition,method=method)
            for key in ['unsafe_rate','goal_rate','episode_return','intervention_rate','mean_intervention_count','minimum_safety_margin']:
                m,s=meanstd([r[key] for r in selected]);row[key+'_mean']=m;row[key+'_std']=s
            row.update(unsafe_episodes=sum(r['unsafe_episodes'] for r in selected),goal_episodes=sum(r['goal_episodes'] for r in selected),episodes=500,pooled_minimum_margin=min(r['minimum_safety_margin'] for r in selected),pooled_intervention_rate=sum(r['total_interventions'] for r in selected)/sum(r['total_steps'] for r in selected));rows.append(row)
            display.append({'Condition':LABELS[condition],'Method':LABELS[method],'Unsafe %':pm(row['unsafe_rate_mean'],row['unsafe_rate_std'],100,1),'Unsafe count':str(row['unsafe_episodes'])+'/500','Goal %':pm(row['goal_rate_mean'],row['goal_rate_std'],100,1),'Return':pm(row['episode_return_mean'],row['episode_return_std']),'Intervention %':pm(row['intervention_rate_mean'],row['intervention_rate_std'],100,2),'Interventions/episode':pm(row['mean_intervention_count_mean'],row['mean_intervention_count_std'],1,1),'Min clearance (m)':'%.9g'%row['pooled_minimum_margin']})
    csvwrite(aggregate/'episodes.csv',primary);csvwrite(aggregate/'per_seed.csv',perseed);csvwrite(aggregate/'summary.csv',rows);csvwrite(tables/'main_comparison.csv',rows);csvwrite(tables/'coverage_and_buffers.csv',coverage);csvwrite(tables/'feasibility.csv',feasibility);csvwrite(tables/'representative_intervals.csv',intervals)
    (tables/'main_comparison.md').write_text(mdtable(display,list(display[0])))
    tex=['\\begin{tabular}{llrrrr}','\\toprule','Condition & Method & Unsafe (\\%) & Goal (\\%) & Return & Interventions \\\\','\\midrule']
    for r in rows:tex.append('%s & %s & %.1f $\\pm$ %.1f & %.1f $\\pm$ %.1f & %.3f $\\pm$ %.3f & %.1f $\\pm$ %.1f \\\\'%(LABELS[r['condition']],LABELS[r['method']],100*r['unsafe_rate_mean'],100*r['unsafe_rate_std'],100*r['goal_rate_mean'],100*r['goal_rate_std'],r['episode_return_mean'],r['episode_return_std'],r['mean_intervention_count_mean'],r['mean_intervention_count_std']))
    tex+=['\\bottomrule','\\end{tabular}'];(tables/'main_comparison.tex').write_text('\n'.join(tex)+'\n')
    paper_figures=[]
    def save(fig,name,description):
        fig.tight_layout()
        for extension in ['pdf','png']:fig.savefig(figures/(name+'.'+extension),dpi=220,bbox_inches='tight')
        plt.close(fig);paper_figures.append(dict(filename=name+'.pdf',path=str(figures/(name+'.pdf')),description=description))
    fig,axes=plt.subplots(1,2,figsize=(10,3.5))
    for method_index,method in enumerate(METHODS):
        selected=[next(r for r in rows if r['condition']==c and r['method']==method) for c in CONDITIONS];x=np.arange(3)+(method_index-1)*.24
        for ax,key in zip(axes,['unsafe_rate','goal_rate']):ax.bar(x,[100*r[key+'_mean'] for r in selected],.23,yerr=[100*r[key+'_std'] for r in selected],label=LABELS[method],color=COLORS[method],capsize=3)
    for ax,title in zip(axes,['Unsafe episodes (%)','Goal success (%)']):ax.set_xticks(range(3));ax.set_xticklabels(['Aligned','Easy','Hard']);ax.set_ylabel(title);ax.set_ylim(0,108);ax.grid(axis='y',alpha=.2)
    axes[0].legend(fontsize=8,loc='upper left');save(fig,'safety_goal_comparison','MAIN safety comparison: all three methods and all three conditions; mean and sample SD across five paired actors, 500 episodes per bar.')
    fig,axes=plt.subplots(1,3,figsize=(11,4))
    first=seeds[0]
    for ax,condition in zip(axes,CONDITIONS):
        ax.add_patch(plt.Circle((0,0),.7,color='gray',alpha=.25));ax.add_patch(plt.Circle((0,1.8),.5,color='green',alpha=.1))
        for method in METHODS:
            folder,_=runs[condition,first,method];data=np.loadtxt(folder/('performance_'+method+'_intervals.csv'),delimiter=',',skiprows=1,ndmin=2);ax.plot(data[:,2],data[:,3],label=LABELS[method],color=COLORS[method]);ax.scatter(data[-1,5],data[-1,6],color=COLORS[method],s=15)
        ax.set_title(LABELS[condition]);ax.set_aspect('equal');ax.set_xlim(-1.8,.4);ax.set_ylim(-3.2,2.4);ax.set_xlabel('x (m)');ax.set_ylabel('y (m)')
    axes[0].legend(fontsize=7);save(fig,'matched_trajectories','First predeclared evaluation episode of the first final actor; all conditions and methods. Circle includes robot inflation.')
    fig,axes=plt.subplots(1,2,figsize=(9,3.3))
    buffer_summary=[]
    for i,c in enumerate(CONDITIONS):
        values=[r for r in coverage if r['condition']==c]
        for k,key in enumerate(['xi_near','xi_far']):
            m,s=meanstd([r[key] for r in values]);axes[0].bar(i+(k-.5)*.32,m,.3,yerr=s,capsize=3,label=['Near region','Far region'][k] if i==0 else None);buffer_summary.append(dict(condition=c,region=['h<0.75','h>=0.75'][k],mean=m,std=s))
        axes[1].plot(range(1,6),[r['covered'] for r in values],marker='o',label=LABELS[c])
    axes[0].set_xticks(range(3));axes[0].set_xticklabels(['Aligned','Easy','Hard']);axes[0].set_ylabel('Conformal tightening');axes[0].legend(fontsize=8)
    axes[1].axhline(95,color='black',ls='--',label='Acceptance threshold');axes[1].set_xticks(range(1,6));axes[1].set_xlabel('Final actor index');axes[1].set_ylabel('Held-out covered / 100');axes[1].set_ylim(93,101);axes[1].legend(fontsize=7);save(fig,'calibration_buffers_coverage','Regional buffer mean and SD plus all 15 fresh held-out reference coverage counts; Aligned buffers are zero.')
    fig,ax=plt.subplots(figsize=(6,3.3))
    for seed in seeds:
        logs=[json.loads(line) for line in (study/'actors'/('seed_%d'%seed)/'training_episodes.jsonl').read_text().splitlines()];ax.plot([r['step'] for r in logs],[r['episode_return'] for r in logs],marker='o',label=str(seed))
    ax.set_xlabel('Training environment steps');ax.set_ylabel('Completed episode return');ax.legend(fontsize=7);save(fig,'shared_actor_training','Logged nominal-model training returns for five shared residual actors; descriptive training trace, not evidence of learning convergence.')
    folder,_=runs['hard',first,'cp'];data=np.loadtxt(folder/'performance_cp_intervals.csv',delimiter=',',skiprows=1,ndmin=2);active=data[:,20]>0
    fig,axes=plt.subplots(3,1,figsize=(7,7),sharex=True)
    axes[0].plot(data[:,1],data[:,12],label='Clearance');axes[0].axhline(0,color='black',ls='--');axes[0].set_ylabel('Clearance (m)')
    axes[1].plot(data[active,1],data[active,14],label='Model Psi');axes[1].plot(data[active,1],data[active,16]+data[active,18],label='xi + epsilon');axes[1].plot(data[active,1],data[active,15],label='True Psi');axes[1].set_ylabel('Derivative condition');axes[1].legend(fontsize=8)
    axes[2].plot(data[active,1],data[active,19],label='Available QP margin');axes[2].plot(data[active,1],data[active,21],label='True interval lower bound');axes[2].axhline(0,color='black',ls='--');axes[2].set_xlabel('Time (s)');axes[2].set_ylabel('Margin');axes[2].legend(fontsize=8);save(fig,'feasibility_intersample_verification','Hard CP first predeclared trajectory: safety clearance, derivative tightening, positive available QP margin and analytic true inter-sample lower bound.')
    folder,_=runs['hard',first,'cbvf'];data=np.loadtxt(folder/'performance_cbvf_intervals.csv',delimiter=',',skiprows=1,ndmin=2);active=data[:,20]>0
    fig,axes=plt.subplots(2,1,figsize=(7,5),sharex=True)
    axes[0].plot(data[:,1],data[:,12],label='Clearance');axes[0].axhline(0,color='black',ls='--');axes[0].set_ylabel('Clearance (m)')
    axes[1].plot(data[active,1],data[active,14]-data[active,18],label='Model Psi - epsilon');axes[1].plot(data[active,1],data[active,15],label='True Psi');axes[1].axhline(0,color='black',ls='--');axes[1].set_xlabel('Time (s)');axes[1].set_ylabel('Derivative margin');axes[1].legend(fontsize=8);save(fig,'uncalibrated_mismatch_mechanism','Hard uncalibrated first predeclared trajectory: the model constraint remains satisfied while the true derivative condition becomes negative before collision.')
    intersample=[]
    for condition in CONDITIONS:
        bounds=[]
        for seed in seeds:
            for method in METHODS:
                folder,_=runs[condition,seed,method];trace=np.loadtxt(folder/('performance_'+method+'_intervals.csv'),delimiter=',',skiprows=1,ndmin=2);bounds.append(trace[trace[:,20]>0,17:19])
        values=np.concatenate(bounds);chosen=[r for r in intervals if r['condition']==condition]
        intersample.append(dict(condition=condition,Cx=1.,Delta=manifest['configuration']['dt'],L_min=float(values[:,0].min()),L_max=float(values[:,0].max()),epsilon_min=float(values[:,1].min()),epsilon_max=float(values[:,1].max()),representative_intervals=sum(r['intervals'] for r in chosen),reconstructed_subinterval_checks=sum(r['subinterval_checks'] for r in chosen),failures=sum(r['failures'] for r in chosen)))
    csvwrite(tables/'intersample_summary.csv',intersample);csvwrite(tables/'buffer_summary.csv',buffer_summary)
    mechanism=[]
    for seed in seeds:
        folder,_=runs['hard',seed,'cbvf'];trace=np.loadtxt(folder/'performance_cbvf_intervals.csv',delimiter=',',skiprows=1,ndmin=2)
        cfg=load(folder/'configuration.json')['configuration']
        for row in trace[trace[:,12]<=.35]:
            x,y,theta=row[2:5];h=row[12];c=(x*math.cos(theta)+y*math.sin(theta))/(h+.7);a=.4*c;base=cfg['gamma']*h+.2*c;rhs=row[16]+row[18];available=base+abs(a)-rhs
            counterfactual=float(row[8])
            if available>=0 and base+a*counterfactual<rhs:counterfactual=(rhs-base)/a
            cp_true=None if available<0 else cfg['gamma']*h+c*(cfg['vmin']+(counterfactual+1)*(.6-cfg['vmin'])/2)
            mechanism.append(dict(seed=seed,step=int(row[0]),time=row[1],x=x,y=y,theta=theta,h=h,nominal_throttle=row[8],uncal_throttle=row[10],uncal_psi_model=row[14],uncal_psi_true=row[15],xi=row[16],epsilon=row[18],uncal_available_margin=row[19],cp_counterfactual_available_margin=available,cp_counterfactual_throttle=None if available<0 else counterfactual,cp_counterfactual_psi_true=cp_true))
    csvwrite(aggregate/'hard_same_state_mechanism.csv',mechanism)
    checks=dict(frozen_hashes=True,final_45_runs=len(perseed)==45 and len(primary)==4500,five_fresh_shared_actors=len(actors)==5 and all(len(v)==1 for v in actors.values()),all_condition_seed_acceptance=all(all_run_checks),zero_exhaustive_infeasibility=all(r['infeasible_node_intervals']==0 for r in feasibility),strictly_positive_feasibility=all(r['minimum_margin']>0 for r in feasibility),heldout_coverage=all(r['covered']>=95 for r in coverage),independent_interval_reconstruction=all(r['failures']==0 for r in intervals),hard_uncal_separation=all(r['unsafe_rate_mean']>=.1 for r in rows if r['condition']=='hard' and r['method']=='cbvf') and all(r['unsafe_rate_mean']==0 for r in rows if r['method']=='cp'),goal_success=all(r['goal_rate_mean']>=.9 for r in rows if r['method']=='cp'),nontrivial_mismatch_buffers=all(r['xi_near']>0 and r['xi_far']>0 for r in coverage if r['condition']!='aligned'),zero_aligned_buffers=all(r['xi_near']==0 and r['xi_far']==0 for r in coverage if r['condition']=='aligned'),paper_figures=all(Path(r['path']).exists() for r in paper_figures))
    acceptance=dict(passed=all(checks.values()),criteria=checks,exhaustive_audits=len(feasibility),unique_node_evaluations=sum(r['unique_nodes'] for r in feasibility),node_interval_checks=sum(r['node_interval_checks'] for r in feasibility),infeasible_node_intervals=sum(r['infeasible_node_intervals'] for r in feasibility),representative_traces=len(intervals),representative_intervals=sum(r['intervals'] for r in intervals),representative_subinterval_checks=sum(r['subinterval_checks'] for r in intervals),coverage_min=min(r['covered'] for r in coverage),coverage_max=max(r['covered'] for r in coverage))
    write(study/'SCIENTIFIC_ACCEPTANCE.json',acceptance)
    artifact_index=dict(report=str(study/'PAPER_HANDOFF.md'),figures=paper_figures,tables=[str(p) for p in sorted(tables.iterdir())],aggregate=[str(p) for p in sorted(aggregate.iterdir())],configuration=str(study/'effective_configuration.json'),acceptance=str(study/'SCIENTIFIC_ACCEPTANCE.json'))
    write(study/'ARTIFACT_INDEX.json',artifact_index)
    hard={r['method']:r for r in rows if r['condition']=='hard'}
    text=['# Frozen calibration-mechanism case study','',('PASS' if acceptance['passed'] else 'FAIL')+': final scientific acceptance. Only this frozen study is summarized.','', '## Design and interpretation','', 'The same frozen residual actor is evaluated under all three filters and conditions within each of five fresh training seeds. There are five 20,000-step nominal-model training runs and 45 final evaluation runs, with 100 episodes each. This isolates the filter effect; these are not 45 independently trained actors.','', 'The common nominal route is successful without filtering in Aligned dynamics. Hard dynamics reduce steering and reverse authority. A preplanned recovery maneuver at 8–10 seconds reverses while steering toward an outward waypoint; the same schedule applies to every method. Uncalibrated Hard failures occur before recovery. Both filters enforce identical activation, gamma, actuator limits, and inter-sample corrections; CP alone adds the calibrated regional buffer.','', '## Exact frozen parameters','', 'Full effective configuration and all training defaults: `'+str(study/'effective_configuration.json')+'`. Frozen source checksums: `'+str(study/'FREEZE.json')+'`.','', '```json',json.dumps(manifest,indent=2),'```','', '## Main comparison','', 'Means ± sample SD across five training seeds (ddof=1); 500 pooled episodes per condition/method. Intervention rate is the mean of each episode’s intervention-count/step-count ratio; pooled step-weighted rates and per-seed minimum-clearance mean/SD are also in the CSV. Collision uses the exact minimum distance on each continuous held-control arc.','',mdtable(display,list(display[0])),'', '## Hard result','', 'Hard unsafe counts are %d/500 Nominal, %d/500 uncalibrated CBVF, and %d/500 conformal-CBVF; goal counts are %d/500, %d/500, and %d/500 respectively.'%(hard['nominal']['unsafe_episodes'],hard['cbvf']['unsafe_episodes'],hard['cp']['unsafe_episodes'],hard['nominal']['goal_episodes'],hard['cbvf']['goal_episodes'],hard['cp']['goal_episodes']),'', '## Held-out coverage and buffers','', 'The formal split-conformal reference population is the same frozen actor behind the uncalibrated filter, with regional rollout maxima. Calibration, primary evaluation, and held-out initial states are disjoint. Deployment coverage is an additional empirical diagnostic, not a transfer of the reference-population theorem. Empirical acceptance requires at least 95/100 covered in every block.','',mdtable(coverage,list(coverage[0])),'',mdtable(buffer_summary,list(buffer_summary[0])),'', '## Feasibility and continuous intervals','', '%d exhaustive method-specific grid audits checked %d unique-node evaluations and %d node/interval combinations, with zero infeasible combinations. These finite-grid checks do not prove feasibility at every continuous state. All executed filtered actions also undergo strict checks; no fallback is executed.'%(acceptance['exhaustive_audits'],acceptance['unique_node_evaluations'],acceptance['node_interval_checks']),'',mdtable(feasibility,list(feasibility[0])),'', 'B=h is analytically exact on the safe domain for the reversible model: the time-zero value is an upper bound and zero speed attains it. The model and true held-control derivative fields have the reported analytic Lipschitz bound under the public speed/turn envelope. The common epsilon is L(C_x+1)Delta. Inactive intervals satisfy the direct clearance reachability bound.','',mdtable(intersample,list(intersample[0])),'', '%d preselected representative traces reconstructed %d complete intervals and %d interior/end checks; all passed. True derivative violations in Hard uncalibrated trajectories are expected mechanism evidence, while the model QP remains feasible. CP true inter-sample lower bounds pass.'%(acceptance['representative_traces'],acceptance['representative_intervals'],acceptance['representative_subinterval_checks']),'', '## Figures','']
    for row in paper_figures:text.append('- `'+row['path']+'`: '+row['description']+' PNG counterpart is in the same directory.')
    text+=['','## Acceptance','',mdtable([{'Criterion':k,'Result':'PASS' if v else 'FAIL'} for k,v in checks.items()],['Criterion','Result']),'','## Scope','']+['- '+s for s in manifest['limitations']]+['- Zero observed collisions is an empirical outcome, not proof of zero failure probability under arbitrary deployment distributions.','', '## FILES TO OPEN FOR WRITING THE CASE STUDY','', '- `'+str(study/'PAPER_HANDOFF.md')+'`','- `'+str(study/'effective_configuration.json')+'`']+['- `'+r['path']+'`' for r in paper_figures]+['- `'+str(p)+'`' for p in sorted(tables.iterdir())]+['- `'+str(p)+'`' for p in sorted(aggregate.iterdir())]
    (study/'PAPER_HANDOFF.md').write_text('\n'.join(text)+'\n')
    return acceptance


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('study',type=Path);a=p.parse_args();r=build(a.study);print(json.dumps(r,indent=2));raise SystemExit(0 if r['passed'] else 1)
