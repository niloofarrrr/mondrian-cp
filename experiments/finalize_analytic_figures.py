"""Presentation-only fixes from accepted data; frozen simulations are untouched."""
import argparse
import csv
import json
import os
os.environ.setdefault('MPLBACKEND','Agg')
import shutil
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def finalize(study):
    study=Path(study).resolve()
    if not json.loads((study/'SCIENTIFIC_ACCEPTANCE.json').read_text())['passed']:raise RuntimeError('Study not accepted.')
    seed=json.loads((study/'FREEZE.json').read_text())['final_seeds'][0];conditions=['aligned','easy','hard'];methods=['nominal','cbvf','cp']
    labels={'nominal':'Nominal RL','cbvf':'Uncalibrated CBVF','cp':'Conformal-CBVF'};colors={'nominal':'#c43c39','cbvf':'#db931c','cp':'#1763a6'}
    figures=study/'figures';history=study/'presentation_history';history.mkdir(exist_ok=True)
    names=['matched_trajectories','calibration_buffers_coverage']
    for name in names:
        for extension in ['pdf','png']:
            old=figures/(name+'.'+extension);backup=history/old.name
            if not backup.exists():shutil.copyfile(old,backup)
    traces={(c,m):np.loadtxt(study/'runs'/('%s_seed%d'%(c,seed))/('performance_'+m+'_intervals.csv'),delimiter=',',skiprows=1,ndmin=2) for c in conditions for m in methods}
    allxy=np.concatenate([v[:,2:4] for v in traces.values()]);xmin=min(float(allxy[:,0].min())-.25,-.95);xmax=max(float(allxy[:,0].max())+.25,.95);ymin=min(float(allxy[:,1].min())-.25,-.95);ymax=max(float(allxy[:,1].max())+.25,2.55)
    fig,axes=plt.subplots(1,3,figsize=(12,5.8),sharex=True,sharey=True)
    for ax,c,title in zip(axes,conditions,['Aligned','Easy Mismatch','Hard Mismatch']):
        ax.add_patch(plt.Circle((0,0),.7,color='gray',alpha=.25));ax.add_patch(plt.Circle((0,1.8),.5,color='green',alpha=.1))
        for m in ['cp','cbvf','nominal']:
            v=traces[c,m];ax.plot(v[:,2],v[:,3],color=colors[m],lw=2,ls={'cp':'-','cbvf':'--','nominal':':'}[m],label=labels[m])
            unsafe=json.loads((study/'runs'/('%s_seed%d'%(c,seed))/('performance_'+m+'.json')).read_text())['rollouts'][0]['unsafe']
            ax.scatter(v[-1,5],v[-1,6],color=colors[m],s=40,marker='x' if unsafe else 'o',zorder=5)
        ax.set_title(title);ax.set_aspect('equal');ax.set_xlim(xmin,xmax);ax.set_ylim(ymin,ymax);ax.set_xlabel('x (m)');ax.grid(alpha=.12)
    axes[0].set_ylabel('y (m)');handles,legends=axes[0].get_legend_handles_labels();fig.legend(handles[::-1],legends[::-1],loc='upper center',ncol=3,frameon=False);fig.tight_layout(rect=[0,0,1,.92])
    for ext in ['pdf','png']:fig.savefig(figures/('matched_trajectories.'+ext),dpi=220,bbox_inches='tight')
    plt.close(fig)
    with (study/'tables/coverage_and_buffers.csv').open() as f:coverage=list(csv.DictReader(f))
    fig,axes=plt.subplots(1,2,figsize=(9,3.5));region_colors=['#1763a6','#db931c']
    for i,c in enumerate(conditions):
        rows=[r for r in coverage if r['condition']==c]
        for k,key in enumerate(['xi_near','xi_far']):
            values=[float(r[key]) for r in rows];axes[0].bar(i+(k-.5)*.32,np.mean(values),.3,yerr=np.std(values,ddof=1),capsize=3,color=region_colors[k],label=['Near region','Far region'][k] if i==0 else None)
        axes[1].plot(range(1,6),[int(r['covered']) for r in rows],marker=['o','s','^'][i],ls=[':','-','--'][i],label=['Aligned','Easy Mismatch','Hard Mismatch'][i])
    axes[0].set_xticks(range(3));axes[0].set_xticklabels(['Aligned','Easy','Hard']);axes[0].set_ylabel('Conformal tightening');axes[0].legend(fontsize=8)
    axes[1].axhline(95,color='black',ls='--',label='Acceptance threshold');axes[1].set_xticks(range(1,6));axes[1].set_xlabel('Final actor index');axes[1].set_ylabel('Held-out covered / 100');axes[1].set_ylim(93,101);axes[1].legend(fontsize=7,loc='lower left');fig.tight_layout()
    for ext in ['pdf','png']:fig.savefig(figures/('calibration_buffers_coverage.'+ext),dpi=220,bbox_inches='tight')
    plt.close(fig)
    (study/'PRESENTATION_VERIFICATION.json').write_text(json.dumps(dict(simulation_results_changed=False,changes=['Trajectory limits now show the entire predeclared trajectory, obstacle, and goal; legend is outside the plotting area.','Buffer colors are consistent by region across all conditions.'],trajectory_limits=dict(x=[xmin,xmax],y=[ymin,ymax]),source_script=str(Path(__file__).resolve()),passed=True),indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('study',type=Path);a=p.parse_args();finalize(a.study)
