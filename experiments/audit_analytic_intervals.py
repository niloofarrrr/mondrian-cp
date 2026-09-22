"""Independently reconstruct every saved representative interval with scalar flow."""
import argparse
import json
import math
from pathlib import Path
import numpy as np
from analytic_cbvf_mechanism import fields,project,step_exact


def audit_directory(directory):
    directory=Path(directory);metadata=json.loads((directory/'configuration.json').read_text());cfg=metadata['configuration']
    buffers=json.loads((directory/'buffers.json').read_text())['buffers'];results={}
    for path in sorted(directory.glob('*_intervals.csv')):
        name=path.stem.removesuffix('_intervals') if hasattr(str,'removesuffix') else path.stem[:-10]
        method='cp' if name in ('performance_cp','heldout_deployment') else ('nominal' if name=='performance_nominal' else 'cbvf')
        rows=np.loadtxt(path,delimiter=',',skiprows=1,ndmin=2);failures=[];max_endpoint=0.;max_lipschitz_excess=0.;min_true_bound=float('inf');min_feas=float('inf');active_count=0;model_true_violations=0
        for index,row in enumerate(rows):
            j,t=row[:2];s=row[2:5];recorded_next=row[5:8];nom=row[8:10];u=row[10:12]
            h,c,L,eps=fields(s,cfg);active=h<=cfg.get('activation_threshold',1.)
            region=[i for i,(lo,hi) in enumerate([(-float('inf'),.75),(.75,float('inf'))]) if h+cfg['dt']>=lo and h-cfg['dt']<=hi]
            xi=max(buffers[i] for i in region)
            # Calibration is the uncalibrated controller; its trace was
            # recorded before the regional quantiles existed.
            recorded_xi=0. if name=='calibration' else xi
            if abs(row[16]-recorded_xi)>1e-12 or abs(row[17]-L)>1e-11 or abs(row[18]-eps)>1e-12:failures.append([index,'bound reconstruction'])
            expected=nom
            if method!='nominal' and active:
                expected,margin,_,_,psi=project(s,nom,cfg,xi if method=='cp' else 0.)
                min_feas=min(min_feas,margin);active_count+=1
                if margin<=0:failures.append([index,'nonpositive available margin'])
            if np.max(np.abs(expected-u))>2e-9:failures.append([index,'QP reconstruction'])
            end,minimum=step_exact(s,u,cfg);error=float(np.max(np.abs(end-recorded_next)));max_endpoint=max(max_endpoint,error)
            if error>2e-10 or abs(minimum-row[13])>2e-10:failures.append([index,'exact flow reconstruction'])
            if index+1<len(rows) and int(rows[index+1,0])==int(j)+1 and np.max(np.abs(rows[index+1,2:5]-end))>2e-10:failures.append([index,'trajectory continuity'])
            v=cfg['vmin']+(u[0]+1)*(.6-cfg['vmin'])/2;psi_true=cfg['gamma']*h+c*v
            psi_model=cfg['gamma']*h+c*(.2+.4*u[0])
            model_true_violations+=int(psi_model>=eps-1e-9 and psi_true<0)
            if abs(psi_true-row[15])>1e-10 or abs(psi_model-row[14])>1e-10:failures.append([index,'derivative reconstruction'])
            if active:
                min_true_bound=min(min_true_bound,psi_true-L*cfg['dt'])
                if method=='cp' and psi_true-L*cfg['dt']<-1e-10:failures.append([index,'true continuous interval bound'])
            elif h-.6*cfg['dt']<=0:failures.append([index,'inactive reachability bound'])
            for fraction in [.25,.5,.75,1.]:
                dt=fraction*cfg['dt'];inside,_=step_exact(s,u,dict(cfg,dt=dt));hh,cc,_,_=fields(inside,cfg)
                p=cfg['gamma']*hh+cc*v;excess=abs(p-psi_true)-L*dt
                max_lipschitz_excess=max(max_lipschitz_excess,excess)
                if excess>1e-10:failures.append([index,'Lipschitz inequality'])
        results[name]=dict(method=method,intervals=len(rows),active_intervals=active_count,subinterval_checks=4*len(rows),failures=len(failures),first_failures=failures[:10],max_endpoint_error=max_endpoint,max_lipschitz_excess=max_lipschitz_excess,min_true_interval_bound=min_true_bound,minimum_feasibility_margin=None if method=='nominal' else min_feas,model_safe_true_negative_intervals=model_true_violations)
    result=dict(passed=all(r['failures']==0 for r in results.values()),representative_selection='first predeclared rollout of every calibration/performance/heldout set; no post-hoc trajectory selection',intervals=sum(r['intervals'] for r in results.values()),subinterval_checks=sum(r['subinterval_checks'] for r in results.values()),results=results)
    (directory/'independent_interval_audit.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory',type=Path);a=p.parse_args();r=audit_directory(a.directory);print(json.dumps(r));raise SystemExit(0 if r['passed'] else 1)
