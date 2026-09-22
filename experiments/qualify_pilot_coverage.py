#!/usr/bin/env python3
"""Wait for one learned pilot and collect coverage only after its other gates pass."""
import argparse,json,os,subprocess,time,fcntl
from pathlib import Path
from audit_pilot_certificates import audit_certificates
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=argparse.ArgumentParser();p.add_argument('run_dir',type=Path);p.add_argument('--condition',choices=['aligned','easy','hard'],required=True);a=p.parse_args();out=a.run_dir.resolve()
 lock=(out/'qualification.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 if (out/'qualification_status.json').exists():
  prior=json.loads((out/'qualification_status.json').read_text());print(json.dumps(prior));return int(not prior['passed'])
 while not (out/'exit_status.json').exists():time.sleep(10)
 failures=[]
 status=json.loads((out/'exit_status.json').read_text())
 if status['returncode']!=0:failures.append('pilot execution did not complete successfully')
 path=out/'training_and_evaluation_results.json'
 if not path.exists():failures.append('missing final pilot results')
 if not failures:
  result=json.loads(path.read_text());cp=result['final_evaluation']['cp']
  if cp['goal_rate']<.90 or cp['unsafe_rate']!=0:failures.append('CP safety/goal criterion failed')
  if cp['infeasible_total']!=0 or cp['rail_fallback_total']!=0 or cp['min_active_feasibility_margin']<=0:failures.append('CP runtime feasibility criterion failed')
  for phase in ['pretraining','final_frozen_policy']:
   audit=json.loads((out/(phase+'_mondrian_calibration.json')).read_text())['offline_cp_qp_feasibility_audit']
   if audit['n_infeasible_grid_nodes'] or audit['n_uncalibrated_infeasible_grid_nodes'] or min(audit['min_feasibility_margin'],audit['min_uncalibrated_feasibility_margin'])<=0:failures.append(phase+' exhaustive criterion failed')
  if a.condition=='hard' and max(result['final_evaluation'][m]['unsafe_rate'] for m in ['nominal','cbvf'])<.10:failures.append('Hard baseline separation criterion failed: neither baseline reaches 10% unsafe')
 if not failures:
  certificate=audit_certificates(out)
  failures.extend(certificate['failures'])
 if not failures:
  env=dict(os.environ,MPLCONFIGDIR='/tmp/mondrian_matplotlib',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',PYTHONUNBUFFERED='1')
  interval_command=['/home/mars/miniconda3/envs/odp/bin/python3.8','experiments/audit_runtime_intervals.py','--run-dir',str(out)]
  with (out/'runtime_interval_audit.log').open('w') as log:r=subprocess.run(interval_command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
  if r.returncode!=0:failures.append('runtime interval reconstruction failed')
 if not failures:
  command=['/home/mars/miniconda3/envs/odp/bin/python3.8','experiments/evaluate_reference_coverage.py',str(out),'--episodes','100','--fresh-calibration','--output',str(out/'heldout_reference_coverage_fresh.json')]
  with (out/'heldout_coverage.log').open('w') as log:r=subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
  if r.returncode!=0:failures.append('held-out coverage execution failed')
  else:
   c=json.loads((out/'heldout_reference_coverage_fresh.json').read_text())
   if c['empirical_simultaneous_coverage']<.95 or not c['score_population_matches_calibration'] or c['disjointness']['overlap_pairs']!=0:failures.append('held-out coverage criterion failed')
 payload={'condition':a.condition,'passed':not failures,'failures':failures}
 (out/'qualification_status.json').write_text(json.dumps(payload,indent=2)+'\n');print(json.dumps(payload),flush=True)
 return int(bool(failures))
if __name__=='__main__':raise SystemExit(main())
