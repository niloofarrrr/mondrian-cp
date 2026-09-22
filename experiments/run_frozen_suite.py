#!/usr/bin/env python3
"""Bounded concurrent execution of a frozen suite with exclusive state ownership."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import datetime
import shutil
import run_full_suite as batch

def coverage_failure(result):
    """Apply the existing final coverage gate as soon as its result exists."""
    if result.get('episodes', 0) < 100:
        return 'Held-out coverage uses fewer than 100 episodes.'
    if not result.get('empirical_simultaneous_coverage', float('nan')) >= 0.95:
        return 'Held-out empirical coverage is below the required 0.95 threshold or invalid.'
    return None

def retire_failed_study(state, active):
    """Reject the entire seed block on failure, retaining every partial artifact."""
    if state.get('whole_study_rejection'):
        return True
    failed = [key for key, run in state['runs'].items() if run['status'] == 'failed']
    if not failed:
        return False
    state['whole_study_rejection'] = {
        'time': batch.now(), 'trigger_runs': failed,
        'reason': 'A required run failed; reject the whole frozen study and retire all reserved seeds.',
        'final_data_used_for_parameter_selection': False,
    }
    for run_id, (process, log, phase) in active.items():
        if process.poll() is None:
            state['runs'][run_id]['termination_reason'] = 'Entire frozen study rejected after a required run failed.'
            log.write('\nWhole-study rejection: preserving partial artifacts and terminating this job.\n')
            log.flush()
            process.terminate()
    return True

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--manifest',required=True)
    p.add_argument('--freeze',required=True)
    p.add_argument('--workers',type=int,default=3)
    a=p.parse_args()
    if not 1<=a.workers<=4: p.error('workers must be between 1 and 4')
    manifest_path=Path(a.manifest).resolve(); freeze=json.loads(Path(a.freeze).read_text())
    manifest=json.loads(manifest_path.read_text())
    if batch.config_hash(manifest)!=freeze['manifest_sha256']:
        raise RuntimeError('Manifest differs from frozen configuration')
    effective_path=Path(a.freeze).resolve().parent/'effective_parameters.json'
    if batch.config_hash(json.loads(effective_path.read_text()))!=freeze['effective_parameters_sha256']:
        raise RuntimeError('Effective parameters differ from freeze')
    for name,digest in freeze['source_hashes'].items():
        if hashlib.sha256((batch.ROOT/name).read_bytes()).hexdigest()!=digest:
            raise RuntimeError('Source differs from freeze: '+name)
    batch.configure_output_paths(manifest)
    batch.RESULTS.mkdir(parents=True,exist_ok=True); batch.LOG_DIR.mkdir(parents=True,exist_ok=True)
    lock=(batch.RESULTS/'suite.lock').open('a'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state=batch.initialize(manifest,manifest_path=manifest_path)
    state['status']='running'; state['frozen_manifest']=str(Path(a.freeze).resolve())
    pending=[k for k in state['order'] if state['runs'][k]['status']=='pending']
    active={}; stopping=[False]
    def stop(*unused): stopping[0]=True
    signal.signal(signal.SIGTERM,stop); signal.signal(signal.SIGINT,stop)
    env=dict(os.environ,MPLCONFIGDIR='/tmp/mondrian_matplotlib',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',PYTHONUNBUFFERED='1')
    def spawn(run_id,command,phase):
        r=state['runs'][run_id]; log=open(r['log'],'a')
        log.write('\nFrozen suite phase: '+phase+'\n'); log.flush()
        process=subprocess.Popen(command,cwd=batch.ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        active[run_id]=(process,log,phase); r['status']='running'; r['phase']=phase
    last_signature=None
    stopping[0] = retire_failed_study(state, active)
    while pending or active:
        while pending and len(active)<a.workers and not stopping[0]:
            run_id=pending.pop(0); r=state['runs'][run_id]
            out=Path(r['result_paths']['result']).parent
            if out.exists() and any(out.iterdir()):
                archive=batch.RESULTS/'attempts'/run_id/datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
                archive.parent.mkdir(parents=True,exist_ok=True); shutil.move(str(out),str(archive))
                r.setdefault('preserved_attempts',[]).append(str(archive))
            out.mkdir(parents=True,exist_ok=True);r['start_time']=batch.now()
            (out/'batch_configuration.json').write_text(json.dumps({'configuration':r['configuration'],'command':r['command'],'config_sha256':r['config_sha256'],'source_hashes':freeze['source_hashes']},indent=2)+'\n')
            spawn(run_id,r['command'],'training_and_evaluation')
        for run_id,(process,log,phase) in list(active.items()):
            code=process.poll()
            if code is None: continue
            log.close(); del active[run_id];r=state['runs'][run_id]
            out=Path(r['result_paths']['result']).parent
            if code==0 and phase=='training_and_evaluation' and (out/'training_and_evaluation_results.json').is_file() and r['configuration']['training_shield_method']=='cp':
                spawn(run_id,[manifest['python'],'experiments/evaluate_reference_coverage.py',str(out),'--episodes',str(manifest['heldout_coverage_episodes']),'--fresh-calibration','--output',str(out/'heldout_reference_coverage_fresh.json')],'heldout_coverage')
                continue
            required=out/('heldout_reference_coverage_fresh.json' if phase=='heldout_coverage' else 'training_and_evaluation_results.json')
            r.update(exit_code=code,completion_time=batch.now(),status='completed' if code==0 and required.is_file() else 'failed')
            if r['status']=='failed': r['error']='Failed phase '+phase+'; see preserved log and artifacts'
            elif phase == 'heldout_coverage':
                failure = coverage_failure(json.loads(required.read_text()))
                if failure:
                    r.update(status='failed', error=failure)
        if retire_failed_study(state, active):
            stopping[0] = True
        state['active_run']=', '.join(active) or None
        signature=tuple((k,r['status'],r.get('phase')) for k,r in state['runs'].items())
        if signature!=last_signature:
            batch.save_state(state)
            last_signature=signature
        if stopping[0] and not active: break
        if active: time.sleep(1)
    state['status']='completed' if all(r['status']=='completed' for r in state['runs'].values()) else ('failed' if state.get('whole_study_rejection') else ('interrupted' if stopping[0] else 'failed'))
    state['active_run']=None;batch.save_state(state);batch.write_final_report(state)
    return 0 if state['status']=='completed' else 1

if __name__=='__main__': raise SystemExit(main())
