#!/usr/bin/env python3
"""Persistent, model-independent design/qualification/freeze/final workflow.

Failed pilots advance a prospectively recorded design queue. Failed final
studies retire their full seed blocks; final metrics never select parameters.
Every subprocess, attempt, and rejection is retained. A process lock prevents
duplicate supervisors. Restarting resumes recorded work without erasing it.
"""
import concurrent.futures
import datetime
import fcntl
import hashlib
import itertools
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

from continue_v5_study import replica_args

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'results_intersample_v5'
OUT = BASE / 'persistent_workflow'
PYTHON = '/home/mars/miniconda3/envs/odp/bin/python3.8'
ENV = dict(os.environ, MPLCONFIGDIR='/tmp/mondrian_matplotlib',
           OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', PYTHONUNBUFFERED='1')


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def atomic(path, payload):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def status(phase, **details):
    payload = dict(phase=phase, updated_utc=now(), pid=os.getpid(), **details)
    atomic(OUT / 'status.json', payload)
    print(json.dumps(payload), flush=True)


def event(kind, **details):
    with (OUT / 'events.jsonl').open('a') as file:
        file.write(json.dumps(dict({'time': now(), 'kind': kind}, **details)) + '\n')


def command(args, label):
    path = OUT / 'logs' / (label + '.log')
    with path.open('a') as log:
        log.write('\n' + now() + ' COMMAND ' + json.dumps(args) + '\n'); log.flush()
        process = subprocess.Popen(args, cwd=ROOT, env=ENV, stdout=log, stderr=subprocess.STDOUT)
        atomic(OUT / 'jobs' / (label + '.json'), dict(pid=process.pid, command=args, log=str(path), started=now()))
        code = process.wait()
        atomic(OUT / 'jobs' / (label + '.json'), dict(pid=process.pid, command=args, log=str(path), finished=now(), returncode=code))
    return code


def read(path):
    return json.loads(path.read_text())


def make_plan():
    source = BASE / 'learned_hard_orbit080_beta043_reverse015_seed620621'
    cfg = read(source / 'batch_configuration.json')['configuration']
    plans = [dict(name=source.name, seed=620621, configuration=cfg,
                  replica_seeds=[620631, 620650, 620651],
                  rationale='Existing orbit pilot; fresh independent replication and all matching conditions are required.')]
    neighbors = [(.43,-.175,.8,-.9,1.5,12.,0.,1.8),
                 (.44,-.175,.8,-.9,1.5,12.,0.,1.8),
                 (.43,-.2,.8,-.9,1.5,12.,0.,1.8),
                 (.43,-.15,.825,-.9,1.5,12.,0.,1.8),
                 (.42,-.175,.825,-.9,1.5,12.,0.,1.8)]
    grid = itertools.product((.43,.44,.45,.42), (-.15,-.175,-.125,-.2),
                             (.8,.825,.85,.775), (-.9,-.8,-1.),
                             (1.5,1.25), (12.,14.), (0.,), (1.8,1.5))
    seen = {(.43,-.15,.8,-.9,1.5,12.,0.,1.8)}
    for item in itertools.chain(neighbors, grid):
        if item in seen:
            continue
        seen.add(item)
        beta, reverse, trigger, x, gamma, duration, gx, gy = item
        index = len(plans) - 1
        seed = 622000 + 4 * index
        candidate = dict(cfg, seed=seed, beta_u=beta, speed_min=reverse,
                         reference_orbit_trigger_radius=trigger, initial_x=x,
                         gamma=gamma, horizon=round(duration/cfg['dt']), goal_x=gx, goal_y=gy)
        plans.append(dict(name='autonomous_orbit_%04d_seed%d' % (index, seed), seed=seed,
                          configuration=candidate, replica_seeds=[seed+1,seed+2,seed+3],
                          rationale='Prospective shared-geometry/steering/reverse-authority refinement; fixed method, delta, quantile, residual range, physical safety definition and qualification gates.'))
    return dict(created_utc=now(), final_data_used_for_selection=False,
                selection='First configuration passing independent design qualification in this fixed order.',
                final_failure_rule='Retire complete final seed block; advance to the next prospectively listed design without reading final metrics for parameter selection.',
                candidates=plans)


def lock_busy(path):
    if not path.exists():
        return False
    with path.open('a') as file:
        try:
            fcntl.flock(file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
    return False


def ensure_pilot(name, seed, cfg, condition):
    folder = BASE / name
    if not folder.exists():
        command(replica_args(name, seed, cfg, condition), name)
    while not (folder / 'exit_status.json').exists():
        if not lock_busy(folder / 'run.lock'):
            raise RuntimeError('An interrupted pilot needs recovery with all prior artifacts preserved: ' + str(folder))
        time.sleep(10)
    qualification = folder / 'qualification_status.json'
    while not qualification.exists() and lock_busy(folder / 'qualification.lock'):
        time.sleep(10)
    if not qualification.exists():
        command([sys.executable, 'experiments/qualify_pilot_coverage.py', str(folder),
                 '--condition', condition], name + '_qualification')
    if not qualification.exists():
        raise RuntimeError('Qualification did not write an outcome; inspect log: ' + name)
    return folder, read(qualification)['passed']


def final_seeds(study_number):
    if study_number == 0:
        return [11000001,21000001,31000001,41000001,51000001]
    return [61000001 + (study_number-1)*5000000 + j*1000000 for j in range(5)]


def run_final(folder):
    manifest, freeze = folder/'manifest.json', folder/'FREEZE.json'
    suite = folder/'final_suite'
    label = folder.name
    status('frozen_final_suite_running', freeze=str(freeze), final_seeds=read(freeze)['reserved_final_seeds'])
    code = command([sys.executable, 'experiments/run_frozen_suite.py', '--manifest', str(manifest),
                    '--freeze', str(freeze), '--workers', '4'], label+'_suite')
    codes = {'suite': code}
    stages = [
        ('static_workflow', [PYTHON, 'train/check_shielded_training_workflow.py', '--configuration', str(manifest),
                             '--shield-sha256', read(freeze)['source_hashes']['conformal_shield.py']]),
        ('aggregate', [PYTHON, 'experiments/aggregate_full_suite.py', '--manifest', str(manifest)]),
        ('acceptance', [sys.executable, 'experiments/audit_full_suite.py', '--results-dir', str(suite)]),
        ('paper_report', [sys.executable, 'experiments/build_v3_paper_report.py', str(suite)]),
    ]
    for phase, args in stages:
        status('final_'+phase, study=str(folder), suite_returncode=code)
        codes[phase] = command(args, label+'_'+phase)
    audit = suite/'aggregate/acceptance_audit.json'
    passed = all(value == 0 for value in codes.values()) and audit.exists() and read(audit)['passed']
    outcome = dict(passed=bool(passed), returncodes=codes, completed_utc=now(),
                   final_data_used_for_parameter_selection=False, results=str(suite))
    atomic(folder/'STUDY_OUTCOME.json', outcome)
    event('final_study_outcome', study=str(folder), **outcome)
    return passed


def run_locked():
    OUT.mkdir(exist_ok=True)
    for name in ('logs', 'jobs', 'studies'):
        (OUT/name).mkdir(exist_ok=True)
    atomic(OUT/'process.json', dict(pid=os.getpid(), started=now(), source=str(Path(__file__).resolve())))
    plan_path = OUT/'design_plan.json'
    if not plan_path.exists():
        atomic(plan_path, make_plan())
    plan = read(plan_path)
    state_path = OUT/'progress.json'
    state = read(state_path) if state_path.exists() else dict(candidate_index=0, study_number=0, rejected=[], final_attempts=[])
    if state.get('completed'):
        status('paper_results_pass', results=state['results'])
        return 0
    if state.get('active_study'):
        folder = Path(state['active_study'])
        if run_final(folder):
            state.update(completed=True, results=str(folder/'final_suite'))
            atomic(state_path, state); status('paper_results_pass', results=state['results']); return 0
        state['final_attempts'].append(str(folder)); state.pop('active_study')
        state['candidate_index'] += 1; state['study_number'] += 1
        atomic(state_path, state)
    while state['candidate_index'] < len(plan['candidates']):
        index = state['candidate_index']; item = plan['candidates'][index]
        if shutil.disk_usage(BASE).free < 2*1024**3:
            status('external_storage_blocker', free_bytes=shutil.disk_usage(BASE).free,
                   explanation='Insufficient storage to preserve all attempts and run safely; existing evidence is retained.')
            return 3
        cfg = item['configuration']
        status('design_qualification', candidate_index=index, candidate=item['name'], rationale=item['rationale'])
        folder, passed = ensure_pilot(item['name'], item['seed'], cfg, 'hard')
        pilots = [('hard', folder)]
        if passed:
            jobs = [(condition, 'autonomous_replica_%s_seed%d' % (condition, seed), seed)
                    for condition, seed in zip(('hard','aligned','easy'), item['replica_seeds'])]
            status('independent_design_replication', candidate=item['name'], jobs=jobs)
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
                futures = [(condition, pool.submit(ensure_pilot, name, seed, cfg, condition))
                           for condition, name, seed in jobs]
                for condition, future in futures:
                    other, ok = future.result(); passed = passed and ok
                    pilots.append((condition, other))
        if not passed:
            rejection = dict(candidate=item['name'], time=now(), evidence=[str(p) for _,p in pilots])
            state['rejected'].append(rejection); event('design_rejected', **rejection)
            state['candidate_index'] += 1; atomic(state_path, state)
            continue
        folder = OUT/'studies'/('study_%03d' % state['study_number'])
        args = [sys.executable, 'experiments/freeze_v5_configuration.py', '--output', str(folder),
                '--final-seeds'] + [str(seed) for seed in final_seeds(state['study_number'])]
        for condition, pilot in pilots:
            args += ['--pilot', condition+':'+str(pilot)]
        status('freezing_qualified_design', candidate=item['name'], study=str(folder))
        if command(args, folder.name+'_freeze') != 0:
            raise RuntimeError('Freeze validation failed; no final run was launched.')
        state['active_study'] = str(folder); atomic(state_path, state)
        if run_final(folder):
            state.update(completed=True, results=str(folder/'final_suite'))
            atomic(state_path, state); status('paper_results_pass', results=state['results'])
            return 0
        state['final_attempts'].append(str(folder)); state.pop('active_study')
        state['study_number'] += 1; state['candidate_index'] += 1; atomic(state_path, state)
    status('design_queue_exhausted_needs_new_design_logic', attempted=len(plan['candidates']),
           explanation='This is not a validated result or an external blocker; further scientific design work is required.')
    return 2


def main():
    OUT.mkdir(exist_ok=True)
    with (OUT/'workflow.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return run_locked()


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception:
        error = traceback.format_exc()
        if OUT.exists():
            (OUT/'runner_error.txt').write_text(error)
            status('runner_error_needs_repair', traceback=error)
        raise
