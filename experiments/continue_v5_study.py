#!/usr/bin/env python3
"""Persist the pilot-replication, freeze, final-suite and reporting workflow.

Only design seeds are used before every qualification gate passes. A failed
frozen final study is reported, never used to select another configuration.
"""
import concurrent.futures
import datetime
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'results_intersample_v5'
OUT = RESULTS / 'autonomous_continuation'
PYTHON = '/home/mars/miniconda3/envs/odp/bin/python3.8'
# Prefer a smaller sufficient mismatch over changing shared task geometry or dt.
CANDIDATES = [
    'learned_hard_shortgoal_beta050_wp090_x090_seed620560',
    'learned_hard_shortgoal_beta048_wp095_x090_seed620561',
    'learned_hard_shortgoal_beta050_reverse020_seed620562',
    'learned_hard_shortgoal_beta050_wp095_reverse020_seed620563',
]
EXISTING = {
    'aligned': 'learned_aligned_outward_wp085_gamma15_dt002_seed620503',
    'easy': 'learned_easy_outward_wp085_gamma15_dt002_seed620504',
}
ENV = dict(os.environ, MPLCONFIGDIR='/tmp/mondrian_matplotlib',
           OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', PYTHONUNBUFFERED='1')


def write_status(phase, **details):
    payload = dict(phase=phase, updated_utc=datetime.datetime.now(
        datetime.timezone.utc).isoformat(), **details)
    temporary = OUT / 'status.tmp'
    temporary.write_text(json.dumps(payload, indent=2) + '\n')
    temporary.replace(OUT / 'status.json')
    print(json.dumps(payload), flush=True)


def command(args, log_name):
    with (OUT / log_name).open('a') as log:
        log.write('\nCOMMAND ' + json.dumps(args) + '\n'); log.flush()
        return subprocess.run(args, cwd=ROOT, env=ENV, stdout=log,
                              stderr=subprocess.STDOUT).returncode


def configuration(folder):
    return json.loads((folder / 'batch_configuration.json').read_text())['configuration']


def qualified(folder):
    return json.loads((folder / 'qualification_status.json').read_text())['passed']


def replica_args(name, seed, cfg, condition):
    beta, vmin = {'aligned': (.8, -.2), 'easy': (.625, -.15),
                  'hard': (cfg['beta_u'], cfg['speed_min'])}[condition]
    return [sys.executable, 'experiments/run_v3_candidate.py', name,
            '--seed', str(seed), '--gamma', str(cfg['gamma']),
            '--beta', str(beta), '--speed-min', str(vmin),
            '--activation', str(cfg['cp_activate_margin']),
            '--initial-x', str(cfg['initial_x']),
            '--initial-theta', str(cfg['initial_theta']),
            '--duration', str(cfg['dt'] * cfg['horizon']), '--dt', str(cfg['dt']),
            '--waypoint-x', str(cfg['reference_waypoint_x']),
            '--waypoint-y', str(cfg['reference_waypoint_y']),
            '--orbit-trigger-radius', str(cfg.get('reference_orbit_trigger_radius', 0.0)),
            '--orbit-radius', str(cfg.get('reference_orbit_radius', 1.0)),
            '--orbit-gain', str(cfg.get('reference_orbit_gain', 2.0)),
            '--goal-x', str(cfg['goal_x']), '--goal-y', str(cfg['goal_y']),
            '--n-calib', str(cfg['n_calib']),
            '--clearance-edges', str(cfg['mondrian_clearance_edges']),
            '--pilot', '--steps', '20000', '--episodes', '100']


def run_replica(name, seed, cfg, condition):
    folder = RESULTS / name
    if (folder / 'qualification_status.json').exists():
        return folder, qualified(folder)
    if not folder.exists():
        command(replica_args(name, seed, cfg, condition), name + '.log')
    elif not (folder / 'exit_status.json').exists():
        raise RuntimeError('Existing unfinished replica needs explicit recovery: ' + str(folder))
    command([sys.executable, 'experiments/qualify_pilot_coverage.py',
             str(folder), '--condition', condition], name + '_qualification.log')
    return folder, qualified(folder)


def main():
    from freeze_v5_configuration import CORE
    OUT.mkdir(exist_ok=True)
    lock = (OUT / 'workflow.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    frozen = RESULTS / 'frozen_study'
    if not (frozen / 'FREEZE.json').exists():
        selected = None
        for index, name in enumerate(CANDIDATES):
            candidate = RESULTS / name
            write_status('waiting_for_design_qualification', candidate=name)
            while not (candidate / 'qualification_status.json').exists():
                time.sleep(10)
            if not qualified(candidate):
                continue
            cfg = configuration(candidate)
            pilots = [('hard', candidate)]
            jobs = [('hard', 'replicate_hard_shortgoal_candidate%d_seed%d' % (index, 620570 + index), 620570 + index)]
            for offset, condition in enumerate(['aligned', 'easy']):
                old = RESULTS / EXISTING[condition]
                old_cfg = configuration(old)
                if qualified(old) and all(cfg.get(k) == old_cfg.get(k) for k in CORE):
                    pilots.append((condition, old))
                else:
                    seed = 620580 + 10 * index + offset
                    jobs.append((condition, 'qualify_%s_shortgoal_candidate%d_seed%d' % (condition, index, seed), seed))
            write_status('replicating_design', candidate=name, jobs=jobs)
            # A qualified initial pilot has released a worker. Reuse that slot
            # without waiting for slower rejected/alternative designs.
            unfinished_initial = sum(not (RESULTS / other / 'qualification_status.json').exists()
                                     for other in CANDIDATES)
            workers = max(1, min(3, 4 - unfinished_initial))
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
                futures = [(condition, pool.submit(run_replica, run_name, seed, cfg, condition))
                           for condition, run_name, seed in jobs]
                passed = True
                for condition, future in futures:
                    folder, ok = future.result()
                    passed = passed and ok
                    pilots.append((condition, folder))
            if not passed:
                write_status('design_replication_failed', candidate=name)
                continue
            args = [sys.executable, 'experiments/freeze_v5_configuration.py']
            for condition, folder in pilots:
                args += ['--pilot', condition + ':' + str(folder)]
            if command(args, 'freeze.log') != 0:
                raise RuntimeError('Freeze validation rejected qualification; inspect freeze.log')
            selected = name
            break
        if selected is None:
            write_status('additional_design_required', final_seeds_used=False)
            return 2
    write_status('frozen_final_suite_running', freeze=str(frozen / 'FREEZE.json'))
    suite_code = command([sys.executable, 'experiments/run_frozen_suite.py',
                          '--manifest', str(frozen / 'manifest.json'),
                          '--freeze', str(frozen / 'FREEZE.json'), '--workers', '4'], 'final_suite.log')
    suite = frozen / 'final_suite'
    codes = {}
    for phase, args in [
        ('aggregate', [PYTHON, 'experiments/aggregate_full_suite.py', '--manifest', str(frozen / 'manifest.json')]),
        ('audit', [sys.executable, 'experiments/audit_full_suite.py', '--results-dir', str(suite)]),
        ('paper_report', [sys.executable, 'experiments/build_v3_paper_report.py', str(suite)]),
    ]:
        write_status('final_' + phase, suite_returncode=suite_code)
        codes[phase] = command(args, phase + '.log')
    audit_path = suite / 'aggregate/acceptance_audit.json'
    accepted = audit_path.exists() and json.loads(audit_path.read_text())['passed']
    success = suite_code == 0 and all(code == 0 for code in codes.values()) and accepted
    write_status('paper_results_pass' if success else 'frozen_final_study_failed',
                 suite_returncode=suite_code, report_returncodes=codes,
                 acceptance_passed=bool(accepted), results=str(suite),
                 final_data_used_for_retuning=False)
    return 0 if success else 1


if __name__ == '__main__':
    raise SystemExit(main())
