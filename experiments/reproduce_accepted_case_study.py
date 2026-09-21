"""Accepted-study-only replay/documentation helper, added September 21, 2026.

This was NOT the historical experiment launcher. It invokes the archived
commands with new output paths, or rebuilds paper outputs from copied raw data.
The accepted study and the historical design queue are never modified.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
ACCEPTED = ROOT / 'results_mechanism_v6/study_001'
CONDITIONS = ('aligned', 'easy', 'hard')
ENV = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
           MPLBACKEND='Agg', MPLCONFIGDIR='/tmp/mechanism-v6-matplotlib')


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def verify():
    freeze = read(ACCEPTED / 'FREEZE.json')
    assert read(ACCEPTED / 'SCIENTIFIC_ACCEPTANCE.json')['passed']
    assert not (ACCEPTED / 'RETIRED.json').exists()
    for name, digest in freeze['sha256'].items():
        if hashlib.sha256((ACCEPTED / name).read_bytes()).hexdigest() != digest:
            raise RuntimeError('Frozen input changed: ' + name)
    for seed in freeze['final_seeds']:
        actor = ACCEPTED / 'actors' / ('seed_%d' % seed) / 'params_20000.pkl'
        digest = hashlib.sha256(actor.read_bytes()).hexdigest()
        for ci, condition in enumerate(CONDITIONS):
            folder = ACCEPTED / 'runs' / ('%s_seed%d' % (condition, seed))
            meta = read(folder / 'configuration.json')
            assert meta['actor_sha256'] == digest
            assert meta['seed'] == seed + 200000 + ci * 100000
            assert meta['n_calibration'] == 199 and meta['delta'] == .01
            assert read(folder / 'qualification.json')['accepted']
            assert read(folder / 'independent_interval_audit.json')['passed']
    return freeze['final_seeds']


def execute(command, log, dry_run=False):
    print(shlex.join([str(x) for x in command]), flush=True)
    if not dry_run:
        with log.open('w') as stream:
            subprocess.run([str(x) for x in command], cwd=str(ROOT), env=ENV,
                           stdout=stream, stderr=subprocess.STDOUT, check=True)


def prepare(destination, copy_outputs=False):
    destination.mkdir(parents=True, exist_ok=False)
    for name in ['source_snapshot', 'postprocessing_source']:
        shutil.copytree(ACCEPTED / name, destination / name)
    for name in ['FREEZE.json', 'effective_configuration.json', 'environment.json',
                 'ANALYTIC_CERTIFICATE_DERIVATION.md']:
        shutil.copyfile(ACCEPTED / name, destination / name)
    if copy_outputs:
        for name in ['actors', 'runs']:
            shutil.copytree(ACCEPTED / name, destination / name)
        shutil.copyfile(ACCEPTED / 'final_gate.json', destination / 'final_gate.json')


def training_command(python, destination, seed):
    return [python, destination / 'source_snapshot/experiments/train_analytic_actor.py',
            '--configuration', destination / 'effective_configuration.json',
            '--out', destination / 'actors' / ('seed_%d' % seed), '--seed', seed]


def condition_commands(python, source, destination, seed, condition, actor_root):
    script = source / 'source_snapshot/experiments'
    output = destination / 'runs' / ('%s_seed%d' % (condition, seed))
    evaluation_seed = seed + 200000 + CONDITIONS.index(condition) * 100000
    evaluation = [python, script / 'qualify_analytic_cbvf.py', '--configuration',
                  source / 'effective_configuration.json', '--out', output,
                  '--seed', evaluation_seed, '--condition', condition, '--actor',
                  actor_root / ('seed_%d' % seed) / 'params_20000.pkl']
    return output, evaluation, [python, script / 'audit_analytic_intervals.py', output]


def supplement(destination):
    """Recover the recorded inline runtime-table/report-addendum calculation."""
    rows = []
    seeds = read(destination / 'FREEZE.json')['final_seeds']
    for condition in CONDITIONS:
        for method, names in [('cbvf', ['calibration', 'performance_cbvf', 'heldout_reference']),
                              ('cp', ['performance_cp', 'heldout_deployment'])]:
            records = [read(destination / 'runs' / ('%s_seed%d' % (condition, seed)) /
                            (name + '.json')) for seed in seeds for name in names]
            row = dict(condition=condition, method=method,
                       rollouts=sum(len(v['rollouts']) for v in records),
                       active_node_checks=sum(v['interval_checks'] for v in records),
                       minimum_available_margin=min(r['min_qp'] for v in records for r in v['rollouts']),
                       minimum_achieved_constraint_residual=min(v['min_constraint_margin'] for v in records),
                       minimum_true_interval_bound=min(v['min_true_interval_margin'] for v in records))
            assert row['minimum_available_margin'] > 0
            assert row['minimum_achieved_constraint_residual'] >= -1e-10
            if method == 'cp':
                assert row['minimum_true_interval_bound'] > 0
            rows.append(row)
    path = destination / 'tables/runtime_feasibility.csv'
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    header = '| ' + ' | '.join(rows[0]) + ' |\n| ' + ' | '.join(['---'] * len(rows[0])) + ' |\n'
    body = '\n'.join('| ' + ' | '.join(str(v) for v in row.values()) + ' |' for row in rows)
    addition = ('## Executed-node feasibility and interval checks\n\n' + header + body +
                '\n\nThe achieved QP residual may be zero at a binding projection; residuals of order 1e-16 are floating-point roundoff. Available control margins are strictly positive. Negative true derivative bounds for Hard uncalibrated CBVF are the model-mismatch mechanism, not a model-QP infeasibility.\n\n')
    for name in ['PAPER_NUMBERS.md', 'PAPER_HANDOFF.md']:
        path = destination / name
        text = path.read_text().replace('## FILES TO OPEN FOR WRITING THE CASE STUDY',
                                        addition + '## FILES TO OPEN FOR WRITING THE CASE STUDY')
        path.write_text(text + '- `' + str(destination / 'tables/runtime_feasibility.csv') + '`\n')
    index = read(destination / 'ARTIFACT_INDEX.json')
    index['paper_numbers'] = str(destination / 'PAPER_NUMBERS.md')
    index['tables'] = sorted(set(index['tables'] + [str(destination / 'tables/runtime_feasibility.csv')]))
    index['aggregate'] = sorted(set(index['aggregate'] + [str(destination / 'aggregate/hard_same_state_mechanism.csv')]))
    index['analytic_derivation'] = str(destination / 'ANALYTIC_CERTIFICATE_DERIVATION.md')
    write(destination / 'ARTIFACT_INDEX.json', index)


def paper(python, destination, dry_run):
    for relative, log in [
        ('source_snapshot/experiments/build_analytic_report.py', 'build_report.log'),
        ('postprocessing_source/extract_analytic_paper_numbers.py', 'extract_paper_numbers.log'),
        ('postprocessing_source/finalize_analytic_figures.py', 'finalize_figures.log')]:
        execute([python, destination / relative, destination], destination / log, dry_run)
    if not dry_run:
        supplement(destination)
        assert read(destination / 'SCIENTIFIC_ACCEPTANCE.json')['passed']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=['verify', 'full', 'condition', 'paper'])
    parser.add_argument('--out', type=Path)
    parser.add_argument('--condition', choices=CONDITIONS)
    parser.add_argument('--actor-seed', type=int, help='Omit to reproduce all five accepted actor seeds.')
    parser.add_argument('--reuse-actors', action='store_true', help='Full replay: copy accepted checkpoints instead of retraining.')
    parser.add_argument('--dry-run', action='store_true', help='Verify inputs and print commands; write nothing.')
    args = parser.parse_args()
    seeds = verify()
    python = Path(read(ACCEPTED / 'environment.json')['executable'])
    if not python.is_file():
        parser.error('Recorded Python executable is missing: ' + str(python))
    if args.operation == 'verify':
        print('PASS: frozen inputs, five actor checkpoints, 15 condition blocks and interval audits verified.')
        return
    if args.out is None:
        parser.error('--out must be a new directory')
    destination = args.out.resolve()
    if destination == ACCEPTED or ACCEPTED in destination.parents or destination in ACCEPTED.parents:
        parser.error('Output must be separate from the accepted study and its ancestors.')
    if destination.exists():
        parser.error('Output already exists; choose a new directory to preserve previous outputs.')
    if args.operation == 'condition' and not args.condition:
        parser.error('condition requires --condition')
    if args.actor_seed is not None and args.actor_seed not in seeds:
        parser.error('--actor-seed must be one of the five accepted seeds')
    if args.operation != 'condition' and args.actor_seed is not None:
        parser.error('--actor-seed applies only to condition')
    if args.operation != 'full' and args.reuse_actors:
        parser.error('--reuse-actors applies only to full')
    if not args.dry_run:
        if args.operation == 'condition':
            (destination / 'runs').mkdir(parents=True)
        else:
            prepare(destination, copy_outputs=args.operation == 'paper')
    if args.operation == 'paper':
        paper(python, destination, args.dry_run)
        return
    if args.operation == 'full':
        if not args.dry_run:
            if args.reuse_actors:
                shutil.copytree(ACCEPTED / 'actors', destination / 'actors')
            else:
                (destination / 'actors').mkdir()
            (destination / 'runs').mkdir()
        if not args.reuse_actors:
            def train(seed):
                execute(training_command(python, destination, seed), destination / 'actors' / ('seed_%d.log' % seed), args.dry_run)
            with ThreadPoolExecutor(max_workers=1 if args.dry_run else 3) as pool:
                list(pool.map(train, seeds))
    selected = [args.actor_seed] if args.actor_seed is not None else seeds
    conditions = [args.condition] if args.operation == 'condition' else CONDITIONS
    source = ACCEPTED if args.operation == 'condition' else destination
    actor_root = source / 'actors'
    def run(spec):
        seed, condition = spec
        output, evaluation, audit = condition_commands(python, source, destination, seed, condition, actor_root)
        execute(evaluation, output.with_suffix('.log'), args.dry_run)
        if not args.dry_run and not read(output / 'qualification.json')['accepted']:
            raise RuntimeError('Replay block failed acceptance: ' + str(output))
        execute(audit, output.with_suffix('.interval_audit.log'), args.dry_run)
        return read(output / 'qualification.json') if not args.dry_run else None
    with ThreadPoolExecutor(max_workers=1 if args.dry_run else 3) as pool:
        results = list(pool.map(run, [(seed, c) for seed in selected for c in conditions]))
    if args.operation == 'full':
        if not args.dry_run:
            write(destination / 'final_gate.json', dict(passed=True, condition_seed_blocks=15,
                                                       evaluation_runs=45, results=results))
        paper(python, destination, args.dry_run)


if __name__ == '__main__':
    main()
