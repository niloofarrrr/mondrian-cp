"""Read the actual CLI parser without importing ML libraries or running training."""
import argparse
import ast
import math
from pathlib import Path


def training_parser(source):
    tree = ast.parse(Path(source).read_text())
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'main')
    statements = []
    for node in main.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'args' for t in node.targets):
            break
        statements.append(node)
    namespace = {'argparse': argparse, 'math': math}
    exec(compile(ast.Module(body=statements, type_ignores=[]), str(source), 'exec'), namespace)
    return namespace['parser']


def effective_parameters(manifest, batch):
    parser = training_parser(batch.ROOT/'train/train_sac_lag.py')
    batch.configure_output_paths(manifest)
    rows = {}
    for run_id, condition, seed, out, command, digest, payload in batch.expanded_runs(manifest):
        args = vars(parser.parse_args(command[2:]))
        if (args['allow_infeasible_cp_diagnostic_run'] or args['allow_coarse_fallback'] or
                not args['require_active_mondrian'] or not args['intersample_protection'] or
                not args['recompute_cbvf'] or args['feasibility_only']):
            raise ValueError('A frozen run contains a diagnostic override or lacks mandatory protection: '+run_id)
        if args['max_steps'] != 250000 or args['eval_episodes'] < 100:
            raise ValueError('Frozen final training/evaluation budget differs from the protocol')
        rows[run_id] = args
    return dict(note='Effective argparse values, including defaults, before runtime-derived paths/bounds; all simulation/training sources are separately hash-frozen.', runs=rows)
