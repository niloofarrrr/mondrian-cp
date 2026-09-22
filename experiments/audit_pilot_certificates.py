#!/usr/bin/env python3
"""Check recorded numerical certificates without changing their acceptance gates.

This verifies interval-bound provenance and arithmetic, not convergence to the
continuous PDE or continuous-state QP feasibility between audited grid nodes.
"""
import argparse
import json
import math
from pathlib import Path


def audit_certificates(folder):
    folder = Path(folder)
    cfg = json.loads((folder / 'batch_configuration.json').read_text())['configuration']
    failures, rows = [], []
    cx = math.hypot(max(abs(cfg['speed']), abs(cfg['speed_min'])), abs(cfg['beta_u']))
    factor = (cx + 1) * cfg['dt']
    mismatch = any(cfg[truth] != cfg[model] for truth, model in (
        ('speed','cbvf_model_speed'), ('speed_min','cbvf_model_speed_min'), ('beta_u','cbvf_model_beta_u')))
    for phase in ('pretraining', 'final_frozen_policy'):
        path = folder / (phase + '_mondrian_calibration.json')
        if not path.exists():
            failures.append(phase + ': missing calibration/certificate')
            continue
        data = json.loads(path.read_text())
        audit = data['offline_cp_qp_feasibility_audit']
        errors = []
        def check(ok, message):
            if not ok:
                errors.append(message)
        def close(a, b):
            return math.isfinite(a) and math.isfinite(b) and math.isclose(a, b, rel_tol=1e-8, abs_tol=1e-12)
        check(data['intersample_protection'] and audit['intersample_protection'], 'inter-sample protection disabled')
        check(close(data['C_x'], cx) and close(audit['C_x'], cx), 'C_x differs from true dynamics bound')
        lip = data['psi_lipschitz']
        check(lip.get('uniform_over_normalized_control_box') is True and lip.get('uses_rollout_data') is False, 'invalid interval-bound provenance')
        check(close(lip['position_radius'], max(abs(cfg['speed']), abs(cfg['speed_min'])) * cfg['dt']), 'position tube radius mismatch')
        check(close(lip['heading_radius'], abs(cfg['beta_u']) * cfg['dt']), 'heading tube radius mismatch')
        check(audit['runtime_decision_intervals_audited'] == cfg['horizon'], 'not every decision interval audited')
        check(audit['n_active_certified_grid_nodes'] > 0, 'empty audit domain')
        for key in ('n_infeasible_grid_nodes', 'n_uncalibrated_infeasible_grid_nodes'):
            check(audit[key] == 0, key + ' nonzero')
        for key in ('min_feasibility_margin', 'min_uncalibrated_feasibility_margin'):
            check(math.isfinite(audit[key]) and audit[key] > 0, key + ' not strictly positive')
        for key, value in audit['L_Psi_distribution'].items():
            check(value >= 0 and close(audit['epsilon_inter_distribution'][key], factor * value), 'epsilon formula fails for ' + key)
        worst = audit['worst_active_certified_grid_node']
        check(close(worst['epsilon_inter'], factor * worst['L_Psi_j']), 'worst-node epsilon formula mismatch')
        cal = data['mondrian_calibration']
        buffers = [r['xi_hat_off'] for r in cal['regions']]
        check(all(math.isfinite(b) and b >= 0 for b in buffers), 'invalid conformal buffers')
        if mismatch:
            check(max(buffers) > 1e-8, 'trivial conformal buffers in a mismatch condition')
        check(close(cal['delta_traj'], cfg['delta']) and close(cal['risk_budget_sum'], cfg['delta']), 'risk budget changed')
        check(cal['tree_cut_is_partition'], 'invalid conformal partition')
        failures.extend(phase + ': ' + error for error in errors)
        rows.append(dict(phase=phase, passed=not errors, C_x=cx,
                         L_Psi_distribution=audit['L_Psi_distribution'],
                         epsilon_inter_distribution=audit['epsilon_inter_distribution'],
                         buffers=buffers, nontrivial_buffer_required=mismatch,
                         audit_nodes=audit['n_active_certified_grid_nodes'],
                         min_calibrated_margin=audit['min_feasibility_margin'],
                         min_uncalibrated_margin=audit['min_uncalibrated_feasibility_margin']))
    payload = dict(passed=not failures, failures=failures, phases=rows,
                   scope='Implemented numerical Psi interval certificates and exhaustive grid audits; not a continuous PDE convergence proof.')
    (folder / 'certificate_acceptance.json').write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')
    return payload


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('run_dir')
    args = parser.parse_args()
    result = audit_certificates(args.run_dir)
    print(json.dumps(result, indent=2))
    raise SystemExit(int(not result['passed']))
