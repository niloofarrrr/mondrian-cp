#!/usr/bin/env python3
"""Build a report solely from frozen configuration and audited raw results."""
import argparse
import csv
import json
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument('results_dir',type=Path)
a=p.parse_args()
root=a.results_dir.resolve()
state=json.loads((root/'run_state.json').read_text())
manifest=json.loads(Path(state['manifest_path']).read_text())
audit=json.loads((root/'aggregate/acceptance_audit.json').read_text())
metrics=list(csv.DictReader((root/'aggregate/final_metrics.csv').open()))
common=manifest['common']
lines=['# Frozen case-study results','','Acceptance: **'+('PASS' if audit['passed'] else 'FAIL')+'**.','',
       f"Completed runs: {sum(r['status']=='completed' for r in state['runs'].values())}/{len(state['runs'])}.",
       f"Final training seeds: {manifest['fixed_seeds']}. No final seed was used for design iteration.",'',
       '## Primary independently trained policy comparison','',
       'Rates below are cross-seed means; returns are mean ± sample SD. Counterfactual evaluations of another method’s actor are excluded from this table.','',
       '| Condition | Method | Goal rate | Unsafe rate | Return | Worst exhaustive margin |',
       '|---|---|---:|---:|---:|---:|']
for r in metrics:
    lines.append(f"| {r['condition']} | {r['method']} | {float(r['goal_rate_mean']):.1%} | {float(r['unsafe_rate_mean']):.1%} | {float(r['return_mean']):.2f} ± {float(r['return_sd']):.2f} | {float(r['offline_min_margin']):.6g} |")
lines += ['', '## Acceptance criteria', '', '| Criterion | Result |', '|---|---|']
for criterion, status in audit.get('acceptance_criteria', {}).items():
    lines.append('| ' + criterion.replace('_', ' ') + ' | ' + status + ' |')
lines += ['','## Coverage and feasibility','',
          f"Mean held-out CP reference coverage: {audit.get('cp_mean_coverage')}. Target: {1-float(common['delta']):.2%}.",
          'Fresh calibration and held-out scoring use the same frozen baseline-CBVF closed-loop population, regional scores, interval margins, and stopping rule. Each per-run coverage artifact includes counts, a Wilson interval, and initial-state disjointness checks. CP deployment changes the trajectory distribution; its coverage is a separate diagnostic, with no automatic transfer of the reference-population guarantee.',
          '', 'All reported exhaustive margins refer to the stored grid nodes within the declared analytic reach set, activation band, and B-invariant envelope. Positive grid margins are not described as an exhaustive proof over every continuous state; runtime feasibility and inter-sample safety observations are reported separately. Values and derivative fields are numerical interpolants: the audits certify the implemented numerical QP, not the exact continuous CBVF/PDE solution.','']
grid_audits=[]
coverage_rows=[]
for run_id, record in state['runs'].items():
    folder=Path(record['result_paths']['result']).parent
    for phase in ['pretraining','final_frozen_policy']:
        path=folder/(phase+'_mondrian_calibration.json')
        if path.exists():
            grid_audits.append(json.loads(path.read_text())['offline_cp_qp_feasibility_audit'])
    coverage_path=folder/'heldout_reference_coverage_fresh.json'
    result_path=folder/'training_and_evaluation_results.json'
    if record['configuration']['training_shield_method']=='cp' and coverage_path.exists() and result_path.exists():
        coverage=json.loads(coverage_path.read_text())
        cp=json.loads(result_path.read_text())['final_evaluation']['cp']
        interval=coverage['coverage_wilson_95']
        coverage_rows.append(
            f"| {record['condition']} | {record['seed']} | {cp['goal_rate']:.1%} | {cp['unsafe_rate']:.1%} | "
            f"{cp['min_active_feasibility_margin']:.6g} | {int(cp['infeasible_total'])} | "
            f"{coverage['covered_episodes']}/{coverage['episodes']} | {interval[0]:.1%}–{interval[1]:.1%} |")
if grid_audits:
    lines += [f"Recorded pretraining/final exhaustive audits: {len(grid_audits)}/{2*len(state['runs'])}; "
              f"total active grid-node/decision-interval evaluations: {sum(x['n_active_certified_grid_nodes'] for x in grid_audits):,}; "
              f"calibrated infeasible entries: {sum(x['n_infeasible_grid_nodes'] for x in grid_audits):,}; "
              f"uncalibrated infeasible entries: {sum(x['n_uncalibrated_infeasible_grid_nodes'] for x in grid_audits):,}.", '']
if coverage_rows:
    lines += ['Per-seed primary CP performance and separate held-out reference coverage:', '',
              '| Condition | Seed | CP goal | CP unsafe | Min runtime QP margin | Infeasible QPs | Reference covered | Coverage Wilson 95% interval |',
              '|---|---:|---:|---:|---:|---:|---:|---:|'] + coverage_rows + ['',
              'Coverage intervals describe each fitted reference population; a finite-sample point estimate above 95% does not prove conditional coverage for every fitted calibration set.', '']
certificate_rows = []
for run_id, record in state['runs'].items():
    folder = Path(record['result_paths']['result']).parent
    path = folder / 'certificate_acceptance.json'
    if not path.exists():
        continue
    cert = json.loads(path.read_text())
    for phase in cert['phases']:
        certificate_rows.append(dict(run_id=run_id, phase=phase['phase'],
            passed=phase['passed'], C_x=phase['C_x'], dt=common['dt'],
            L_Psi_min=phase['L_Psi_distribution']['min'],
            L_Psi_max=phase['L_Psi_distribution']['max'],
            epsilon_inter_min=phase['epsilon_inter_distribution']['min'],
            epsilon_inter_max=phase['epsilon_inter_distribution']['max'],
            conformal_buffers=json.dumps(phase['buffers']),
            audit_nodes=phase['audit_nodes'],
            min_calibrated_margin=phase['min_calibrated_margin'],
            min_uncalibrated_margin=phase['min_uncalibrated_margin']))
if certificate_rows:
    with (root/'aggregate/interval_certificate_table.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(certificate_rows[0]))
        writer.writeheader(); writer.writerows(certificate_rows)
    lines += ['## Inter-sample certificates and conformal buffers', '',
              'The per-interval correction is epsilon_inter,j = L_Psi,j (C_x + 1) dt. '
              'C_x = sqrt(max(|v_min|, |v_max|)^2 + beta_true^2). '
              'The local bounds enclose the position/heading reachable tube and are uniform over the control box; no rollout fit supplies these bounds.', '',
              '`aggregate/interval_certificate_table.csv` records every pretraining and final audit, all regional buffers, C_x, the L_Psi and epsilon ranges, active-node counts, and both feasibility margins. '
              'The full distributions and formula checks are in each run’s `certificate_acceptance.json`; interval/node details remain in the calibration and runtime artifacts.', '']
    lines += ['Nontrivial mismatch buffers are required in the Easy/Hard conditions. Identical true and modeled dynamics in Aligned produce zero model-mismatch scores, so a zero conformal mismatch buffer there is valid; the inter-sample correction remains separately enforced.', '']
    lines += ['Each primary CP run also includes `final_representative_interval_certificates.csv`, with every interval of its first held-out episode: state, time, Delta_j, C_x, L_Psi,j, epsilon_inter,j, buffer, and independently reconstructed feasibility margin. `aggregate/runtime_interval_audit.json` checks reconstruction against the recorded runtime margins for all 15 CP runs. These fixed representatives supplement the exhaustive and all-episode audits; they do not replace them.', '']
if not audit['passed']:
    lines += ['Audit failures:','']+[f'- {f}' for f in audit['failures']]+['']
lines += ['## Frozen simulation and training settings','',
          f"Controller interval {common['dt']} s; horizon {common['horizon']} steps ({common['dt']*common['horizon']:g} s); integration substeps {common['integration_substeps']}.",
          f"Goal center ({common.get('goal_x',2.)}, {common.get('goal_y',2.)}); success radius 0.5 m including the robot radius.",
          f"Shared reference waypoint ({common.get('reference_waypoint_x',-.65)}, {common.get('reference_waypoint_y',.65)}); switch to the goal when y reaches 0.45 m. The same reference and residual-action range apply to all methods.",
          f"Shared orbit-reference trigger radius {common.get('reference_orbit_trigger_radius',0.)} m (zero disables it); reference circle radius {common.get('reference_orbit_radius',1.)} m; radial gain {common.get('reference_orbit_gain',2.)}. Before the goal switch, inside the trigger radius the desired direction is (y,-x)/r + gain*(reference_radius-r)*(x,y)/r around the origin-centered obstacle.",
          f"Initial-state center ({common['initial_x']}, {common['initial_y']}, {common['initial_theta']}); position jitter ±{common['start_jitter_xy']}; heading jitter ±{common['start_jitter_theta']}.",
          f"Shared activation threshold {common['cp_activate_margin']}; release margin {common['shield_release_margin']}; gamma {common['gamma']}.",
          f"Policies learn a residual action bounded by ±{common['residual_reference_scale']} around the shared reference. Replay warmup lasts {common['start_training']} steps; reference-only proposals last {common['reference_proposal_steps']} steps.",
          f"Training steps per independently initialized policy: {common['max_steps']}; calibration episodes: {common['n_calib']}; final evaluation episodes: {common['eval_episodes']}.",'',
          'Condition dynamics:','', '```json',json.dumps(manifest['conditions'],indent=2),'```','',
          '## Artifacts','',
          '- `aggregate/main_results_table.csv` and `.tex`: publication table.',
          '- `aggregate/certification_feasibility_table.csv` and `.tex`: coverage and audit evidence.',
          '- `aggregate/interval_certificate_table.csv`: all phase-specific buffers and inter-sample constants.',
          '- `aggregate/figure1_learning_comparison.pdf` through Figure 5: figures with predeclared representative seeds.',
          '- `aggregate/acceptance_audit.json`: full acceptance evidence, including any failures.',
          '- `run_state.json`: commands, configuration hashes, and run outcomes.',
          '', 'Conformal interpretation follows [Angelopoulos and Bates](https://arxiv.org/abs/2107.07511); changes of test population do not automatically preserve ordinary split-conformal validity ([Tibshirani et al.](https://papers.neurips.cc/paper_files/paper/2019/hash/8fb21ee7a2207526da55a679f0332de2-Abstract.html)).','']
hard_metrics = {r['method']: r for r in metrics if r['condition'] == 'hard'}
if hard_metrics and float(hard_metrics['cbvf']['unsafe_rate_mean']) == float(hard_metrics['cp']['unsafe_rate_mean']) == 0:
    lines += ['## Interpretation of the baseline comparison', '',
              'Both uncalibrated CBVF and conformal-CBVF have zero observed Hard unsafe outcomes in this study. The empirical safety separation is against Nominal; these results do not demonstrate an incremental safety improvement of conformal calibration over uncalibrated CBVF. Returns and intervention costs must be assessed alongside safety and task success.', '']
artifacts = {
    'figures': sorted(str(path) for path in root.rglob('*') if path.suffix in ('.png','.pdf')),
    'tables_and_interval_series': sorted(str(path) for path in root.rglob('*') if path.suffix in ('.csv','.tex')),
    'manifest': str(root.parent/'manifest.json'), 'freeze': str(root.parent/'FREEZE.json'),
    'effective_parameters': str(root.parent/'effective_parameters.json'),
    'acceptance': str(root/'aggregate/acceptance_audit.json'),
    'runtime_interval_audit': str(root/'aggregate/runtime_interval_audit.json'),
}
(root/'ARTIFACT_INDEX.json').write_text(json.dumps(artifacts,indent=2)+'\n')
lines += ['## Exact file locations', '',
          f"Frozen configuration: [{root.parent/'manifest.json'}](<{root.parent/'manifest.json'}>).", '',
          f"Effective values of all CLI arguments, including defaults, for all 45 jobs: [{root.parent/'effective_parameters.json'}](<{root.parent/'effective_parameters.json'}>).", '',
          f"Complete figure/table/interval-series inventory: [{root/'ARTIFACT_INDEX.json'}](<{root/'ARTIFACT_INDEX.json'}>).", '',
          'Publication figures and tables:', '']
for path in sorted((root/'aggregate').glob('*')):
    if path.suffix in ('.png','.pdf','.csv','.tex'):
        lines.append(f'- [{path.name}](<{path}>).')
report='\n'.join(lines)+'\n'
(root/'PAPER_RESULTS.md').write_text(report)
project=Path(__file__).resolve().parents[1]
(project/'LATEST_CASE_STUDY_REPORT.md').write_text(report)
print(root/'PAPER_RESULTS.md')
