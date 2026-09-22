#!/usr/bin/env python3
"""Strict scientific acceptance audit for the definitive full suite."""
import json
import math
import argparse
import subprocess
from pathlib import Path
from audit_pilot_certificates import audit_certificates

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "results" / "run_state.json"
OUT_JSON = ROOT / "results" / "aggregate" / "acceptance_audit.json"
OUT_MD = ROOT / "results" / "aggregate" / "ACCEPTANCE_AUDIT.md"


def main() -> int:
    global STATE, OUT_JSON, OUT_MD
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default=str(ROOT / "results"))
    args = parser.parse_args()
    results_dir = Path(args.results_dir).resolve()
    STATE = results_dir / "run_state.json"
    OUT_JSON = results_dir / "aggregate" / "acceptance_audit.json"
    OUT_MD = results_dir / "aggregate" / "ACCEPTANCE_AUDIT.md"
    state = json.loads(STATE.read_text())
    failures = []
    evidence = []
    if len(state["runs"]) != 45:
        failures.append(f"expected 45 runs, found {len(state['runs'])}")
    incomplete = [k for k, r in state["runs"].items() if r["status"] != "completed"]
    if incomplete:
        failures.append(f"incomplete runs: {incomplete}")
    runtime_intervals_passed = False
    if len(state['runs']) == 45 and not incomplete:
        OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
        manifest=json.loads(Path(state['manifest_path']).read_text())
        command=[manifest['python'], str(ROOT/'experiments/audit_runtime_intervals.py'), '--results-dir', str(results_dir)]
        with (OUT_JSON.parent/'runtime_interval_execution.log').open('w') as log:
            result=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        runtime_intervals_passed=result.returncode==0
        if not runtime_intervals_passed:
            failures.append('runtime interval reconstruction failed; see runtime_interval_execution.log')

    hard_cbvf_unsafe = []
    hard_nominal_unsafe = []
    cp_coverages = []
    cp_goal_rates = []
    returns_by_condition_method = {}
    certificate_results = []
    for run_id, record in state["runs"].items():
        if record["status"] != "completed":
            continue
        result_path = Path(record["result_paths"]["result"])
        if not result_path.is_file():
            failures.append(f"{run_id}: missing result")
            continue
        result = json.loads(result_path.read_text())
        certificate = audit_certificates(result_path.parent)
        certificate_results.append(certificate)
        failures.extend(run_id + ': ' + failure for failure in certificate['failures'])
        method = record["configuration"]["training_shield_method"]
        final = result["final_evaluation"][method]
        audit = result["offline_cp_qp_feasibility_audit"]
        pretraining_path = result_path.parent / "pretraining_mondrian_calibration.json"
        if not pretraining_path.is_file():
            failures.append(f"{run_id}: missing pretraining audit")
        else:
            pretraining = json.loads(pretraining_path.read_text())["offline_cp_qp_feasibility_audit"]
            if (pretraining["n_infeasible_grid_nodes"] != 0 or
                    pretraining["n_uncalibrated_infeasible_grid_nodes"] != 0 or
                    pretraining["min_feasibility_margin"] <= 0 or
                    pretraining["min_uncalibrated_feasibility_margin"] <= 0):
                failures.append(f"{run_id}: pretraining audit lacks zero infeasibility and positive margins")
        training_diag = result.get("training_qp_feasibility_diagnostics", {})
        for key in ("infeasible_cp_steps", "rail_fallback_steps",
                    "evaluation_infeasible_cp_steps_across_recorded_evaluations",
                    "evaluation_rail_fallback_steps_across_recorded_evaluations"):
            if training_diag.get(key, 0) != 0:
                failures.append(f"{run_id}: nonzero {key}")
        row = {
            "run_id": run_id,
            "condition": record["condition"],
            "method": method,
            "seed": record["seed"],
            "goal_rate": final["goal_rate"],
            "unsafe_rate": final["unsafe_rate"],
            "coverage": (
                final["empirical_simultaneous_coverage"]
                if math.isfinite(final["empirical_simultaneous_coverage"])
                else None
            ),
            "infeasible_total": final["infeasible_total"],
            "fallback_total": final["rail_fallback_total"],
            "offline_min_margin": audit["min_feasibility_margin"],
            "offline_calibrated_infeasible": audit["n_infeasible_grid_nodes"],
            "offline_uncalibrated_infeasible": audit["n_uncalibrated_infeasible_grid_nodes"],
            "runtime_min_margin": final["min_active_feasibility_margin"],
        }
        evidence.append(row)
        returns_by_condition_method.setdefault((record["condition"], method), []).append(final["return"])
        if (result.get("status") != "completed" or
                result.get("result_validity") != "no_cp_infeasibility_observed_on_executed_training_states"):
            failures.append(f"{run_id}: result is not marked completed and valid")
        workflow = result.get("workflow_invariants", {})
        if not workflow.get("post_training_recalibration"):
            failures.append(f"{run_id}: missing post-training frozen-policy recalibration")
        if workflow.get("final_evaluation_coverage_claim") != "split_conformal_for_frozen_policy_with_disjoint_evaluation_seeds":
            failures.append(f"{run_id}: invalid final calibration/evaluation separation")
        if not result.get("cbvf", {}).get("recomputed"):
            failures.append(f"{run_id}: CBVF was not freshly recomputed")
        if Path(result.get("calibration_manifest", "")).name != "final_frozen_policy_mondrian_calibration.json":
            failures.append(f"{run_id}: final calibration manifest is not the frozen-policy manifest")
        if audit["n_infeasible_grid_nodes"] != 0 or audit["n_uncalibrated_infeasible_grid_nodes"] != 0:
            failures.append(f"{run_id}: offline QP infeasibility")
        if audit["min_feasibility_margin"] <= 0:
            failures.append(f"{run_id}: nonpositive offline feasibility margin")
        if audit["min_uncalibrated_feasibility_margin"] <= 0:
            failures.append(f"{run_id}: nonpositive uncalibrated offline feasibility margin")
        if method == "cp":
            if not math.isfinite(final['min_active_feasibility_margin']) or final['min_active_feasibility_margin'] <= 0:
                failures.append(f"{run_id}: nonpositive or missing CP runtime margin")
            cp_goal_rates.append(final["goal_rate"])
            coverage_path = result_path.parent / "heldout_reference_coverage_fresh.json"
            if not coverage_path.is_file():
                failures.append(f"{run_id}: missing fresh held-out conformal coverage result")
                heldout_coverage = float("nan")
            else:
                heldout = json.loads(coverage_path.read_text())
                heldout_coverage = float(heldout["empirical_simultaneous_coverage"])
                if heldout.get("episodes", 0) < 100:
                    failures.append(f"{run_id}: held-out conformal coverage uses fewer than 100 episodes")
                if not heldout.get("fresh_calibration", False):
                    failures.append(f"{run_id}: held-out conformal coverage did not use fresh calibration")
                if not heldout.get("score_population_matches_calibration", False):
                    failures.append(f"{run_id}: held-out score population not verified")
                if heldout.get("disjointness", {}).get("overlap_pairs") != 0:
                    failures.append(f"{run_id}: held-out calibration/evaluation starts overlap")
            row["deployment_diagnostic_coverage"] = row["coverage"]
            row["coverage"] = heldout_coverage if math.isfinite(heldout_coverage) else None
            # Missing coverage is already a hard failure above.  Keep NaN out
            # of the aggregate so the audit can still serialize a complete
            # failure report instead of crashing before writing evidence.
            if math.isfinite(heldout_coverage):
                cp_coverages.append(heldout_coverage)
            if final["unsafe_rate"] != 0 or final["total_safety_violations"] != 0:
                failures.append(f"{run_id}: calibrated safety violation")
            if final.get("total_intersample_safety_violations", 0) != 0:
                failures.append(f"{run_id}: calibrated inter-sample safety violation")
            if final["infeasible_total"] != 0 or final["rail_fallback_total"] != 0:
                failures.append(f"{run_id}: calibrated infeasibility/fallback")
            if final["minimal_projection_failure_total"] != 0:
                failures.append(f"{run_id}: calibrated projection minimality failure")
            if final["goal_rate"] < 0.90:
                failures.append(f"{run_id}: calibrated goal rate below 0.90")
            if not math.isfinite(heldout_coverage):
                failures.append(f"{run_id}: missing held-out empirical coverage")
            elif heldout_coverage < 0.95:
                failures.append(f"{run_id}: held-out empirical coverage below 0.95")
        if record["condition"] == "hard" and method == "cbvf":
            hard_cbvf_unsafe.append(final["unsafe_rate"])
        if record["condition"] == "hard" and method == "nominal":
            hard_nominal_unsafe.append(final["unsafe_rate"])

    if cp_coverages and sum(cp_coverages) / len(cp_coverages) < 0.95:
        failures.append("mean CP empirical coverage below the 0.95 target")
    if hard_cbvf_unsafe and hard_nominal_unsafe and max(
            sum(hard_cbvf_unsafe) / len(hard_cbvf_unsafe),
            sum(hard_nominal_unsafe) / len(hard_nominal_unsafe)) < 0.10:
        failures.append("neither hard-mismatch baseline reaches the 10% separation floor")
    return_comparisons = {}
    for condition in ("aligned", "easy", "hard"):
        means = {}
        for method in ("nominal", "cbvf", "cp"):
            values = returns_by_condition_method.get((condition, method), [])
            if values:
                means[method] = sum(values) / len(values)
        if len(means) == 3:
            best_baseline = max(means["nominal"], means["cbvf"])
            tolerance = max(100.0, 0.25 * abs(best_baseline))
            return_comparisons[condition] = {"means": means, "allowed_gap": tolerance}
            # Return is reported, but no post-hoc absolute-gap threshold is a
            # certification condition.  Useful task performance is enforced
            # above by the predeclared per-seed CP goal-rate floor of 0.90.

    complete = len(evidence) == 45 and not incomplete
    phases = [phase for cert in certificate_results for phase in cert['phases']]
    cp_rows = [row for row in evidence if row['method'] == 'cp']
    criteria = {
        'complete_45_run_suite': complete,
        'zero_calibrated_and_uncalibrated_audit_infeasibility': complete and len(phases) == 90 and all(
            row['offline_calibrated_infeasible'] == row['offline_uncalibrated_infeasible'] == 0 for row in evidence) and not any('infeasible_grid_nodes' in failure for cert in certificate_results for failure in cert['failures']),
        'strictly_positive_feasibility_margins': complete and len(phases) == 90 and all(
            phase['min_calibrated_margin'] > 0 and phase['min_uncalibrated_margin'] > 0 for phase in phases) and all(row['runtime_min_margin'] > 0 for row in cp_rows),
        'hard_safety_separation': complete and len(hard_nominal_unsafe) == len(hard_cbvf_unsafe) == 5 and max(
            sum(hard_nominal_unsafe) / 5, sum(hard_cbvf_unsafe) / 5) >= .10 and all(row['unsafe_rate'] == 0 for row in cp_rows),
        'acceptable_goal_success': complete and len(cp_goal_rates) == 15 and min(cp_goal_rates) >= .90,
        'nontrivial_conformal_buffers_in_mismatch_conditions': complete and len(phases) == 90 and all(
            max(phase['buffers']) > 1e-8 for phase in phases if phase['nontrivial_buffer_required']),
        'numerical_intersample_lipschitz_certificates': complete and len(certificate_results) == 45 and all(cert['passed'] for cert in certificate_results),
        'per_interval_runtime_reconstruction': complete and runtime_intervals_passed,
        'heldout_100_rollout_reference_coverage': complete and len(cp_coverages) == 15 and min(cp_coverages) >= .95 and not any('held-out' in failure for failure in failures),
        'zero_cp_runtime_infeasibility_and_unsafe_outcomes': complete and len(cp_rows) == 15 and all(
            row['unsafe_rate'] == row['infeasible_total'] == row['fallback_total'] == 0 for row in cp_rows) and not any('infeasible_cp_steps' in failure or 'safety violation' in failure for failure in failures),
    }
    failures.extend('criterion failed: ' + key for key, passed in criteria.items() if not passed)
    payload = {
        "passed": not failures,
        "failures": failures,
        "n_runs": len(evidence),
        "cp_mean_goal_rate": sum(cp_goal_rates) / len(cp_goal_rates) if cp_goal_rates else None,
        "cp_mean_coverage": sum(cp_coverages) / len(cp_coverages) if cp_coverages else None,
        "hard_cbvf_total_unsafe_rate": sum(hard_cbvf_unsafe),
        "hard_nominal_total_unsafe_rate": sum(hard_nominal_unsafe),
        "return_comparisons": return_comparisons,
        "acceptance_criteria": {key: 'PASS' if passed else 'FAIL' for key, passed in criteria.items()},
        "runs": evidence,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    lines = ["# Full-suite acceptance audit", "", f"Status: **{'PASS' if not failures else 'FAIL'}**", "",
             "Coverage is evaluated from each CP run's fresh 100-episode, disjoint, guarantee-matched baseline-CBVF population; conformal-feedback trajectory coverage is retained only as a deployment diagnostic.", ""]
    lines += [f"- {x}" for x in failures] if failures else ["All strict acceptance checks passed."]
    OUT_MD.write_text("\n".join(lines) + "\n")
    print(json.dumps({k: v for k, v in payload.items() if k != "runs"}, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
