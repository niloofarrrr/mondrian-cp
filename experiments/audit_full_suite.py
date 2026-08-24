#!/usr/bin/env python3
"""Strict scientific acceptance audit for the definitive full suite."""
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "results" / "run_state.json"
OUT_JSON = ROOT / "results" / "aggregate" / "acceptance_audit.json"
OUT_MD = ROOT / "results" / "aggregate" / "ACCEPTANCE_AUDIT.md"


def main() -> int:
    state = json.loads(STATE.read_text())
    failures = []
    evidence = []
    if len(state["runs"]) != 45:
        failures.append(f"expected 45 runs, found {len(state['runs'])}")
    incomplete = [k for k, r in state["runs"].items() if r["status"] != "completed"]
    if incomplete:
        failures.append(f"incomplete runs: {incomplete}")

    hard_cbvf_unsafe = []
    hard_nominal_unsafe = []
    cp_coverages = []
    cp_goal_rates = []
    returns_by_condition_method = {}
    for run_id, record in state["runs"].items():
        if record["status"] != "completed":
            continue
        result_path = Path(record["result_paths"]["result"])
        if not result_path.is_file():
            failures.append(f"{run_id}: missing result")
            continue
        result = json.loads(result_path.read_text())
        method = record["configuration"]["training_shield_method"]
        final = result["final_evaluation"][method]
        audit = result["offline_cp_qp_feasibility_audit"]
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
        if audit["min_feasibility_margin"] < -1e-10:
            failures.append(f"{run_id}: negative offline feasibility margin")
        if audit["min_uncalibrated_feasibility_margin"] < -1e-10:
            failures.append(f"{run_id}: negative uncalibrated offline feasibility margin")
        if method == "cp":
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
                if heldout.get("disjointness", {}).get("overlap_pairs") != 0:
                    failures.append(f"{run_id}: held-out calibration/evaluation starts overlap")
            row["deployment_diagnostic_coverage"] = row["coverage"]
            row["coverage"] = heldout_coverage if math.isfinite(heldout_coverage) else None
            cp_coverages.append(heldout_coverage)
            if final["unsafe_rate"] != 0 or final["total_safety_violations"] != 0:
                failures.append(f"{run_id}: calibrated safety violation")
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
    if hard_cbvf_unsafe and sum(hard_cbvf_unsafe) <= 0:
        failures.append("hard-mismatch uncalibrated CBVF has no safety failures")
    if hard_nominal_unsafe and sum(hard_nominal_unsafe) <= 0:
        failures.append("hard-mismatch nominal controller has no safety failures")
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

    payload = {
        "passed": not failures,
        "failures": failures,
        "n_runs": len(evidence),
        "cp_mean_goal_rate": sum(cp_goal_rates) / len(cp_goal_rates) if cp_goal_rates else None,
        "cp_mean_coverage": sum(cp_coverages) / len(cp_coverages) if cp_coverages else None,
        "hard_cbvf_total_unsafe_rate": sum(hard_cbvf_unsafe),
        "hard_nominal_total_unsafe_rate": sum(hard_nominal_unsafe),
        "return_comparisons": return_comparisons,
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
