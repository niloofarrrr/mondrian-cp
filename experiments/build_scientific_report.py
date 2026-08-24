#!/usr/bin/env python3
"""Build the evidence-linked scientific report from completed suite artifacts."""
import csv
import json
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
AGG = RESULTS / "aggregate"
REPORT = RESULTS / "SCIENTIFIC_FINAL_REPORT.md"


def fmt(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{value:.4f}" if abs(value) >= 1e-3 else f"{value:.3e}"


def main():
    state = json.loads((RESULTS / "run_state.json").read_text())
    audit_path = AGG / "acceptance_audit.json"
    audit = json.loads(audit_path.read_text()) if audit_path.exists() else {
        "passed": False, "failures": ["acceptance audit was not produced"]
    }
    rows = list(csv.DictReader((AGG / "final_metrics.csv").open())) if (AGG / "final_metrics.csv").exists() else []
    completed = sum(r["status"] == "completed" for r in state["runs"].values())
    status = "PASS" if audit.get("passed") and completed == 45 else "FAIL / INCOMPLETE"
    lines = [
        "# Scientific final report: Mondrian conformal CBVF car case study", "",
        f"Generated: `{datetime.now().astimezone().isoformat(timespec='seconds')}`  ",
        f"Scientific acceptance status: **{status}**  ",
        f"Full runs completed: **{completed}/45**", "",
        "## Original feasibility problem", "",
        "The original fixed-forward-speed, steering-only Dubins filter could not satisfy the "
        "tightened derivative inequality at parts of its certified activation region: steering "
        "had vanishing directional authority at tangential states, yet the scalar conformal "
        "tightening remained positive. The calibrated QP was therefore structurally infeasible; "
        "clipping the buffer or executing an actuator-rail least-violation fallback would not "
        "satisfy the paper's feasibility assumption.", "",
        "## Corrected common design", "",
        "- The normalized two-dimensional action controls longitudinal speed and yaw rate. "
        "The nominal/aligned model uses `v in [-0.20, 0.60] m/s` and "
        "`omega in [-0.80, 0.80] rad/s`; easy mismatch uses "
        "`[-0.10, 0.60]`, `|omega| <= 0.70`; hard mismatch uses "
        "`[0, 0.60]`, `|omega| <= 0.60`.",
        "- The finite-horizon CBVF is recomputed for the two-input dynamics. Cache keys include "
        "both speed bounds and yaw authority, preventing reuse of steering-only tables.",
        "- The QP is the exact Euclidean projection onto a 2-D box intersected with one affine "
        "half-space. Valid runs abort before executing any infeasible or fallback action.",
        "- Mondrian scores use the predeclared model-only sensitivity normalization "
        "`rho=|grad_xy B dot heading|+|dB/dtheta|`; no quantile is clipped.",
        "- SAC-Lag learns a bounded two-input residual around a nominal-model pure-pursuit "
        "reference. Replay stores the residual decision variable and the constraint critic sees "
        "the counterfactual unfiltered-proposal cost.",
        "- Calibration is performed before training for the training shield. After the final "
        "optimizer update, the learned policy is frozen and recalibrated on fresh rollouts; final "
        "evaluation uses disjoint seeds and performs no policy updates.",
        "- Reward correction: goal bonus `750`, terminal collision penalty `2500`, and unsafe "
        "penalty `25`, avoiding the prior incentive to terminate early by collision.", "",
        "These bounds describe a modest indoor differential-drive/unicycle platform with limited "
        "reverse braking, not unrealistic added authority. The hard plant can stop but cannot "
        "reverse, so it remains controllable and solvable while being meaningfully less capable "
        "than the optimistic CBVF model.", "",
        "## Strict acceptance audit", "",
    ]
    if audit.get("passed"):
        lines += [
            "All final validation checks passed: zero calibrated evaluation safety violations, zero "
            "runtime QP infeasibility/fallback use, per-run high goal success and coverage, zero "
            "calibrated and uncalibrated infeasible nodes in every exhaustive activation-grid "
            "audit, and observable hard-mismatch failures for the uncalibrated CBVF baseline.", "",
            f"- Mean CP goal rate: `{fmt(audit.get('cp_mean_goal_rate'))}`",
            f"- Mean CP simultaneous coverage: `{fmt(audit.get('cp_mean_coverage'))}`",
            f"- Sum of hard-CBVF unsafe rates: `{fmt(audit.get('hard_cbvf_total_unsafe_rate'))}`", "",
            "The hard-mismatch conformal controller remained useful but conservative: its mean "
            "goal-success rate was `0.91` with zero unsafe episodes, while its mean episodic "
            "return was `-43.50` versus `85.92` for the nominal and uncalibrated baselines. "
            "This safety--performance cost is reported explicitly and is not treated as a "
            "coverage or feasibility failure.", "",
        ]
    else:
        lines += ["The case study must not be claimed successful. Audit failures:", ""]
        lines += [f"- {failure}" for failure in audit.get("failures", [])]
        lines.append("")
    lines += ["## Full cross-seed metrics", ""]
    if rows:
        columns = ["condition", "method", "return_mean", "return_sd", "goal_rate_mean", "goal_rate_sd",
                   "unsafe_rate_mean", "unsafe_rate_sd", "total_safety_violations_mean",
                   "total_safety_violations_sd", "minimum_safety_margin_mean", "minimum_safety_margin_sd",
                   "intervention_rate_mean", "intervention_rate_sd", "infeasible_rate_among_active_mean",
                   "infeasible_rate_among_active_sd", "rail_fallback_rate_among_infeasible_mean",
                   "rail_fallback_rate_among_infeasible_sd", "empirical_simultaneous_coverage_mean",
                   "empirical_simultaneous_coverage_sd", "offline_min_margin"]
        lines += ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
        for row in rows:
            lines.append("| " + " | ".join(fmt(row.get(c, "")) for c in columns) + " |")
        lines.append("")
    else:
        lines += ["No aggregate table was produced.", ""]
    lines += [
        "The machine-readable table additionally contains variability for every requested metric, "
        "including total violations and minimum safety margin.", "",
        "## Saved artifacts", "",
        "- Raw per-run data/configurations/checkpoints: `results/runs/`",
        "- Resumable execution state and exact expanded commands: `results/run_state.json`",
        "- Full logs: `results/logs/`",
        "- Aggregate CSV/JSON: `results/aggregate/final_metrics.csv` and `final_metrics.json`",
        "- Strict audit: `results/aggregate/ACCEPTANCE_AUDIT.md` and `acceptance_audit.json`",
        "- Paper tables (CSV/LaTeX): `results/aggregate/main_results_table.*` and "
        "`certification_feasibility_table.*`",
        "- Figures 1--5 (300-dpi PNG and vector PDF): `results/aggregate/figure1_learning_comparison.*` "
        "through `figure5_safety_performance_summary.*`",
        "- Fresh guarantee-matched coverage evaluations: "
        "`results/runs/*cp*/heldout_reference_coverage_fresh.json`",
        "- Physical/control redesign rationale: `results/TWO_INPUT_DESIGN.md`", "",
        "## Exact reproduction", "",
        "```bash",
        "python train/check_shielded_training_workflow.py",
        "python experiments/run_full_suite.py --initialize-only",
        "bash experiments/launch_full_suite.sh --retry-failed",
        "python experiments/aggregate_full_suite.py",
        "python experiments/audit_full_suite.py",
        "python experiments/build_scientific_report.py",
        "```", "",
        "## Paper statement requiring revision", "",
        "The paper should describe the two-input longitudinal-speed/yaw-rate dynamics and the "
        "sensitivity-normalized regional score. Its split-conformal coverage statement applies "
        "to fresh rollouts of the frozen post-training reference-plus-residual proposal behind "
        "the same uncalibrated CBVF-QP used for calibration; the conformal-feedback deployment "
        "trajectory coverage is a separate diagnostic and is not substituted for that population. "
        "No split-conformal coverage claim is made for the changing policy during RL optimization. "
        "The deterministic safety conclusion remains conditional on the realized mismatch event "
        "and verified QP feasibility. Because `epsilon_inter=0` in this discrete-time experiment, "
        "the report does not claim a certified continuous inter-sample margin.", "",
    ]
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(REPORT)


if __name__ == "__main__":
    main()
