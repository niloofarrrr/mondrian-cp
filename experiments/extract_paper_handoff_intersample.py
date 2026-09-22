#!/usr/bin/env python3
"""Memory-bounded extraction of paper metrics from the completed final suite.

This script does not simulate, train, calibrate, or alter run artifacts.  It
loads one completed result at a time and writes provenance-preserving tables.
"""
import csv
import json
import math
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUITE = ROOT / "results_intersample" / "final_suite"
OUT = SUITE / "aggregate"
T95_4 = 2.7764451051977987


def stats(values):
    values = [float(x) for x in values if x is not None and math.isfinite(float(x))]
    if not values:
        return {"n": 0, "mean": None, "sd": None, "ci95_halfwidth": None}
    mean = statistics.mean(values)
    sd = statistics.stdev(values) if len(values) > 1 else 0.0
    return {
        "n": len(values), "mean": mean, "sd": sd,
        "ci95_halfwidth": T95_4 * sd / math.sqrt(len(values)) if len(values) == 5 else None,
    }


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    state = json.loads((SUITE / "run_state.json").read_text())
    if len(state["runs"]) != 45 or any(r["status"] != "completed" for r in state["runs"].values()):
        raise SystemExit("Refusing to aggregate anything except the complete 45-run suite")
    groups = {}
    audits = {}
    calibrations = {}
    run_evidence = []
    metric_names = None
    for run_id in state["order"]:
        record = state["runs"][run_id]
        result_path = Path(record["result_paths"]["result"])
        result = json.loads(result_path.read_text())
        condition = record["condition"]
        method = record["configuration"]["training_shield_method"]
        final = result["final_evaluation"][method]
        groups.setdefault((condition, method), []).append(final)
        audits.setdefault(condition, []).append(result["offline_cp_qp_feasibility_audit"])
        if method == "cp":
            calibrations.setdefault(condition, []).append(result["calibration_stats"])
        if metric_names is None:
            metric_names = sorted(k for k, v in final.items() if isinstance(v, (int, float)) and not isinstance(v, bool))
        run_evidence.append({
            "run_id": run_id, "condition": condition, "method": method,
            "seed": record["seed"], "result": str(result_path.relative_to(ROOT)),
            "status": result.get("status"), "result_validity": result.get("result_validity"),
        })

    metric_rows = []
    conditions = ("aligned", "easy", "hard")
    methods = ("nominal", "cbvf", "cp")
    for condition in conditions:
        for method in methods:
            finals = groups[(condition, method)]
            row = {"condition": condition, "method": method, "seeds": "7;42;123;0;99", "evaluation_episodes": 100}
            for name in metric_names:
                s = stats([x.get(name) for x in finals])
                for suffix, value in s.items():
                    row[f"{name}_{suffix}"] = value
            row["pooled_unsafe_episodes"] = sum(round(float(x["unsafe_rate"]) * 20) for x in finals)
            row["pooled_goal_episodes"] = sum(round(float(x["goal_rate"]) * 20) for x in finals)
            row["pooled_safety_violations"] = sum(float(x["total_safety_violations"]) for x in finals)
            row["pooled_intersample_violations"] = sum(float(x.get("total_intersample_safety_violations", 0)) for x in finals)
            metric_rows.append(row)

    audit_rows = []
    for condition in conditions:
        values = audits[condition]
        node_counts = sorted(set(int(x["n_active_certified_grid_nodes"]) for x in values))
        audit_rows.append({
            "condition": condition,
            "audits_embedded_in_runs": len(values),
            "unique_active_certified_node_counts": ";".join(map(str, node_counts)),
            "sum_nodes_across_15_run_embedded_audits": sum(int(x["n_active_certified_grid_nodes"]) for x in values),
            "calibrated_infeasible_nodes_sum": sum(int(x["n_infeasible_grid_nodes"]) for x in values),
            "uncalibrated_infeasible_nodes_sum": sum(int(x["n_uncalibrated_infeasible_grid_nodes"]) for x in values),
            "infeasible_rate_max": max(float(x["infeasible_grid_node_rate"]) for x in values),
            "minimum_feasibility_margin_across_runs": min(float(x["min_feasibility_margin"]) for x in values),
            "minimum_uncalibrated_margin_across_runs": min(float(x["min_uncalibrated_feasibility_margin"]) for x in values),
            "C_x": values[0]["C_x"], "L_Psi_min": min(x["L_Psi_min"] for x in values),
            "L_Psi_max": max(x["L_Psi_max"] for x in values),
            "epsilon_inter_min": min(x["epsilon_inter_min"] for x in values),
            "epsilon_inter_max": max(x["epsilon_inter_max"] for x in values),
            "decision_intervals": values[0]["runtime_decision_intervals_audited"],
            "grid_shape": "x".join(map(str, values[0]["grid_shape"])),
            "activation_upper": values[0]["activation_upper"],
            "certified_initial_B_floor": values[0]["certified_initial_B_floor"],
        })

    calibration_rows = []
    for condition in conditions:
        values = calibrations[condition]
        for region in (0, 1):
            buffers = [x["regions"][region]["xi_hat_off"] for x in values]
            s = stats(buffers)
            row = {
                "condition": condition, "effective_region": region,
                "region_label": values[0]["regions"][region]["base_region_labels"][0],
                "delta_m": values[0]["regions"][region]["delta_m"],
                "n_scores_per_seed": values[0]["regions"][region]["n_scores"],
                "buffers_by_seed_7_42_123_0_99": ";".join(format(float(x), ".17g") for x in buffers),
            }
            row.update({f"xi_hat_{k}": v for k, v in s.items()})
            calibration_rows.append(row)
        for key in ("score_min", "score_mean", "score_p50", "score_p90", "score_max", "nonzero_frac"):
            s = stats([x[key] for x in values])
            calibration_rows[-2][f"all_scores_{key}_cross_seed_mean"] = s["mean"]
            calibration_rows[-2][f"all_scores_{key}_cross_seed_sd"] = s["sd"]

    OUT.mkdir(parents=True, exist_ok=True)
    write_csv(OUT / "paper_metrics_intersample.csv", metric_rows)
    write_csv(OUT / "paper_feasibility_audits_intersample.csv", audit_rows)
    write_csv(OUT / "paper_calibration_intersample.csv", calibration_rows)
    figure_rows = []
    for path in sorted(SUITE.glob("runs/*/plots/*")):
        if path.suffix.lower() in (".png", ".pdf"):
            run_id = path.parents[1].name
            condition, method, _, seed = run_id.rsplit("_", 3)
            figure_rows.append({
                "relative_path": str(path.relative_to(ROOT)), "run_id": run_id,
                "condition": condition, "method": method, "seed": seed,
                "figure_kind": path.stem, "source_result": str((path.parents[1] / "training_and_evaluation_results.json").relative_to(ROOT)),
                "source_script": "train/train_sac_lag.py:_plot_training_outputs",
                "provenance": "final_suite",
            })
    for path in sorted(OUT.glob("figure*")):
        figure_rows.append({
            "relative_path": str(path.relative_to(ROOT)), "run_id": "aggregate",
            "condition": "aligned;easy;hard", "method": "nominal;cbvf;cp",
            "seed": "7;42;123;0;99", "figure_kind": path.stem,
            "source_result": "results_intersample/final_suite/runs/*/training_and_evaluation_results.json",
            "source_script": "experiments/aggregate_full_suite_intersample.py",
            "provenance": "final_suite_postprocessing",
        })
    write_csv(OUT / "paper_figure_manifest_intersample.csv", figure_rows)
    payload = {
        "source_state": str((SUITE / "run_state.json").relative_to(ROOT)),
        "suite_status": state["status"], "n_runs": 45,
        "heldout_reference_coverage_files_found": len(list((SUITE / "runs").glob("*cp*/heldout_reference_coverage_fresh.json"))),
        "metrics": metric_rows, "feasibility_audits": audit_rows,
        "calibration": calibration_rows, "run_evidence": run_evidence,
        "figures": figure_rows,
    }
    (OUT / "paper_handoff_data_intersample.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(OUT)


if __name__ == "__main__":
    main()
