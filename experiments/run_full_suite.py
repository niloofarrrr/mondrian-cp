#!/usr/bin/env python3
"""Resumable, run-granularity batch driver for the Mondrian CBVF suite."""

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import traceback


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
STATE_PATH = RESULTS / "run_state.json"
STATUS_PATH = RESULTS / "RUN_STATUS.md"
FINAL_PATH = RESULTS / "FINAL_REPORT.md"
LOG_DIR = RESULTS / "logs"


def now():
    return dt.datetime.now(dt.timezone.utc).astimezone().isoformat(timespec="seconds")


def atomic_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(str(tmp), str(path))


def config_hash(payload):
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def cli_name(key):
    return "--" + key.replace("_", "-")


def expanded_runs(manifest):
    common = manifest["common"]
    py = manifest["python"]
    for condition, dynamics in manifest["conditions"].items():
        for method in manifest["methods"]:
            for seed in manifest["fixed_seeds"]:
                run_id = "%s_%s_seed_%s" % (condition, method, seed)
                out = RESULTS / "runs" / run_id
                cfg = dict(common)
                cfg.update(dynamics)
                cfg["training_shield_method"] = method
                cfg["seed"] = int(seed)
                cfg["checkpoint_dir"] = str(out)
                command = [py, "train/train_sac_lag.py"]
                flags = {"dense_safety_cost", "terminate_on_collision", "recompute_cbvf", "no_tqdm"}
                for key, value in cfg.items():
                    command.extend([cli_name(key), str(value)])
                for key in sorted(flags):
                    command.append(cli_name(key))
                command.extend([
                    "--odp-root", str(ROOT),
                    "--solver-file", str(ROOT / "solver_cbvf.py"),
                    "--results-json", str(out / "training_and_evaluation_results.json"),
                    "--plot-dir", str(out / "plots"),
                ])
                source_hashes = {
                    str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in (
                        ROOT / "conformal_shield.py",
                        ROOT / "solver_cbvf.py",
                        ROOT / "train" / "train_sac_lag.py",
                        ROOT / "redexp" / "envs" / "conformal_dubins_3d_env.py",
                    )
                }
                payload = {"suite": manifest["suite_name"], "run_id": run_id, "configuration": cfg, "command": command, "source_hashes": source_hashes}
                yield run_id, condition, int(seed), out, command, config_hash(payload), payload


def result_paths(out):
    return {
        "result": str(out / "training_and_evaluation_results.json"),
        "policy_config": str(out / "policy_config.json"),
        "calibration": str(out / "final_frozen_policy_mondrian_calibration.json"),
        "evaluations": str(out / "training_eval_metrics.jsonl"),
        "actions": str(out / "training_actions.jsonl"),
        "qp_diagnostics": str(out / "qp_feasibility_diagnostics.jsonl"),
        "plots": str(out / "plots"),
    }


def render_status(state):
    rows = []
    for run_id in state["order"]:
        r = state["runs"][run_id]
        paths = r.get("result_paths", {})
        result = paths.get("result", "")
        result_cell = "[%s](%s)" % (Path(result).name, Path(result).as_posix()) if result else "—"
        rows.append("| %s | %s | %s | %s | %s | %s |" % (
            run_id, r["condition"], r["seed"], r["status"],
            r.get("completion_time", "—"), result_cell,
        ))
    counts = {}
    for r in state["runs"].values():
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    active = state.get("active_run") or "none"
    text = """# Mondrian CBVF full-suite run status

Last updated: `{updated}`  
Suite state: **{suite_status}**  
Active run: **{active}**  
tmux session: `{tmux}`  
Configuration manifest: [`experiments/full_suite.json`](../experiments/full_suite.json)  
Machine-readable state: [`results/run_state.json`](run_state.json)

Counts: `{counts}`

| Run | Condition | Seed | Status | Completion time | Primary result |
|---|---:|---:|---|---|---|
{rows}

Each run's exact command, configuration hash, log, and all expected result paths are recorded in `run_state.json`. A failed run remains failed until explicitly retried; completed runs are reused only when their configuration hash and result file match.
""".format(updated=state["updated"], suite_status=state["status"], active=active,
           tmux=state.get("tmux_session", "mondrian-full-suite"), counts=json.dumps(counts, sort_keys=True),
           rows="\n".join(rows))
    tmp = STATUS_PATH.with_suffix(".md.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(str(tmp), str(STATUS_PATH))


def save_state(state):
    state["updated"] = now()
    atomic_json(STATE_PATH, state)
    render_status(state)


def initialize(manifest, retry_failed=False):
    prior = json.loads(STATE_PATH.read_text(encoding="utf-8")) if STATE_PATH.exists() else {"runs": {}}
    state = {
        "suite_name": manifest["suite_name"], "status": "queued", "active_run": None,
        "created": prior.get("created", now()), "updated": now(),
        "tmux_session": os.environ.get("MONDRIAN_TMUX_SESSION", "mondrian-full-suite"),
        "manifest_sha256": config_hash(manifest), "order": [], "runs": {},
    }
    for run_id, condition, seed, out, command, digest, payload in expanded_runs(manifest):
        old = prior.get("runs", {}).get(run_id, {})
        valid_complete = (old.get("status") == "completed" and old.get("config_sha256") == digest
                          and Path(result_paths(out)["result"]).is_file())
        status = "completed" if valid_complete else "pending"
        if old.get("status") == "failed" and old.get("config_sha256") == digest and not retry_failed:
            status = "failed"
        record = {
            "run_id": run_id, "condition": condition, "seed": seed, "status": status,
            "config_sha256": digest, "configuration": payload["configuration"],
            "command": command, "command_shell": shlex.join(command),
            "log": str(LOG_DIR / (run_id + ".log")), "result_paths": result_paths(out),
            "start_time": old.get("start_time") if valid_complete else None,
            "completion_time": old.get("completion_time") if valid_complete else None,
            "exit_code": old.get("exit_code") if valid_complete else None,
            "error": old.get("error") if valid_complete else None,
        }
        state["order"].append(run_id)
        state["runs"][run_id] = record
    return state


def write_final_report(state):
    completed = [r for r in state["runs"].values() if r["status"] == "completed"]
    failed = [r for r in state["runs"].values() if r["status"] == "failed"]
    lines = ["# Mondrian CBVF full-suite final report", "", "Generated: `%s`" % now(), "",
             "Suite status: **%s**" % state["status"], "",
             "Completed runs: **%d/%d**; failed runs: **%d**." % (len(completed), len(state["runs"]), len(failed)), ""]
    if failed:
        lines.extend(["## Failures", ""])
        for r in failed:
            lines.append("- `%s`: exit %s; log `%s`" % (r["run_id"], r.get("exit_code"), r["log"]))
        lines.append("")
    lines.extend(["## Run artifacts", ""])
    for r in completed:
        lines.append("- `%s`: `%s`" % (r["run_id"], r["result_paths"]["result"]))
    lines.extend(["", "## Reproduction", "", "```bash",
                  "tmux new-session -d -s mondrian-full-suite 'cd %s && bash experiments/launch_full_suite.sh --retry-failed'" % ROOT,
                  "```", "",
                  "This automatically generated report is an execution ledger. Scientific success criteria must be checked from the aggregated raw metrics before any theorem-level success claim is made.", ""])
    FINAL_PATH.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=str(ROOT / "experiments/full_suite.json"))
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--initialize-only", action="store_true")
    args = parser.parse_args()
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    RESULTS.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    state = initialize(manifest, retry_failed=args.retry_failed)
    if args.initialize_only:
        state["status"] = "not started"
        save_state(state)
        return 0
    state["status"] = "running"
    save_state(state)

    interrupted = {"value": False}
    def stop(signum, frame):
        interrupted["value"] = True
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)

    try:
        for run_id in state["order"]:
            r = state["runs"][run_id]
            if r["status"] in ("completed", "failed"):
                continue
            if interrupted["value"]:
                break
            r.update(status="running", start_time=now(), completion_time=None, exit_code=None, error=None)
            state["active_run"] = run_id
            save_state(state)
            out = Path(r["result_paths"]["result"]).parent
            out.mkdir(parents=True, exist_ok=True)
            (out / "batch_configuration.json").write_text(json.dumps({
                "config_sha256": r["config_sha256"], "configuration": r["configuration"],
                "command": r["command"], "start_time": r["start_time"]
            }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            try:
                with open(r["log"], "a", encoding="utf-8") as log:
                    log.write("\n=== batch start %s config=%s ===\n" % (r["start_time"], r["config_sha256"]))
                    log.flush()
                    proc = subprocess.run(r["command"], cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT)
                r["exit_code"] = proc.returncode
                result_ok = Path(r["result_paths"]["result"]).is_file()
                r["status"] = "completed" if proc.returncode == 0 and result_ok else "failed"
                if proc.returncode == 0 and not result_ok:
                    r["error"] = "process exited 0 but primary result file is missing"
                elif proc.returncode != 0:
                    r["error"] = "process exited with code %d; see log" % proc.returncode
            except Exception:
                r["status"] = "failed"
                r["exit_code"] = -1
                r["error"] = traceback.format_exc()
            r["completion_time"] = now()
            state["active_run"] = None
            save_state(state)
            if r["status"] == "failed":
                state["status"] = "failed"
                save_state(state)
                write_final_report(state)
                return 1
        statuses = [r["status"] for r in state["runs"].values()]
        state["status"] = "completed" if all(s == "completed" for s in statuses) else "interrupted"
        state["active_run"] = None
        save_state(state)
        write_final_report(state)
        return 0 if state["status"] == "completed" else 130
    except Exception:
        state["status"] = "failed"
        state["fatal_error"] = traceback.format_exc()
        state["active_run"] = None
        save_state(state)
        write_final_report(state)
        raise


if __name__ == "__main__":
    sys.exit(main())
