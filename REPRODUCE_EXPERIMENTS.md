# Reproducing the completed 45-run experiment matrix

This document records how the finalized `mondrian_cbvf_full_suite_v1` experiment was executed. It was reconstructed without rerunning training from:

- `experiments/full_suite.json`;
- `results/run_state.json`, whose `command_shell` fields are the authoritative verbatim expanded commands;
- each run's `results/runs/<run_id>/batch_configuration.json`;
- `experiments/run_full_suite.py` and `experiments/launch_full_suite.sh`;
- the per-run logs in `results/logs/`;
- the saved post-training coverage artifacts and finalization scripts.

The suite state is `completed`, with 45 of 45 runs completed.

## 1. Working directory and environment

The exact working directory recorded by the batch driver was:

```bash
cd '/home/mars/Desktop/conformal shield'
```

The original launcher did **not** activate Conda. It invoked this interpreter directly:

```bash
/home/mars/miniconda3/envs/odp/bin/python3.8
```

The following activation command is **reconstructed from the saved interpreter path**, not recorded verbatim in the experiment logs:

```bash
source /home/mars/miniconda3/etc/profile.d/conda.sh
conda activate odp
```

Using the absolute interpreter is the closest reproduction of the recorded execution and does not require activation.

The recorded launcher also established the following plotting environment:

```bash
export MPLBACKEND=Agg
export MPLCONFIGDIR="${TMPDIR:-/tmp}/mondrian-matplotlib"
mkdir -p "$MPLCONFIGDIR" results/logs
```

## 2. Setup, CBVF computation, calibration, and evaluation workflow

### Recorded project workflow commands

The finalized scientific report records this workflow:

```bash
/home/mars/miniconda3/envs/odp/bin/python3.8 train/check_shielded_training_workflow.py
/home/mars/miniconda3/envs/odp/bin/python3.8 experiments/run_full_suite.py --initialize-only
bash experiments/launch_full_suite.sh --retry-failed
```

`train/check_shielded_training_workflow.py` performs static workflow checks. `--initialize-only` expands the frozen manifest into `results/run_state.json` without training.

There was no separate CBVF-preprocessing command for each run. Every recorded training command included `--recompute-cbvf`; the trainer therefore rebuilt the finite-horizon CBVF before training using the specified model bounds, grid, solver, and target.

There was also no separate primary calibration or final-evaluation command. Each training invocation performed:

1. pretraining Mondrian calibration with 500 rollouts;
2. 250,000 training steps;
3. periodic evaluation every 25,000 steps using 5 episodes;
4. frozen-policy recalibration using 500 fresh rollouts after the last optimizer update;
5. final evaluation using 20 episodes and disjoint seeds;
6. offline calibrated and uncalibrated QP-feasibility audits;
7. per-run plot and result generation.

The relevant primary outputs were written below each run directory:

```text
training_and_evaluation_results.json
final_frozen_policy_mondrian_calibration.json
training_eval_metrics.jsonl
training_actions.jsonl
qp_feasibility_diagnostics.jsonl
params_25000.pkl ... params_250000.pkl
plots/
```

## 3. Exact recorded training-command expansion

The function below expands to the same argument list stored in every matching `results/run_state.json` `command` field. All flags and values in it are recorded configuration, not inferred defaults.

```bash
cd '/home/mars/Desktop/conformal shield'

run_one() {
  condition="$1"
  method="$2"
  seed="$3"

  case "$condition" in
    aligned)
      speed_min='-0.2'
      speed='0.6'
      beta_u='0.8'
      ;;
    easy)
      speed_min='-0.1'
      speed='0.6'
      beta_u='0.7'
      ;;
    hard)
      speed_min='0.0'
      speed='0.6'
      beta_u='0.6'
      ;;
    *)
      echo "Unknown condition: $condition" >&2
      return 2
      ;;
  esac

  run_id="${condition}_${method}_seed_${seed}"
  out="/home/mars/Desktop/conformal shield/results/runs/${run_id}"

  /home/mars/miniconda3/envs/odp/bin/python3.8 train/train_sac_lag.py \
    --algorithm sac_lag \
    --max-steps 250000 \
    --start-training 25000 \
    --warmup-policy reference \
    --reference-proposal-steps 75000 \
    --residual-reference-scale 0.02 \
    --batch-size 256 \
    --cost-limit 1.0 \
    --goal-bonus 750.0 \
    --collision-terminal-penalty 2500.0 \
    --unsafe-penalty 25.0 \
    --eval-interval 25000 \
    --save-interval 25000 \
    --eval-episodes 20 \
    --periodic-eval-episodes 5 \
    --n-calib 500 \
    --delta 0.05 \
    --mondrian-clearance-edges 0.25,0.75,1.50 \
    --dt 0.05 \
    --horizon 400 \
    --cbvf-nx 81 \
    --cbvf-ny 81 \
    --cbvf-nth 61 \
    --cbvf-dt 0.1 \
    --cbvf-terminal-guard 5.0 \
    --gamma 0.1 \
    --target-shape signed_distance \
    --cbvf-activate-margin 0.0 \
    --cp-activate-margin 0.1 \
    --epsilon-inter 0.0 \
    --speed-min "$speed_min" \
    --speed "$speed" \
    --beta-u "$beta_u" \
    --cbvf-model-speed-min -0.2 \
    --cbvf-model-speed 0.6 \
    --cbvf-model-beta-u 0.8 \
    --training-shield-method "$method" \
    --seed "$seed" \
    --checkpoint-dir "$out" \
    --dense-safety-cost \
    --no-tqdm \
    --recompute-cbvf \
    --terminate-on-collision \
    --odp-root '/home/mars/Desktop/conformal shield' \
    --solver-file '/home/mars/Desktop/conformal shield/solver_cbvf.py' \
    --results-json "$out/training_and_evaluation_results.json" \
    --plot-dir "$out/plots"
}
```

The coverage target is `1 - delta = 0.95`. The normalized action bounds were trainer defaults and therefore do not appear as command-line flags. No omitted command-line flag was added to the function above.

### The 45 recorded condition/method/seed commands

These calls enumerate the exact manifest order. Each call expands through `run_one` to the full recorded argument list above.

```bash
run_one aligned nominal 7
run_one aligned nominal 42
run_one aligned nominal 123
run_one aligned nominal 0
run_one aligned nominal 99

run_one aligned cbvf 7
run_one aligned cbvf 42
run_one aligned cbvf 123
run_one aligned cbvf 0
run_one aligned cbvf 99

run_one aligned cp 7
run_one aligned cp 42
run_one aligned cp 123
run_one aligned cp 0
run_one aligned cp 99

run_one easy nominal 7
run_one easy nominal 42
run_one easy nominal 123
run_one easy nominal 0
run_one easy nominal 99

run_one easy cbvf 7
run_one easy cbvf 42
run_one easy cbvf 123
run_one easy cbvf 0
run_one easy cbvf 99

run_one easy cp 7
run_one easy cp 42
run_one easy cp 123
run_one easy cp 0
run_one easy cp 99

run_one hard nominal 7
run_one hard nominal 42
run_one hard nominal 123
run_one hard nominal 0
run_one hard nominal 99

run_one hard cbvf 7
run_one hard cbvf 42
run_one hard cbvf 123
run_one hard cbvf 0
run_one hard cbvf 99

run_one hard cp 7
run_one hard cp 42
run_one hard cp 123
run_one hard cp 0
run_one hard cp 99
```

Method identifiers map as follows:

| Saved method | Paper label |
|---|---|
| `nominal` | Nominal RL |
| `cbvf` | Uncalibrated CBVF |
| `cp` | Mondrian conformal CBVF |

### Output directory for every command

Every invocation above writes to:

```text
/home/mars/Desktop/conformal shield/results/runs/<condition>_<method>_seed_<seed>/
```

The complete explicit mapping is:

| Condition | Method | Seeds | Output directories |
|---|---|---|---|
| aligned | nominal | 7, 42, 123, 0, 99 | `results/runs/aligned_nominal_seed_<seed>/` |
| aligned | cbvf | 7, 42, 123, 0, 99 | `results/runs/aligned_cbvf_seed_<seed>/` |
| aligned | cp | 7, 42, 123, 0, 99 | `results/runs/aligned_cp_seed_<seed>/` |
| easy | nominal | 7, 42, 123, 0, 99 | `results/runs/easy_nominal_seed_<seed>/` |
| easy | cbvf | 7, 42, 123, 0, 99 | `results/runs/easy_cbvf_seed_<seed>/` |
| easy | cp | 7, 42, 123, 0, 99 | `results/runs/easy_cp_seed_<seed>/` |
| hard | nominal | 7, 42, 123, 0, 99 | `results/runs/hard_nominal_seed_<seed>/` |
| hard | cbvf | 7, 42, 123, 0, 99 | `results/runs/hard_cbvf_seed_<seed>/` |
| hard | cp | 7, 42, 123, 0, 99 | `results/runs/hard_cp_seed_<seed>/` |

The corresponding stdout/stderr log is:

```text
/home/mars/Desktop/conformal shield/results/logs/<condition>_<method>_seed_<seed>.log
```

## 4. Compact launcher for the complete matrix

### Recorded launcher

This is the launcher actually provided by the project. It reads `experiments/full_suite.json`, expands the commands, appends each run's output to its log, and records state in `results/run_state.json`:

```bash
cd '/home/mars/Desktop/conformal shield'
bash experiments/launch_full_suite.sh --retry-failed
```

The automatically generated execution ledger also records this detached form:

```bash
tmux new-session -d -s mondrian-full-suite "cd '/home/mars/Desktop/conformal shield' && bash experiments/launch_full_suite.sh --retry-failed"
```

### Optional convenience loop — not the original batch execution

After defining `run_one`, the following directly invokes the same 45 expanded training commands, but bypasses the resumable state ledger and is therefore not the preferred reproduction path:

```bash
for condition in aligned easy hard; do
  for method in nominal cbvf cp; do
    for seed in 7 42 123 0 99; do
      run_one "$condition" "$method" "$seed"
    done
  done
done
```

## 5. Fresh held-out conformal coverage evaluation

The 15 saved CP artifacts named `heldout_reference_coverage_fresh.json` record:

- 100 evaluation episodes;
- `fresh_calibration: true`;
- no truth/model/horizon/residual overrides;
- no deployment evaluation.

The original shell command was not preserved in `run_state.json` or a dedicated log. The following is therefore **reconstructed from the saved artifact fields and the parser in `experiments/evaluate_reference_coverage.py`**. No flag below is uncertain, but shell-loop syntax and invocation order are not recoverable:

```bash
cd '/home/mars/Desktop/conformal shield'
for condition in aligned easy hard; do
  for seed in 7 42 123 0 99; do
    run_dir="results/runs/${condition}_cp_seed_${seed}"
    /home/mars/miniconda3/envs/odp/bin/python3.8 \
      experiments/evaluate_reference_coverage.py "$run_dir" \
      --episodes 100 \
      --fresh-calibration \
      --output "$run_dir/heldout_reference_coverage_fresh.json"
  done
done
```

## 6. Final evaluation, aggregation, audit, tables, and Figures 1–5

### Final evaluation

Primary final evaluation was integrated into each `train/train_sac_lag.py` command through:

```text
--eval-episodes 20
--periodic-eval-episodes 5
--eval-interval 25000
```

There was no separate primary final-evaluation command. The fresh guarantee-matched 100-episode coverage evaluation is documented above.

### Aggregation, tables, and figures

The following command creates the aggregate CSV/JSON tables, LaTeX tables, and Figures 1–5 in `results/aggregate/`:

```bash
cd '/home/mars/Desktop/conformal shield'
/home/mars/miniconda3/envs/odp/bin/python3.8 experiments/aggregate_full_suite.py
```

Outputs include:

```text
results/aggregate/final_metrics.csv
results/aggregate/final_metrics.json
results/aggregate/main_results_table.csv
results/aggregate/main_results_table.tex
results/aggregate/certification_feasibility_table.csv
results/aggregate/certification_feasibility_table.tex
results/aggregate/figure1_learning_comparison.png
results/aggregate/figure1_learning_comparison.pdf
results/aggregate/figure2_trajectories_boundaries.png
results/aggregate/figure2_trajectories_boundaries.pdf
results/aggregate/figure3_conformal_calibration.png
results/aggregate/figure3_conformal_calibration.pdf
results/aggregate/figure4_feasibility_verification.png
results/aggregate/figure4_feasibility_verification.pdf
results/aggregate/figure5_safety_performance_summary.png
results/aggregate/figure5_safety_performance_summary.pdf
```

### Acceptance audit

```bash
cd '/home/mars/Desktop/conformal shield'
/home/mars/miniconda3/envs/odp/bin/python3.8 experiments/audit_full_suite.py
```

Outputs:

```text
results/aggregate/acceptance_audit.json
results/aggregate/ACCEPTANCE_AUDIT.md
```

### Scientific report generation

```bash
cd '/home/mars/Desktop/conformal shield'
/home/mars/miniconda3/envs/odp/bin/python3.8 experiments/build_scientific_report.py
```

Output:

```text
results/SCIENTIFIC_FINAL_REPORT.md
```

### Recorded combined finalizer

The project also contains a watcher that waits for `results/run_state.json` to become `completed`, then aggregates, audits, and generates the scientific report:

```bash
cd '/home/mars/Desktop/conformal shield'
bash experiments/finalize_when_complete.sh
```

This command is recorded in the project script, but no saved shell transcript proves whether the wrapper itself or its three constituent Python commands were used for the finalized artifacts.

## 7. Safe resume behavior

The recorded resumable command is:

```bash
cd '/home/mars/Desktop/conformal shield'
bash experiments/launch_full_suite.sh --retry-failed
```

On startup, `experiments/run_full_suite.py` recomputes each configuration hash. A run is preserved as completed only when all three conditions hold:

1. its previous status is `completed`;
2. its saved configuration SHA-256 matches the current frozen manifest expansion;
3. its primary `training_and_evaluation_results.json` exists.

Those valid completed runs are skipped. Failed runs are reset to pending only because `--retry-failed` is supplied. Pending or interrupted runs are executed in manifest order.

To inspect what would be resumed without starting work:

```bash
cd '/home/mars/Desktop/conformal shield'
/home/mars/miniconda3/envs/odp/bin/python3.8 experiments/run_full_suite.py --initialize-only --retry-failed
```

This updates the state/status ledger but does not run training. Because initialization changes `results/run_state.json`, it should not be used merely to inspect an already finalized archival copy unless updating that ledger is intended.

The detached resume command recorded in the final execution ledger is:

```bash
tmux new-session -d -s mondrian-full-suite "cd '/home/mars/Desktop/conformal shield' && bash experiments/launch_full_suite.sh --retry-failed"
```

## 8. Provenance and unrecovered details

### Recorded verbatim

- All 45 argument arrays and shell-quoted commands in `results/run_state.json`.
- The same per-run argument arrays in `batch_configuration.json`.
- The run output and log paths.
- The manifest order and configuration values.
- The batch launcher, resume logic, aggregation, audit, and report-generation commands in project scripts and reports.

### Reconstructed from saved configuration

- `conda activate odp`: the environment name is recoverable from the absolute interpreter path, but activation was not used or logged by the launcher.
- The 15 fresh held-out coverage invocations: their effective flags are recoverable from the saved JSON fields, but their original shell-loop syntax, command order, and whether absolute or environment Python was typed are not recorded.

### Optional commands not originally established by the ledger

- The direct `run_one` convenience function and nested 45-run loop. They expand to the recorded argument lists but bypass the batch state manager.
- The Conda activation form; the recorded execution used the absolute Python interpreter.

No credentials, tokens, temporary process IDs, or machine-specific tmux process identifiers are included.
