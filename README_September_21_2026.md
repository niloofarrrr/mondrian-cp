Last updated: September 21, 2026

# Final accepted ECC case study — reproducibility reference

This README is the **single authoritative guide to the final paper results, their commands, configuration, and locations**. Numerical reports linked below are supporting artifacts. All repository-relative paths are relative to `/home/mars/Desktop/conformal shield`.

## 1. FINAL ACCEPTED CASE STUDY

The authoritative final study is:

```text
results_mechanism_v6/study_001
```

This accepted matched-policy study compares **Aligned, Easy Mismatch, and Hard Mismatch**, each under **Nominal RL, uncalibrated CBVF, and conformal-CBVF**. Five independently trained actors are frozen; each actor is reused across all conditions and filters, with paired initial conditions across the three filters in each condition–seed block. Five training runs precede **45 evaluation runs, 100 episodes each (4,500 primary episodes; 500 per condition/method)**. All final feasibility, held-out coverage, and inter-sample checks passed.

| Hard-Mismatch method | Unsafe episodes | Goal successes |
| --- | ---: | ---: |
| Nominal RL | 500/500 | 0/500 |
| Uncalibrated CBVF | 500/500 | 0/500 |
| Conformal-CBVF | 0/500 | 500/500 |

All methods achieved zero unsafe episodes and 100% goal success in Aligned and Easy. There were **30 exhaustive grid audits**, **968,226,000 eligible node/interval combinations**, zero calibrated or uncalibrated infeasible cases, and strictly positive available margins. **29,675,759 executed active-node checks** and **449,040 reconstructed representative intervals** passed. Held-out reference coverage was **1,490/1,500 (99.33%)**, ranging from **96/100 to 100/100** across 15 blocks; every block met the predeclared empirical threshold of **95/100**. The formal conformal target is **99%**, not the empirical acceptance threshold.

Feasibility is established on the declared finite grid and executed trajectories, not every continuous state. The split-conformal reference population is uncalibrated-filter rollouts of the same frozen actor; deployment coverage is separately empirical. These scope statements apply to all paper claims.

## 2. EXACT COMMANDS USED FOR THE FINAL STUDY

### 2.1 Historical main launch command

**Working directory for every command in this README:** `/home/mars/Desktop/conformal shield`, unless an explicit `cd` states otherwise.

The recorded persistent launch was:

```bash
tmux -S /tmp/mondrian-full-suite.sock new-session -d -s mechanism-persistent-study -c '/home/mars/Desktop/conformal shield' 'export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1; exec /home/mars/miniconda3/envs/odp/bin/python3.8 experiments/run_analytic_study.py > results_mechanism_v6/persistent_workflow.log 2>&1'
```

Produces the persistent workflow log and orchestrates training, evaluations, audits, aggregation, and acceptance. The equivalent main Python process command is:

```bash
/home/mars/miniconda3/envs/odp/bin/python3.8 experiments/run_analytic_study.py
```

The outer `tmux` launch is recovered from the recorded session command, not from a persisted shell-argv file. Repository evidence for the Python entrypoint and its child commands is `experiments/run_analytic_study.py`, its identical accepted snapshot at `results_mechanism_v6/study_001/source_snapshot/experiments/run_analytic_study.py`, `results_mechanism_v6/persistent_workflow.log`, and `results_mechanism_v6/persistent_status.json`.

**Do not rerun that historical launcher to reproduce only the accepted study:** it traverses the development queue and is not an accepted-study-only CLI. Use the new accepted-only replay command in §2.6. No historical configuration is needed for that replay.

The accepted block's recorded UTC stages are:

| Stage | Recorded UTC |
| --- | --- |
| Frozen configuration; five-actor training started | 2026-09-16 19:38:07 |
| 45 matched evaluations started | 2026-09-16 19:51:36 |
| Paper generation and acceptance started | 2026-09-16 19:56:44 |
| Accepted paper-ready result | 2026-09-16 19:56:54 |

### 2.2 Exact child-command environment

For the historical component commands below, initialize:

```bash
cd '/home/mars/Desktop/conformal shield'
ECC_PYTHON=/home/mars/miniconda3/envs/odp/bin/python3.8
ECC_STUDY="$PWD/results_mechanism_v6/study_001"
ECC_SCRIPTS="$ECC_STUDY/source_snapshot/experiments"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLBACKEND=Agg
export MPLCONFIGDIR=/tmp/mechanism-v6-matplotlib
```

Sets the recorded interpreter, accepted paths, and child-process environment. The interpreter is Python **3.8.18** in the `odp` environment. The complete recorded package versions are in `results_mechanism_v6/study_001/environment.json`; key versions are NumPy 1.24.4, JAX/JAXLIB 0.4.13, Flax 0.7.2, Optax 0.1.7, Gymnasium 0.29.1, Matplotlib 3.7.3, and Distrax 0.1.3. These commands use the existing environment; no guessed dependency-install command is substituted for the recorded environment.

The loops below expand to the **exact accepted child-process arguments** reconstructed from the runner and cross-checked against per-actor/per-run metadata. The original runner used up to three concurrent workers, so the loops express submission order and dependencies, not an invented serial wall-clock order. Their original output directories already exist: treat §2.3–2.5 as the historical command record. Training/evaluation entrypoints refuse existing output directories; report entrypoints overwrite derived reports. Use §2.6, §7, or §8 for a new replay without touching accepted files.

### 2.3 Train the five frozen actors

```bash
for ECC_SEED in 111000001 112000001 113000001 114000001 115000001; do
  "$ECC_PYTHON" "$ECC_SCRIPTS/train_analytic_actor.py" \
    --configuration "$ECC_STUDY/effective_configuration.json" \
    --out "$ECC_STUDY/actors/seed_$ECC_SEED" \
    --seed "$ECC_SEED"
done
```

Produces `params_20000.pkl`, the effective training configuration, training episode/progress logs, and completion metadata for each actor. The runner captured stdout/stderr in `actors/seed_<actor-seed>.log`. Training uses Aligned dynamics without a filter; each learned residual is frozen and shared by all subsequent filter comparisons.

### 2.4 Calibrate, evaluate all conditions/methods, certify, and audit

```bash
for ECC_SEED in 111000001 112000001 113000001 114000001 115000001; do
  for ECC_CONDITION in aligned easy hard; do
    case "$ECC_CONDITION" in
      aligned) ECC_OFFSET=200000 ;;
      easy)    ECC_OFFSET=300000 ;;
      hard)    ECC_OFFSET=400000 ;;
    esac
    ECC_EVAL_SEED=$((ECC_SEED + ECC_OFFSET))
    ECC_RUN="$ECC_STUDY/runs/${ECC_CONDITION}_seed${ECC_SEED}"
    "$ECC_PYTHON" "$ECC_SCRIPTS/qualify_analytic_cbvf.py" \
      --configuration "$ECC_STUDY/effective_configuration.json" \
      --out "$ECC_RUN" \
      --seed "$ECC_EVAL_SEED" \
      --condition "$ECC_CONDITION" \
      --actor "$ECC_STUDY/actors/seed_$ECC_SEED/params_20000.pkl"
    "$ECC_PYTHON" "$ECC_SCRIPTS/audit_analytic_intervals.py" "$ECC_RUN"
  done
done
```

The first command produces one complete condition–seed block, including all three methods. The second independently reconstructs every saved representative interval and writes `independent_interval_audit.json`. The runner captured their logs in `runs/<condition>_seed<actor-seed>.log` and `.interval_audit.log` respectively. Although the entrypoint is named `qualify_analytic_cbvf.py`, the accepted calls load `design_only=false` from the frozen configuration and constitute the final evaluations.

Within **each first command**, the actual execution order is:

| Order | Operation | Accepted output in that run directory |
| ---: | --- | --- |
| 1 | Save effective configuration and disjoint calibration/evaluation/coverage starts | `configuration.json`, `initial_states.npz` |
| 2 | Run 199 uncalibrated-filter calibration trajectories | `calibration.json`, `calibration_intervals.csv` |
| 3 | Compute regional conformal order statistics | `buffers.json` |
| 4 | Exhaustively audit both QPs using those freshly computed buffers | `feasibility_audit.json` |
| 5 | Run 100 Nominal RL episodes | `performance_nominal.json`, `performance_nominal_intervals.csv` |
| 6 | Run the paired 100 uncalibrated CBVF episodes | `performance_cbvf.json`, `performance_cbvf_intervals.csv` |
| 7 | Run the paired 100 conformal-CBVF episodes | `performance_cp.json`, `performance_cp_intervals.csv` |
| 8 | Run 100 fresh held-out uncalibrated-reference trajectories | `heldout_reference.json`, `heldout_reference_intervals.csv` |
| 9 | Evaluate CP deployment on the held-out starts | `heldout_deployment.json`, `heldout_deployment_intervals.csv` |
| 10 | Summarize coverage and all condition-level acceptance criteria | `coverage.json`, `qualification.json`, `status.json` |

Executed-node QP feasibility and analytic inter-sample bounds are checked **inside every filtered rollout** by the frozen `analytic_cbvf_batch.py`. They are not a later optional switch. Their counts and minima are stored in each rollout JSON (`interval_checks`, `min_constraint_margin`, `min_true_interval_margin`, and per-episode `min_qp`).

There were **no separate historical CLI commands** for calibration-only, Nominal-only, uncalibrated-only, CP-only, held-out-only, exhaustive-audit-only, or a separately named “final conformal certification.” Those are integrated stages above. The accepted CLI has no `--method` option. Final conformal certification comprises the fresh grid audit, strict executed-node checks, held-out coverage, condition criteria, independent interval reconstruction, and final study acceptance; no additional certificate executable is implied.

### 2.5 Aggregate, generate figures/tables, and check scientific acceptance

After all 15 blocks and their interval audits completed, the runner wrote `final_gate.json` and invoked:

```bash
"$ECC_PYTHON" "$ECC_SCRIPTS/build_analytic_report.py" "$ECC_STUDY"
```

Produces aggregate metrics, tables, the initial six figures, `PAPER_HANDOFF.md`, `ARTIFACT_INDEX.json`, and `SCIENTIFIC_ACCEPTANCE.json`; it checks the frozen inputs, all blocks, actor matching, feasibility, coverage, intervals, and acceptance. The accepted stdout is `results_mechanism_v6/study_001/build_report.log`. Aggregation, table creation, figure creation, and final scientific acceptance are bundled in this command, not separate undocumented CLIs.

The recorded subsequent paper extraction command was:

```bash
/home/mars/miniconda3/envs/odp/bin/python3.8 experiments/extract_analytic_paper_numbers.py results_mechanism_v6/study_001
```

Produces `PAPER_NUMBERS.md` and `aggregate/hard_same_state_mechanism.csv` from accepted outputs, also checking that Aligned zero-buffer comparisons agree within roundoff tolerance.

The recorded presentation-finalization command was:

```bash
MPLCONFIGDIR=/tmp/mechanism-v6-matplotlib /home/mars/miniconda3/envs/odp/bin/python3.8 experiments/finalize_analytic_figures.py results_mechanism_v6/study_001
```

Produces the final `matched_trajectories.pdf/.png` with the complete trajectories visible and the final `calibration_buffers_coverage.pdf/.png` with consistent region colors; writes `PRESENTATION_VERIFICATION.json`. It does not change simulations or numerical results. The identical archived copies of these two scripts are in `results_mechanism_v6/study_001/postprocessing_source/` and are preferred for replay.

`tables/runtime_feasibility.csv`, its report addenda, and final artifact-index additions were originally generated by an inline Python calculation, **not an archived standalone CLI**. No separate historical command filename can be recovered for that step. The new replay helper's `supplement()` function preserves the calculation: sum `interval_checks` and episode counts, take minima across calibration/primary/held-out uncalibrated runs or primary/held-out CP runs, and verify positive available margins. Its regenerated table is checked against the accepted table in §10.

### 2.6 Accepted-only full replay — new command, not the historical launcher

A documentation/replay helper was added on September 21, 2026:

```bash
cd '/home/mars/Desktop/conformal shield'
/home/mars/miniconda3/envs/odp/bin/python3.8 experiments/reproduce_accepted_case_study.py full \
  --out results_reproduced/study_001_full
```

Replays **only the accepted configuration**: trains five actors, runs the 15 condition–seed blocks/45 method evaluations, independently verifies intervals, aggregates, checks acceptance, and applies both final paper postprocessors. It invokes the unchanged frozen scripts with the same seeds, parameters, environment, and up-to-three-worker scheduling; only output paths are redirected. It copies frozen inputs into the new destination, preserves accepted results, and stops if a replay fails a check. It does not traverse any design queue or tune parameters.

To prepare the already frozen actors rather than retrain:

```bash
/home/mars/miniconda3/envs/odp/bin/python3.8 experiments/reproduce_accepted_case_study.py full \
  --reuse-actors --out results_reproduced/study_001_existing_actors
```

Copies the five accepted checkpoints and training records, then reruns the same calibration, 45 evaluations, coverage, audits, and paper pipeline.

Add `--dry-run` to either command to verify accepted inputs and print the exact child command lines without creating outputs or running simulations. Every replay destination must be new; the helper refuses existing directories and paths within the accepted study. The helper is **new replay tooling**, not evidence that it was used to create the original accepted study.

## 3. EXACT FINAL CONFIGURATION

| File | Exact repository-relative path | Role |
| --- | --- | --- |
| `effective_configuration.json` | `results_mechanism_v6/study_001/effective_configuration.json` | Frozen master configuration used by every final training/evaluation command |
| `environment.json` | `results_mechanism_v6/study_001/environment.json` | Recorded Python executable, package versions, and thread settings |
| `FREEZE.json` | `results_mechanism_v6/study_001/FREEZE.json` | Final seeds and integrity checks for frozen source/configuration |
| `training_configuration.json` | `results_mechanism_v6/study_001/actors/seed_<actor-seed>/training_configuration.json` | Exact effective parameters for each of the five training runs |
| `configuration.json` | `results_mechanism_v6/study_001/runs/<condition>_seed<actor-seed>/configuration.json` | Exact condition-specific dynamics, checkpoint, calibration size, and RNG seeds |
| `buffers.json` | `results_mechanism_v6/study_001/runs/<condition>_seed<actor-seed>/buffers.json` | Final seed-specific calibrated values and quantile orders |

Here `<condition>` is exactly `aligned`, `easy`, or `hard`; `<actor-seed>` is one of the five rows below. These templates enumerate actual files, not extra configurations to choose or tune.

| Actor seed | Aligned calibration/base seed | Easy calibration/base seed | Hard calibration/base seed |
| ---: | ---: | ---: | ---: |
| 111000001 | 111200001 | 111300001 | 111400001 |
| 112000001 | 112200001 | 112300001 | 112400001 |
| 113000001 | 113200001 | 113300001 | 113400001 |
| 114000001 | 114200001 | 114300001 | 114400001 |
| 115000001 | 115200001 | 115300001 | 115400001 |

Within each condition, primary-evaluation RNG seed = base seed + 10,000; held-out RNG seed = base seed + 20,000. Exact starts are saved in `initial_states.npz`. Training initialization uses the actor seed, replay sampling uses actor seed + 2, and training resets use actor seed + 100,000. The three methods within a condition–seed block share the same primary starts and frozen actor.

**Dynamics and geometry.** For normalized `(a,b)` in `[-1,1]^2`, `x_dot=v(a)cos(theta)`, `y_dot=v(a)sin(theta)`, `theta_dot=beta*b`:

| Condition | Forward-speed map `v(a)` | `beta` | Minimum/maximum speed |
| --- | --- | ---: | --- |
| Approximate model / Aligned | `0.2 + 0.4*a` | 0.8 | -0.2 / 0.6 |
| Easy | `0.2125 + 0.3875*a` | 0.625 | -0.175 / 0.6 |
| Hard | `0.225 + 0.375*a` | 0.43 | -0.15 / 0.6 |

Workspace/audit box is `[-4,4]^2`; the safety boundary is the origin-centered obstacle of radius 0.5 m inflated by robot radius 0.2 m, giving radius 0.7 m. Goal is `(0,1.8)` with radius 0.5 m. Resets are independent uniforms centered at `(-0.9,-3,0.7)` with half-width `0.02` in all three coordinates. Horizon is 25 s; sampling interval is 0.002 s (at most 12,500 intervals). Integration and interval collision detection use exact held-control unicycle flow and exact arc minima.

**CBVF/filter.** The CBVF is the analytic value `B=h=hypot(x,y)-0.7` on the safe domain; no numerical CBVF-construction grid is used. The exhaustive **audit** grid is `31 x 31 x 25`, spanning `[-4,4]^2` and 25 periodic headings. `gamma=1.5`; both filters activate at `B<=1`, with no hysteresis. Both project the same actor proposal into the same actuator box with the same inter-sample term. Uncalibrated requires `Psi_model >= epsilon`; CP requires `Psi_model >= epsilon + xi`. There is no infeasibility fallback or safety-constraint clipping.

**Calibration.** There are two Mondrian regions: `h<0.75` and `h>=0.75`. There are 199 calibration trajectories per condition–seed block, `delta_traj=0.01`, and regional `delta_m=0.005`; quantile order is `ceil(200*0.995)=199`. Every calibration trajectory visits both regions. The reachable-region maximum buffer is applied. The formal target is 99% reference-trajectory coverage; empirical acceptance is at least 95/100 in each of the 15 held-out blocks. Aligned buffers are exactly zero. Seed-specific buffers are read from the 15 `buffers.json` files and summarized in `tables/coverage_and_buffers.csv` and `tables/buffer_summary.csv`.

**Inter-sample parameters.** Public physical bounds are `|v|<=0.6`, `|omega|<=0.8`, giving `C_x=1`. With `r=hypot(x,y)` and `Delta=0.002`,

```text
L_Psi = hypot(gamma + 0.6/(r - 0.6*Delta), 0.6)
epsilon_inter = L_Psi*(C_x+1)*Delta = 0.004*L_Psi
```

**Shared actor/reference.** `u_nom=clip(u_ref+0.02*pi_theta,-1,1)`; the residual network is evaluated deterministically. Before 8 s, the waypoint is `(-0.9,0.65)` while `y<0.45`. From 8 s, the waypoint is `(-1.4,-1.2)` while `x>-1.2` and `y<0.45`, otherwise `(-1.4,0.65)` while `y<0.45`. At `y>=0.45`, the target is the goal. Throttle is -1 during `[8,10)` s and +1 otherwise; yaw is `clip(1.5*wrapped_heading_error/0.8,-1,1)`. The orbit trigger is zero. This entire reference, including the reverse maneuver, is identical across filters.

**Training defaults.** SAC-Lag; 20,000 steps; start training and reference-proposal cutoff at 1,000; replay capacity 20,000; batch size 256; update ratio 1; two 256-unit ReLU hidden layers; actor/critic/temperature/Lagrange learning rates all 0.0003; discount 0.99; target update `tau=0.005`; two Qs; `num_min_qs=None`; no critic dropout or layer normalization; target entropy -1; initial temperature 0.1; initial Lagrange value 0; entropy backup enabled; cost limit 1. Observation is `[x,y,cos(theta),sin(theta),0,1.8]`. Reward is `-Delta*distance_to_goal(next_state)+100*goal-100*collision`; collision cost is 1, otherwise 0. Goal/collision terminate; horizon truncation bootstraps. Training uses Aligned dynamics and no filter. Every remaining effective default is explicitly stored in the master/training JSON files.

## 4. WHERE ALL FINAL RESULTS ARE STORED

```text
results_mechanism_v6/study_001/
├── effective_configuration.json
├── environment.json
├── FREEZE.json
├── actors/seed_<actor-seed>/
├── runs/aligned_seed<actor-seed>/
├── runs/easy_seed<actor-seed>/
├── runs/hard_seed<actor-seed>/
├── source_snapshot/experiments/
├── source_snapshot/jaxrl5/
├── postprocessing_source/
├── aggregate/
├── tables/
├── figures/
├── final_gate.json
├── SCIENTIFIC_ACCEPTANCE.json
├── PAPER_NUMBERS.md
├── PAPER_HANDOFF.md
└── ARTIFACT_INDEX.json
```

In the following table, `S` denotes the exact prefix `results_mechanism_v6/study_001`; `R` denotes any of the 15 actual directories `S/runs/<condition>_seed<actor-seed>` enumerated by the condition names and seed table above.

| Exact path/pattern | Contents |
| --- | --- |
| `S/actors/seed_<actor-seed>/params_20000.pkl` | Frozen learned actor checkpoint |
| `S/actors/seed_<actor-seed>/training_configuration.json` | Per-actor training parameters |
| `S/actors/seed_<actor-seed>/training_episodes.jsonl` | Completed training episode returns/outcomes |
| `S/actors/seed_<actor-seed>/training_progress.jsonl` | Training step progress |
| `S/actors/seed_<actor-seed>/training_complete.json` | Training completion and outcome counts |
| `S/runs/aligned_seed<actor-seed>/` | Raw Aligned outputs for all three methods |
| `S/runs/easy_seed<actor-seed>/` | Raw Easy Mismatch outputs for all three methods |
| `S/runs/hard_seed<actor-seed>/` | Raw Hard Mismatch outputs for all three methods |
| `R/configuration.json`, `R/initial_states.npz` | Exact block configuration and all initial-state splits |
| `R/performance_nominal.json` | 100 Nominal episode records and summary |
| `R/performance_cbvf.json` | 100 uncalibrated-CBVF episode records and summary |
| `R/performance_cp.json` | 100 conformal-CBVF episode records and summary |
| `R/performance_<method>_intervals.csv` | Every interval of the first preselected primary trajectory; method is `nominal`, `cbvf`, or `cp` |
| `R/calibration.json`, `R/calibration_intervals.csv` | Calibration scores, episode records, and first representative trajectory |
| `R/buffers.json` | Final regional buffers, calibration counts, deltas, and quantile orders |
| `R/heldout_reference.json`, `R/heldout_reference_intervals.csv` | Fresh held-out reference results and first representative trajectory |
| `R/heldout_deployment.json`, `R/heldout_deployment_intervals.csv` | Separate CP deployment coverage diagnostic |
| `R/coverage.json` | Reference/deployment covered counts, target, and empirical threshold |
| `R/feasibility_audit.json` | Exhaustive calibrated/uncalibrated grid margins, counts, and inter-sample bounds |
| `R/calibration.json`, `R/performance_cbvf.json`, `R/performance_cp.json`, `R/heldout_reference.json`, `R/heldout_deployment.json` | Executed-node counts and minima; there is no separate all-node action-log file |
| `R/independent_interval_audit.json` | Independent representative interval reconstruction and Lipschitz checks |
| `R/qualification.json`, `R/status.json` | Completed condition–seed acceptance criteria |
| `S/aggregate/` | Episode-level, seed-level, pooled, and same-state mechanism data |
| `S/tables/` | Paper tables and certification summaries |
| `S/figures/` | Final PDF figures and PNG counterparts |
| `S/final_gate.json` | All 15 block results plus final suite gate |
| `S/SCIENTIFIC_ACCEPTANCE.json` | Final paper acceptance results |
| `S/PAPER_NUMBERS.md` | Concise accepted-only numerical extraction |
| `S/PAPER_HANDOFF.md` | Full paper-ready handoff |
| `S/ANALYTIC_CERTIFICATE_DERIVATION.md` | Analytic CBVF, mismatch, and inter-sample derivation |
| `S/ARTIFACT_INDEX.json` | Supporting final artifact index |
| `S/PRESENTATION_VERIFICATION.json` | Final plot corrections and unchanged-simulation confirmation |

## 5. FINAL FIGURES USED FOR THE PAPER

`build_analytic_report.py` in the accepted `source_snapshot/experiments/` generates the initial figures. The finalization command in §2.5 then regenerates the two figures indicated below. PNG counterparts have identical basenames in the same folder.

| Paper role | Exact repository-relative path | What it shows | Generating command/script |
| --- | --- | --- | --- |
| **PRIMARY — main safety/performance comparison** | `results_mechanism_v6/study_001/figures/safety_goal_comparison.pdf` | Unsafe and goal rates for all methods/conditions, with across-seed SDs | §2.5 `build_analytic_report.py` |
| **PRIMARY — trajectories/safety boundary** | `results_mechanism_v6/study_001/figures/matched_trajectories.pdf` | Complete first preselected matched trajectories, inflated obstacle, and goal | §2.5 `build_analytic_report.py`, then `finalize_analytic_figures.py` |
| **PRIMARY — mismatch mechanism** | `results_mechanism_v6/study_001/figures/uncalibrated_mismatch_mechanism.pdf` | Hard model-safe but true-negative derivative condition before uncalibrated collision | §2.5 `build_analytic_report.py` |
| Calibration and buffers | `results_mechanism_v6/study_001/figures/calibration_buffers_coverage.pdf` | Regional buffer mean/SD and all 15 held-out coverage counts | §2.5 `build_analytic_report.py`, then `finalize_analytic_figures.py` |
| Feasibility/inter-sample verification | `results_mechanism_v6/study_001/figures/feasibility_intersample_verification.pdf` | Hard CP clearance, derivative tightening, QP margin, and true interval lower bound | §2.5 `build_analytic_report.py` |
| Optional training trace | `results_mechanism_v6/study_001/figures/shared_actor_training.pdf` | Logged returns of the five shared actors; not a convergence claim | §2.5 `build_analytic_report.py` |

## 6. FINAL TABLES AND NUMERICAL DATA

| Exact repository-relative path | Purpose | Generator |
| --- | --- | --- |
| `results_mechanism_v6/study_001/tables/main_comparison.tex` | Main paper LaTeX results table | §2.5 `build_analytic_report.py` |
| `results_mechanism_v6/study_001/tables/main_comparison.csv` | Full numerical comparison, means/SDs/counts | Same |
| `results_mechanism_v6/study_001/tables/main_comparison.md` | Readable main comparison | Same |
| `results_mechanism_v6/study_001/tables/coverage_and_buffers.csv` | All 15 coverage counts and seed-specific regional buffers | Same |
| `results_mechanism_v6/study_001/tables/buffer_summary.csv` | Buffer mean and sample SD across five seeds | Same |
| `results_mechanism_v6/study_001/tables/feasibility.csv` | 30 exhaustive method-specific grid audit summaries | Same |
| `results_mechanism_v6/study_001/tables/intersample_summary.csv` | Condition-wise `C_x`, `Delta`, `L_Psi`, epsilon ranges and reconstruction totals | Same |
| `results_mechanism_v6/study_001/tables/representative_intervals.csv` | All representative trace audit summaries | Same |
| `results_mechanism_v6/study_001/tables/runtime_feasibility.csv` | All executed active-node counts and minimum margins | Recorded inline calculation; now `supplement()` in `reproduce_accepted_case_study.py` |
| `results_mechanism_v6/study_001/aggregate/episodes.csv` | All 4,500 primary episode records | §2.5 `build_analytic_report.py` |
| `results_mechanism_v6/study_001/aggregate/per_seed.csv` | 45 seed/condition/method summaries | Same |
| `results_mechanism_v6/study_001/aggregate/summary.csv` | Nine pooled comparison rows and across-seed SDs | Same |
| `results_mechanism_v6/study_001/aggregate/hard_same_state_mechanism.csv` | Paired-state derivative, buffer, action and feasibility diagnostics | Frozen `build_analytic_report.py` generates it; §2.5 `extract_analytic_paper_numbers.py` regenerates it |

The raw JSON/CSV files in §4 and the five actors' `training_episodes.jsonl` files are the inputs required to rebuild these artifacts. Rates and returns are averaged per seed and then summarized using sample SD (`ddof=1`). Minimum clearance is a pooled continuous-arc minimum. Intervention rate is the mean of episode-level intervention fractions, with the pooled step-weighted rate separately available in the CSV.

## 7. REPRODUCING ONLY ONE CONDITION

These copy-paste-ready commands reuse the five accepted frozen checkpoints and the exact final configuration/seeds. They rerun only the requested condition, including fresh calibration, all three paired methods, held-out coverage, exhaustive/executed-node checks, and independent interval verification. They do not retrain. Output directories must not already exist.

**Aligned — working directory: repository root.**

```bash
cd '/home/mars/Desktop/conformal shield'
/home/mars/miniconda3/envs/odp/bin/python3.8 experiments/reproduce_accepted_case_study.py condition \
  --condition aligned --out results_reproduced/study_001_aligned
```

Produces the five Aligned blocks (15 method evaluations) under the new destination's `runs/` directory.

**Easy Mismatch — working directory: repository root.**

```bash
cd '/home/mars/Desktop/conformal shield'
/home/mars/miniconda3/envs/odp/bin/python3.8 experiments/reproduce_accepted_case_study.py condition \
  --condition easy --out results_reproduced/study_001_easy
```

Produces the five Easy blocks (15 method evaluations) and their calibration/coverage/audits.

**Hard Mismatch — working directory: repository root.**

```bash
cd '/home/mars/Desktop/conformal shield'
/home/mars/miniconda3/envs/odp/bin/python3.8 experiments/reproduce_accepted_case_study.py condition \
  --condition hard --out results_reproduced/study_001_hard
```

Produces the five Hard blocks (15 method evaluations) and their calibration/coverage/audits.

To reproduce a single accepted actor/condition block, add `--actor-seed 111000001` (or another accepted seed) and choose a new `--out` path. To inspect all exact child arguments without executing them, add `--dry-run`.

For example, the following direct frozen command reproduces the first Aligned block at a new output path; it is the historical accepted command with **only `--out` redirected**:

```bash
cd '/home/mars/Desktop/conformal shield'
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLBACKEND=Agg MPLCONFIGDIR=/tmp/mechanism-v6-matplotlib \
/home/mars/miniconda3/envs/odp/bin/python3.8 \
  results_mechanism_v6/study_001/source_snapshot/experiments/qualify_analytic_cbvf.py \
  --configuration results_mechanism_v6/study_001/effective_configuration.json \
  --out results_reproduced/aligned_seed111000001 \
  --seed 111200001 --condition aligned \
  --actor results_mechanism_v6/study_001/actors/seed_111000001/params_20000.pkl
```

Produces calibration, all three primary method evaluations, held-out results, and condition-level checks for that block. Follow it with:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLBACKEND=Agg MPLCONFIGDIR=/tmp/mechanism-v6-matplotlib \
/home/mars/miniconda3/envs/odp/bin/python3.8 \
  results_mechanism_v6/study_001/source_snapshot/experiments/audit_analytic_intervals.py \
  results_reproduced/aligned_seed111000001
```

Produces independent representative interval/Lipschitz verification for that reproduced block.

**Nominal-only / uncalibrated-only / conformal-only:** no standalone method-selecting CLI exists in the accepted scripts. Do not add a fictitious `--method` flag. The accepted condition command evaluates `nominal`, then `cbvf`, then `cp` on the same initial states and writes separate `performance_nominal.json`, `performance_cbvf.json`, and `performance_cp.json` files. Reading an individual method's completed result requires no rerun.

## 8. REGENERATING ONLY THE PAPER OUTPUTS

**Working directory: repository root.**

```bash
cd '/home/mars/Desktop/conformal shield'
/home/mars/miniconda3/envs/odp/bin/python3.8 experiments/reproduce_accepted_case_study.py paper \
  --out results_reproduced/study_001_paper
```

Copies the completed accepted actors/run outputs and frozen inputs into a new directory, then regenerates **all aggregate metrics, the main comparison table, coverage/buffer tables, feasibility tables, inter-sample summaries, final figures, paper reports, and final scientific acceptance**. It does **not** train actors, rerun simulation, recalibrate, or regenerate held-out samples. It reuses the already completed audits. This is the recommended single command for paper-output regeneration.

The exact postprocessing sequence used by this helper is:

```bash
ECC_PAPER="$PWD/results_reproduced/study_001_paper"
ECC_PYTHON=/home/mars/miniconda3/envs/odp/bin/python3.8
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLBACKEND=Agg
export MPLCONFIGDIR=/tmp/mechanism-v6-matplotlib
"$ECC_PYTHON" "$ECC_PAPER/source_snapshot/experiments/build_analytic_report.py" "$ECC_PAPER"
"$ECC_PYTHON" "$ECC_PAPER/postprocessing_source/extract_analytic_paper_numbers.py" "$ECC_PAPER"
"$ECC_PYTHON" "$ECC_PAPER/postprocessing_source/finalize_analytic_figures.py" "$ECC_PAPER"
```

These component commands are shown for inspection or rerunning a prepared **copy**; the helper creates that copy first and also performs the runtime-table/report supplement. The first command generates the aggregate data, tables, initial figures, and acceptance report; the second extracts paper numbers; the third applies the final trajectory and buffer-plot presentation. **Do not stop after the first command** if the goal is to reproduce the final figures. There are no separate table-only flags; the inexpensive combined builder reuses completed runs.

A paper-only replay regenerates scientific content, not byte-identical PDFs or provenance paths: PDF timestamps and report paths can differ. It does not replace the original study or its historical artifact hashes, and it does not allocate new evaluation seeds.

## 9. LEGACY / OLD RESULTS

> All other pilot, debug, retired-study, design-sweep, and earlier results directories are historical development artifacts and are NOT the final case-study results used in the paper.
>
> Do not delete them.
>
> The authoritative final study is: **`results_mechanism_v6/study_001`**.

The prior main command README was renamed to this file and rewritten for the accepted study. Its original text is preserved at `docs/history/README_FINAL_COMMANDS_before_September_21_2026.md`. Other older documentation is not an alternative source of final paper commands or numbers.

## 10. FINAL VERIFICATION

Run this read-only verification from the repository root:

```bash
/home/mars/miniconda3/envs/odp/bin/python3.8 experiments/reproduce_accepted_case_study.py verify
```

Checks the recorded frozen inputs, five actor checkpoint identities, all 15 condition configurations and seed mappings, accepted block criteria, and saved interval audit passes. The final scientific acceptance file is `results_mechanism_v6/study_001/SCIENTIFIC_ACCEPTANCE.json`; the suite gate is `results_mechanism_v6/study_001/final_gate.json`.

Documentation verification on September 21, 2026:

- The exact child arguments were cross-checked against the accepted runner snapshot, master config, all five training configs, all 15 condition configs, and their logs.
- All accepted file paths and both archived postprocessing scripts were checked against the filesystem; live postprocessors matched their archived copies.
- The historical `tmux` command was recovered from the recorded launch, while repository logs confirm the accepted stage progression. No persisted full shell-argv log is claimed.
- Calibration, method evaluation, coverage, grid audits, runtime checks, and acceptance were traced to their actual integrated stages; no nonexistent standalone command or CLI option is advertised.
- The new replay helper was syntax-checked and its full/condition dry runs verified. Full training and simulations were not unnecessarily rerun for this documentation change.
- Paper-only regeneration was actually executed in a separate temporary directory. Its numerical CSVs, coverage/buffer tables, feasibility tables, inter-sample summaries, and scientific acceptance were compared against the accepted study; verification details are recorded in `docs/verification/accepted_study_readme_verification_2026-09-21.json`.
- Accepted results, frozen source/configuration, checkpoints, and historical result folders were left unchanged.

For writing, use this README together with the accepted `PAPER_NUMBERS.md`, the three primary figures in §5, and `tables/main_comparison.tex` / `tables/main_comparison.csv`. No other study is a source of final paper results.
