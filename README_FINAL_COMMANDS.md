# Mondrian-shielded RL training

The training pipeline now uses the finalized Mondrian conformal CBVF shield from the beginning. conformal_shield.py is included unchanged and its SHA-256 is checked by train/check_shielded_training_workflow.py.

## Enforced order

1. Create a fresh SAC/SAC-Lag agent. No earlier checkpoint is restored.
2. Recompute the finite-horizon CBVF. Cached loading is not available in this training entry point.
3. Before the first training reset or optimizer update, freeze the initial actor and collect calibration rollouts under the baseline CBVF-QP with xi=0.
4. Construct the finalized radial-clearance Mondrian buffers using the strict-tree sparse-region fallback and uniform Bonferroni allocation from conformal_shield.py.
5. Route every training action through the calibrated conformal CBVF-QP. Store the filtered action, executed reward, executed cost, and resulting next observation in replay.
6. Keep the same CBVF table and Mondrian buffers fixed for all training and periodic/final evaluation. No post-training recalibration is performed.

The replay bootstrap mask is exactly `1 - terminated`; a TimeLimit truncation still bootstraps. Both Gym TimeLimits are set from `--horizon` and checked against the environment and CBVF horizons before calibration. Calibration and RL resets share the environment's initial-state sampler, and all three evaluation methods use identical seeded starts. After calibration, every stored certified CBVF grid node in the CP activation/hysteresis band is checked offline for `xi_hat_app + epsilon_inter <= max_u_psi`. Strict training begins only if this discrete gate passes; runtime gating covers interpolated states. The grid audit is not mislabeled as a continuous-envelope proof.

## Dependencies

Install the Python packages in requirements_rl.txt. Fresh CBVF computation also requires:

- the optimized_dp source tree containing the odp package; pass its parent with --odp-root;
- the local CBVF-capable HJ solver file defining HJSolver; pass it with --solver-file.

The program exits if either dependency is absent. It does not load a cached CBVF or continue without the shield.

## Static check

~~~bash
python train/check_shielded_training_workflow.py
python -m compileall -q .
bash -n run_aligned_no_mismatch.sh
MPLCONFIGDIR=/tmp/mondrian_matplotlib python train/smoke_test_mondrian_core.py
MPLCONFIGDIR=/tmp/mondrian_matplotlib python train/smoke_test_qp_projection.py
~~~

## 20k QP-feasibility diagnosis

Use the same model/truth and validated `--epsilon-inter` value as the experiment being diagnosed. The example below uses the easy mismatch (`beta_true=0.75`, `beta_model=0.50`) and explicitly sets `epsilon_inter=0` so the regional buffer can be separated from the inter-sample term.

~~~bash
python train/train_sac_lag.py \
  --algorithm sac_lag \
  --max-steps 20000 \
  --start-training 10000 \
  --batch-size 256 \
  --eval-interval 20000 \
  --save-interval 20000 \
  --eval-episodes 2 \
  --n-calib 500 \
  --delta 0.05 \
  --mondrian-clearance-edges 0.25,0.75,1.50 \
  --speed 0.60 \
  --beta-u 0.75 \
  --cbvf-model-speed 0.60 \
  --cbvf-model-beta-u 0.50 \
  --dt 0.05 \
  --horizon 400 \
  --cbvf-nx 81 \
  --cbvf-ny 81 \
  --cbvf-nth 61 \
  --cbvf-dt 0.10 \
  --epsilon-inter 0.0 \
  --qp-diagnostic-episodes 2 \
  --qp-diagnostic-print-limit 200 \
  --allow-infeasible-cp-diagnostic-run \
  --recompute-cbvf \
  --odp-root /path/to/optimized_dp \
  --solver-file /path/to/solver_cbvf.py \
  --checkpoint-dir rl_checkpoints/easy_mismatch_qp_diagnosis \
  --no-tqdm
~~~

For each active CP evaluation step, the console and `qp_feasibility_diagnostics.jsonl` report `xi_hat_app`, `epsilon_inter`, requested RHS, `max_u_psi`, feasibility margin, achieved margin, both actions, and the exact fallback classification. A negative feasibility margin means Assumption 5 failed at that state. The override is diagnosis-only; if the offline grid gate or any runtime feasibility check fails, the final status is `diagnostic_completed_with_cp_feasibility_failure` and the result is not a validated calibrated-QP experiment.

## Short smoke test

~~~bash
python train/train_sac_lag.py \
  --max-steps 20 \
  --start-training 10 \
  --batch-size 8 \
  --eval-interval 20 \
  --save-interval 20 \
  --eval-episodes 2 \
  --n-calib 20 \
  --delta 0.20 \
  --horizon 20 \
  --cbvf-nx 11 \
  --cbvf-ny 11 \
  --cbvf-nth 9 \
  --cbvf-dt 0.10 \
  --recompute-cbvf \
  --allow-single-effective-region \
  --allow-infeasible-cp-diagnostic-run \
  --odp-root /path/to/optimized_dp \
  --solver-file /path/to/solver_cbvf.py \
  --checkpoint-dir rl_checkpoints/smoke_mondrian_shielded \
  --no-tqdm
~~~

The two smoke-only overrides permit a deliberately sparse calibration and allow the finalized least-violation branch to be exercised if necessary. They do not change the partition, conformal method, or solver. The complete command below uses neither override: it fails before RL training if the partition collapses and stops before executing any infeasible CP action.

## Complete aligned-model run

~~~bash
python train/train_sac_lag.py \
  --algorithm sac_lag \
  --max-steps 250000 \
  --start-training 25000 \
  --batch-size 256 \
  --cost-limit 1.0 \
  --dense-safety-cost \
  --dense-safety-weight 0.25 \
  --safety-cost-margin 0.10 \
  --unsafe-penalty 1.0 \
  --collision-terminal-penalty 200.0 \
  --terminate-on-collision \
  --eval-interval 10000 \
  --save-interval 25000 \
  --eval-episodes 20 \
  --n-calib 500 \
  --delta 0.05 \
  --mondrian-clearance-edges 0.25,0.75,1.50 \
  --speed 0.60 \
  --beta-u 0.50 \
  --cbvf-model-speed 0.60 \
  --cbvf-model-beta-u 0.50 \
  --dt 0.05 \
  --horizon 400 \
  --cbvf-nx 81 \
  --cbvf-ny 81 \
  --cbvf-nth 61 \
  --cbvf-dt 0.10 \
  --gamma 0.10 \
  --target-shape signed_distance \
  --cbvf-activate-margin 0.0 \
  --cp-activate-margin 0.10 \
  --epsilon-inter 0.0 \
  --recompute-cbvf \
  --odp-root /path/to/optimized_dp \
  --solver-file /path/to/solver_cbvf.py \
  --checkpoint-dir rl_checkpoints/reach_avoid_mondrian_shielded_from_scratch
~~~

The explicit zero inter-sample term makes the provided aligned command reproducible and separates QP feasibility from an externally derived Lipschitz margin. It is not automatically a validated value for the paper. For a paper claim, replace it with the independently certified `L_Psi*(C_bar+1)*dt` value; the offline and runtime gates then test that exact requested RHS. The code never silently reduces or clips it to recover feasibility.

You can also set the two dependency variables and run the script:

~~~bash
export MONDRIAN_ODP_ROOT=/path/to/optimized_dp
export MONDRIAN_SOLVER_FILE=/path/to/solver_cbvf.py
bash run_aligned_no_mismatch.sh
~~~

## Outputs

The checkpoint directory contains:

- params_<step>.pkl: periodic and final checkpoints;
- policy_config.json: algorithm, environment, and enforced workflow metadata;
- pretraining_mondrian_calibration.json: the fixed buffers used during training;
- training_actions.jsonl: nominal, filtered, and environment-applied action per transition;
- qp_feasibility_diagnostics.jsonl: per-active-step CP feasibility breakdown for the selected evaluation episodes;
- training_eval_metrics.jsonl: periodic nominal/CBVF/CP comparisons;
- training_and_evaluation_results.json: final evaluation and generated-file index;
- plots/shielded_training_evaluation.png;
- plots/nominal_and_filtered_actions.png.

In the action plot, red `x` markers identify an infeasible least-violation fallback. A normal/full run cannot contain these markers because it aborts before executing the first infeasible CP action. The evaluation legend calls the no-filter curve an “unshielded actor diagnostic”: this actor was trained only behind the CP shield and is not an independently trained nominal baseline.

The calibration manifest and final results include `near_obstacle_score_count`, `per_base_leaf_minimum_without_merging`, `n_effective_regions`, and training/evaluation candidate-region statistics. At `n_calib=500`, `delta=0.05`, and four base regions, the unmerged per-leaf minimum is 79 scores. A full run requires `n_effective_regions > 1`; it does not claim that every base leaf remains unmerged.

Periodic nominal and CBVF evaluations are comparison diagnostics only. RL data collection itself is always performed with the Mondrian conformal CBVF-QP.

The paper's finite-sample coverage statement is for the fixed reference policy used during calibration. Because the actor changes during RL optimization, the same buffers are retained as deterministic safety margins, but the reference-policy coverage probability does not automatically transfer to later actor iterates.
