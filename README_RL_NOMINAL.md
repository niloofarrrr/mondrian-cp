> Historical training documentation. For final ECC paper results and reproduction commands, use [README_September_21_2026.md](README_September_21_2026.md).

# RL policy training behind the finalized Mondrian shield

The RL policy is no longer trained nominally and calibrated afterward. The only training path in train/train_sac_lag.py is:

- fresh SAC or SAC-Lag initialization;
- fresh finite-horizon CBVF computation;
- pre-training Mondrian calibration using the frozen initial actor and baseline CBVF-QP with xi=0;
- conformal CBVF-QP filtering of every proposed training action.

The Gymnasium plant, reward, safety cost, SAC/SAC-Lag learners, replay buffer, and exact CBVF-QP are unchanged. Replay transitions now consistently use the action executed by the environment. A counterfactual nominal-action cost is logged for diagnosis but is never stored as the cost of the executed replay transition.

The replay mask is `1 - terminated`, so time-limit truncations bootstrap but genuine terminals do not. Gym's TimeLimit is driven by the configured CBVF horizon. Calibration and deployment use the environment's same start sampler and reject uncertified realized starts. Periodic nominal/CBVF/CP comparisons share identical seeded start states.

Full experiments require more than one effective Mondrian region and report the closest-region calibration count plus mean/max candidate effective regions. These diagnostics show whether the regional mechanism is actually active. They do not create a coverage claim for the changing policy during RL training.

QP feasibility is now strict by default. If the active calibrated constraint has `max_u_psi - (xi_hat_app + epsilon_inter) < 0`, training/evaluation stops before that fallback action is executed. A short diagnosis may explicitly use `--allow-infeasible-cp-diagnostic-run`; those results are labeled diagnostic and cannot be presented as validation of the calibrated QP. Every active evaluation step is written to `qp_feasibility_diagnostics.jsonl` for the configured number of episodes.

The unshielded method in evaluation is only a counterfactual test of the same actor trained behind the CP shield. It is not a separately trained nominal baseline, so its unsafe/goal rates must not be described as a fair shielded-versus-unshielded training comparison.

Use README_FINAL_COMMANDS.md for the smoke and complete commands. Do not run the standalone conformal_shield.py as the evaluation stage for this workflow, because that script intentionally calibrates the loaded deployment policy. Training already performs periodic and final evaluation with the fixed buffers created before training, avoiding post-training recalibration.
