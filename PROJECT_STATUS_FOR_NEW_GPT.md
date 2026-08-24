# Current project status

The project now trains SAC/SAC-Lag from scratch behind the finalized Mondrian conformal CBVF-QP.

Authoritative entry points:

- train/train_sac_lag.py: mandatory fresh-CBVF, pre-training-calibration, shielded training and evaluation pipeline.
- train/check_shielded_training_workflow.py: dependency-free static invariants.
- README_FINAL_COMMANDS.md: smoke and complete commands.
- run_aligned_no_mismatch.sh: complete aligned-model run after setting MONDRIAN_ODP_ROOT and MONDRIAN_SOLVER_FILE.

Important invariants:

- no pretrained checkpoint;
- no cached CBVF path;
- no training episode or optimizer update before calibration;
- baseline calibration uses CBVF-QP with xi=0;
- finalized Mondrian radial partition and strict-tree fallback;
- fixed pre-training buffers throughout training;
- every environment training step receives the filtered action;
- replay action and cost correspond to the executed transition;
- replay bootstrap mask is `1 - terminated`;
- Gym TimeLimit, internal environment horizon, and CBVF horizon are asserted equal;
- calibration and RL share the same environment-owned start sampler and certified-start check;
- nominal/CBVF/CP evaluation methods use identical per-episode start seeds;
- full runs require `n_effective_regions > 1` and report near-obstacle/candidate-region activity;
- no coverage claim is made for changing training-policy iterates;
- feasible CP actions are verified as exact minimum-distance projections;
- full runs abort before executing an infeasible CP fallback;
- a diagnostic-only override records whether the least-violation fallback is an actuator rail;
- each active CP evaluation step reports `xi_hat_app`, `epsilon_inter`, `max_u_psi`, and feasibility margin;
- unshielded evaluation is labeled as a counterfactual diagnostic, not a separately trained nominal baseline;
- no ACI and no post-training recalibration.

The current conformal_shield.py audited by the workflow check has SHA-256
f101c4520a0ae09eb0747d73f1533b2a3181aea7057a953a209cc5bd2ee40ce4.
See `results/FEASIBILITY_BLOCKER.md` for the full-resolution counterexample
that prevents a valid Assumption-6 full suite with the present first-order QP.
