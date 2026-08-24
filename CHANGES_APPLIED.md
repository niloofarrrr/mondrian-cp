# Changes applied

## Training integration

- Replaced optional CBVF-only/nominal training with mandatory Mondrian conformal CBVF-QP training.
- Removed pretrained-checkpoint restoration from the training entry point.
- Hard-coded fresh CBVF recomputation and fail if a fresh table is not produced.
- Reused MondrianPartition, calibrate_mondrian, MondrianCalibration.applied_buffer, ShieldConfig, and solve_cbvf_projection_exact from the finalized conformal_shield.py.
- Calibration runs before the first RL environment reset and before the first optimizer update.
- Calibration uses a frozen copy of the initial actor and the baseline CBVF-QP with xi=0.
- The calibrated regional buffers remain fixed for the entire run.
- Every training action is checked by the CP shield; there is no unshielded or CBVF-only training flag.
- Replay stores the filtered action and the reward/cost/next state produced by that action.
- Added per-transition nominal/filtered/environment-action logging.
- Added periodic and final evaluation with the same pre-training buffers.
- Added checkpoint, calibration, result, and plot manifests.
- Added a dependency-free static workflow checker.

## Review corrections

- Fixed the replay bootstrap mask to `float(not terminated)`. Time-limit truncation still bootstraps, while a true terminal never does, including when `terminated` and `truncated` are simultaneously true.
- Removed the registration-time 400-step constant, passed `max_episode_steps=args.horizon` to both Gym environments, and added a startup assertion that the Gym TimeLimit, environment horizon, CBVF horizon, and time table agree.
- Added one environment-owned initial-state sampler. Calibration and RL resets now use that exact sampler, and the calibration `World` base state is derived from the environment configuration.
- Rejects any realized calibration, training, or evaluation start with a negative initial CBVF value instead of silently conditioning calibration and deployment on different start sets.
- Nominal, baseline-CBVF, and conformal evaluations now reuse identical per-episode seeds and therefore identical start states.
- Added near-obstacle calibration-count, effective-region, and candidate-region diagnostics. Full runs abort before RL training if strict-tree fallback produces only one effective region; `--allow-single-effective-region` is an explicit smoke/debug-only override.
- Results and manifests state that no finite-sample coverage claim attaches to changing training-policy iterates.

## QP-feasibility review correction

- Confirmed without modifying `conformal_shield.py` that its infeasible fallback is the least-violation affine maximizer. For nonzero control authority this is an actuator rail; with zero control authority it keeps the nominal action.
- Added active-step diagnostics for `xi_hat_app`, `epsilon_inter`, requested right-hand side, `max_u_psi`, feasibility margin, achieved constraint margin, affine terms, nominal/filtered actions, candidate regions, and fallback type/bound.
- Feasible solutions are independently checked against the analytic one-dimensional minimum-distance projection. A mismatch aborts immediately.
- Normal training now aborts before executing any infeasible CP action. The previous behavior can continue only with `--allow-infeasible-cp-diagnostic-run`, which marks the run/result as diagnostic and invalid for a calibrated-QP claim if any fallback occurs.
- Added a vectorized pre-training feasibility gate over every stored certified CBVF grid node in the CP activation plus hysteresis band. The manifest identifies this as a discrete exhaustive audit, not a continuous-envelope proof.
- Added `qp_feasibility_diagnostics.jsonl` and evaluation/training summary rates for active, infeasible, rail-fallback, and zero-authority steps.
- Marked the unshielded evaluation honestly as a counterfactual diagnostic of the CP-shield-trained actor, not an independently trained nominal baseline.
- Updated plots to mark infeasible fallback samples explicitly.
- Added `train/smoke_test_qp_projection.py` for feasible projection and both fallback branches.

## Shield corrections

The supplied shield was corrected to remove an inadmissible hard-coded
conformal-buffer ceiling and to support deployment lookup for a stationary
single-slice CBVF. Its audited SHA-256 is:

~~~text
SHA-256 f101c4520a0ae09eb0747d73f1533b2a3181aea7057a953a209cc5bd2ee40ce4
~~~

No ACI, alternate partition, or alternate conformal method was added.
