# Frozen calibration-mechanism case study

PASS: final scientific acceptance. Only this frozen study is summarized.

## Design and interpretation

The same frozen residual actor is evaluated under all three filters and conditions within each of five fresh training seeds. There are five 20,000-step nominal-model training runs and 45 final evaluation runs, with 100 episodes each. This isolates the filter effect; these are not 45 independently trained actors.

The common nominal route is successful without filtering in Aligned dynamics. Hard dynamics reduce steering and reverse authority. A preplanned recovery maneuver at 8–10 seconds reverses while steering toward an outward waypoint; the same schedule applies to every method. Uncalibrated Hard failures occur before recovery. Both filters enforce identical activation, gamma, actuator limits, and inter-sample corrections; CP alone adds the calibrated regional buffer.

## Exact frozen parameters

Full effective configuration and all training defaults: `/home/mars/Desktop/conformal shield/results_mechanism_v6/study_001/effective_configuration.json`. Frozen source checksums: `/home/mars/Desktop/conformal shield/results_mechanism_v6/study_001/FREEZE.json`.

```json
{
  "configuration": {
    "beta": 0.43,
    "vmin": -0.15,
    "orbit_trigger": 0.0,
    "waypoint_x": -0.9,
    "gamma": 1.5,
    "dt": 0.002,
    "duration": 25.0,
    "route_switch_time": 8.0,
    "route_waypoint_x": -1.4,
    "route_waypoint_y": -1.2,
    "route_gate_x": -1.2,
    "initial_theta": 0.7,
    "reset_jitter": 0.02,
    "reverse_duration": 2.0,
    "n_calibration": 199,
    "delta": 0.01,
    "design_only": false,
    "activation_threshold": 1.0,
    "residual_scale": 0.02
  },
  "training": {
    "algorithm": "sac_lag",
    "max_steps": 20000,
    "start_training": 1000,
    "reference_proposal_steps": 1000,
    "batch_size": 256,
    "hidden_dims": [
      256,
      256
    ],
    "actor_lr": 0.0003,
    "critic_lr": 0.0003,
    "temp_lr": 0.0003,
    "lag_lr": 0.0003,
    "discount": 0.99,
    "tau": 0.005,
    "num_qs": 2,
    "num_min_qs": null,
    "critic_dropout_rate": null,
    "critic_layer_norm": false,
    "target_entropy": -1.0,
    "init_temperature": 0.1,
    "init_lag": 0.0,
    "backup_entropy": true,
    "cost_limit": 1.0,
    "utd_ratio": 1,
    "residual_scale": 0.02,
    "replay_capacity": 20000,
    "training_condition": "aligned",
    "training_filter": "nominal",
    "observation": "[x,y,cos(theta),sin(theta),goal_x,goal_y]; time-dependent shared reference is composed outside residual network",
    "reward": "-dt*distance_to_goal(next_state)+100*goal-100*collision",
    "cost": "1 on exact-arc collision, 0 otherwise",
    "termination": "exact-arc collision or endpoint goal; horizon truncation bootstraps",
    "logging_interval": 1000
  },
  "conditions": {
    "aligned": {
      "vmin": -0.2,
      "beta": 0.8
    },
    "easy": {
      "vmin": -0.175,
      "beta": 0.625
    },
    "hard": {
      "vmin": -0.15,
      "beta": 0.43
    }
  },
  "model": {
    "vmax": 0.6,
    "vmin": -0.2,
    "beta": 0.8
  },
  "geometry": {
    "obstacle_center": [
      0,
      0
    ],
    "obstacle_radius": 0.5,
    "robot_radius": 0.2,
    "goal": [
      0,
      1.8
    ],
    "goal_radius": 0.5,
    "audit_world": [
      -4,
      4,
      -4,
      4
    ]
  },
  "reset": {
    "center": [
      -0.9,
      -3,
      0.7
    ],
    "independent_uniform_half_width": 0.02
  },
  "certificate": {
    "B": "hypot(x,y)-0.7, exact discounted safety value on B>=0 for reversible nominal model",
    "Cx": 1.0,
    "L": "hypot(gamma+0.6/(r-0.6*Delta),0.6)",
    "epsilon": "L*(Cx+1)*Delta",
    "activation": "B<=1; no release hysteresis",
    "qp": "min ||u-u_nom||^2, u in [-1,1]^2; Psi_model>=epsilon+(xi if CP else 0)",
    "true_dynamics_in_uncal_controller": false,
    "integration": "exact constant-control unicycle flow and exact arc radial minimum"
  },
  "qualification_directories": [
    "/home/mars/Desktop/conformal shield/results_mechanism_v6/queue_design_001/hard_seed940001",
    "/home/mars/Desktop/conformal shield/results_mechanism_v6/queue_design_001/hard_seed940002",
    "/home/mars/Desktop/conformal shield/results_mechanism_v6/queue_design_001/aligned_seed940003",
    "/home/mars/Desktop/conformal shield/results_mechanism_v6/queue_design_001/easy_seed940004"
  ],
  "entry": {
    "id": 1,
    "n_calibration": 199,
    "delta": 0.01,
    "qualification_seed_base": 940001,
    "final_training_seeds": [
      111000001,
      112000001,
      113000001,
      114000001,
      115000001
    ]
  },
  "training_runs": 5,
  "evaluation_runs": 45,
  "episodes_per_evaluation": 100,
  "heldout_reference_rollouts_per_condition_seed": 100,
  "heldout_deployment_rollouts_per_condition_seed": 100,
  "representative": "first rollout only, all intervals",
  "standard_deviation": "sample standard deviation across five training seeds, ddof=1",
  "limitations": [
    "Finite-grid feasibility plus strict runtime checks is not a global continuous-state feasibility theorem.",
    "Split-conformal coverage refers to uncalibrated-filter reference rollouts; deployment coverage is separately empirical.",
    "Learned policy is a bounded residual on a specified shared reference; this is not a learning-efficiency comparison."
  ]
}
```

## Main comparison

Means ± sample SD across five training seeds (ddof=1); 500 pooled episodes per condition/method. Intervention rate is the mean of each episode’s intervention-count/step-count ratio; pooled step-weighted rates and per-seed minimum-clearance mean/SD are also in the CSV. Collision uses the exact minimum distance on each continuous held-control arc.

| Condition | Method | Unsafe % | Unsafe count | Goal % | Return | Intervention % | Interventions/episode | Min clearance (m) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Aligned | Nominal RL | 0.0 ± 0.0 | 0/500 | 100.0 ± 0.0 | 78.880 ± 0.046 | 0.00 ± 0.00 | 0.0 ± 0.0 | 0.0840435834 |
| Aligned | Uncalibrated CBVF | 0.0 ± 0.0 | 0/500 | 100.0 ± 0.0 | 78.878 ± 0.049 | 0.50 ± 0.94 | 19.7 ± 37.3 | 0.0840985785 |
| Aligned | Conformal-CBVF | 0.0 ± 0.0 | 0/500 | 100.0 ± 0.0 | 78.878 ± 0.049 | 0.50 ± 0.94 | 19.7 ± 37.3 | 0.0840985785 |
| Easy Mismatch | Nominal RL | 0.0 ± 0.0 | 0/500 | 100.0 ± 0.0 | 76.972 ± 0.194 | 0.00 ± 0.00 | 0.0 ± 0.0 | 0.0363785904 |
| Easy Mismatch | Uncalibrated CBVF | 0.0 ± 0.0 | 0/500 | 100.0 ± 0.0 | 76.159 ± 0.339 | 8.71 ± 1.43 | 470.9 ± 81.0 | 0.0382352126 |
| Easy Mismatch | Conformal-CBVF | 0.0 ± 0.0 | 0/500 | 100.0 ± 0.0 | 75.576 ± 0.496 | 11.15 ± 1.62 | 611.4 ± 96.1 | 0.039080594 |
| Hard Mismatch | Nominal RL | 100.0 ± 0.0 | 500/500 | 0.0 ± 0.0 | -116.376 ± 0.050 | 0.00 ± 0.00 | 0.0 ± 0.0 | -0.000597590535 |
| Hard Mismatch | Uncalibrated CBVF | 100.0 ± 0.0 | 500/500 | 0.0 ± 0.0 | -121.724 ± 0.252 | 45.43 ± 0.91 | 1542.2 ± 58.4 | -1.3656755e-05 |
| Hard Mismatch | Conformal-CBVF | 0.0 ± 0.0 | 0/500 | 100.0 ± 0.0 | 42.896 ± 0.292 | 20.72 ± 0.09 | 2188.0 ± 5.8 | 0.0258294824 |


## Hard result

Hard unsafe counts are 500/500 Nominal, 500/500 uncalibrated CBVF, and 0/500 conformal-CBVF; goal counts are 0/500, 0/500, and 500/500 respectively.

## Held-out coverage and buffers

The formal split-conformal reference population is the same frozen actor behind the uncalibrated filter, with regional rollout maxima. Calibration, primary evaluation, and held-out initial states are disjoint. Deployment coverage is an additional empirical diagnostic, not a transfer of the reference-population theorem. Empirical acceptance requires at least 95/100 covered in every block.

| condition | seed | evaluation_seed | covered | total | deployment_covered | target | empirical_acceptance_threshold | n_calibration | xi_near | xi_far |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| aligned | 111000001 | 111200001 | 100 | 100 | 100 | 0.99 | 0.95 | 199 | 0.0 | 0.0 |
| easy | 111000001 | 111300001 | 100 | 100 | 100 | 0.99 | 0.95 | 199 | 0.022136026189534024 | 0.025054010051844645 |
| hard | 111000001 | 111400001 | 100 | 100 | 100 | 0.99 | 0.95 | 199 | 0.04779315809074445 | 0.05011064674802458 |
| aligned | 112000001 | 112200001 | 100 | 100 | 100 | 0.99 | 0.95 | 199 | 0.0 | 0.0 |
| easy | 112000001 | 112300001 | 98 | 100 | 98 | 0.99 | 0.95 | 199 | 0.02220797037573035 | 0.025053969240803883 |
| hard | 112000001 | 112400001 | 100 | 100 | 100 | 0.99 | 0.95 | 199 | 0.047807839803426226 | 0.05011055471759527 |
| aligned | 113000001 | 113200001 | 100 | 100 | 100 | 0.99 | 0.95 | 199 | 0.0 | 0.0 |
| easy | 113000001 | 113300001 | 96 | 100 | 96 | 0.99 | 0.95 | 199 | 0.022105322769173017 | 0.02505395570116774 |
| hard | 113000001 | 113400001 | 99 | 100 | 99 | 0.99 | 0.95 | 199 | 0.047697404496918484 | 0.050110488254822144 |
| aligned | 114000001 | 114200001 | 100 | 100 | 100 | 0.99 | 0.95 | 199 | 0.0 | 0.0 |
| easy | 114000001 | 114300001 | 99 | 100 | 99 | 0.99 | 0.95 | 199 | 0.021974212847306496 | 0.025053962950575607 |
| hard | 114000001 | 114400001 | 99 | 100 | 99 | 0.99 | 0.95 | 199 | 0.04755682968617269 | 0.05011044015772685 |
| aligned | 115000001 | 115200001 | 100 | 100 | 100 | 0.99 | 0.95 | 199 | 0.0 | 0.0 |
| easy | 115000001 | 115300001 | 99 | 100 | 99 | 0.99 | 0.95 | 199 | 0.02194046661012052 | 0.02505395054547751 |
| hard | 115000001 | 115400001 | 100 | 100 | 100 | 0.99 | 0.95 | 199 | 0.047547296762320355 | 0.050110418849512055 |


| condition | region | mean | std |
| --- | --- | --- | --- |
| aligned | h<0.75 | 0.0 | 0.0 |
| aligned | h>=0.75 | 0.0 | 0.0 |
| easy | h<0.75 | 0.02207279975837288 | 0.00011242538057048419 |
| easy | h>=0.75 | 0.025053969697973875 | 2.3647938456095268e-08 |
| hard | h<0.75 | 0.047680505767916444 | 0.00012473019065869977 |
| hard | h>=0.75 | 0.05011050974553618 | 9.268032305407944e-08 |


## Feasibility and continuous intervals

30 exhaustive method-specific grid audits checked 81000 unique-node evaluations and 968226000 node/interval combinations, with zero infeasible combinations. These finite-grid checks do not prove feasibility at every continuous state. All executed filtered actions also undergo strict checks; no fallback is executed.

| condition | seed | method | unique_nodes | node_interval_checks | infeasible_nodes | infeasible_node_intervals | minimum_margin |
| --- | --- | --- | --- | --- | --- | --- | --- |
| aligned | 111000001 | cbvf | 2700 | 32274200 | 0 | 0 | 0.07815763928849538 |
| aligned | 111000001 | cp | 2700 | 32274200 | 0 | 0 | 0.07815763928849538 |
| easy | 111000001 | cbvf | 2700 | 32274200 | 0 | 0 | 0.07815763928849538 |
| easy | 111000001 | cp | 2700 | 32274200 | 0 | 0 | 0.05602161309896136 |
| hard | 111000001 | cbvf | 2700 | 32274200 | 0 | 0 | 0.07815763928849538 |
| hard | 111000001 | cp | 2700 | 32274200 | 0 | 0 | 0.030364481197750932 |
| aligned | 112000001 | cbvf | 2700 | 32274200 | 0 | 0 | 0.07815763928849538 |
| aligned | 112000001 | cp | 2700 | 32274200 | 0 | 0 | 0.07815763928849538 |
| easy | 112000001 | cbvf | 2700 | 32274200 | 0 | 0 | 0.07815763928849538 |
| easy | 112000001 | cp | 2700 | 32274200 | 0 | 0 | 0.05594966891276503 |
| hard | 112000001 | cbvf | 2700 | 32274200 | 0 | 0 | 0.07815763928849538 |
| hard | 112000001 | cp | 2700 | 32274200 | 0 | 0 | 0.030349799485069155 |
| aligned | 113000001 | cbvf | 2700 | 32274200 | 0 | 0 | 0.07815763928849538 |
| aligned | 113000001 | cp | 2700 | 32274200 | 0 | 0 | 0.07815763928849538 |
| easy | 113000001 | cbvf | 2700 | 32274200 | 0 | 0 | 0.07815763928849538 |
| easy | 113000001 | cp | 2700 | 32274200 | 0 | 0 | 0.05605231651932237 |
| hard | 113000001 | cbvf | 2700 | 32274200 | 0 | 0 | 0.07815763928849538 |
| hard | 113000001 | cp | 2700 | 32274200 | 0 | 0 | 0.030460234791576897 |
| aligned | 114000001 | cbvf | 2700 | 32274200 | 0 | 0 | 0.07815763928849538 |
| aligned | 114000001 | cp | 2700 | 32274200 | 0 | 0 | 0.07815763928849538 |
| easy | 114000001 | cbvf | 2700 | 32274200 | 0 | 0 | 0.07815763928849538 |
| easy | 114000001 | cp | 2700 | 32274200 | 0 | 0 | 0.056183426441188886 |
| hard | 114000001 | cbvf | 2700 | 32274200 | 0 | 0 | 0.07815763928849538 |
| hard | 114000001 | cp | 2700 | 32274200 | 0 | 0 | 0.03060080960232269 |
| aligned | 115000001 | cbvf | 2700 | 32274200 | 0 | 0 | 0.07815763928849538 |
| aligned | 115000001 | cp | 2700 | 32274200 | 0 | 0 | 0.07815763928849538 |
| easy | 115000001 | cbvf | 2700 | 32274200 | 0 | 0 | 0.07815763928849538 |
| easy | 115000001 | cp | 2700 | 32274200 | 0 | 0 | 0.056217172678374866 |
| hard | 115000001 | cbvf | 2700 | 32274200 | 0 | 0 | 0.07815763928849538 |
| hard | 115000001 | cp | 2700 | 32274200 | 0 | 0 | 0.030610342526175026 |


B=h is analytically exact on the safe domain for the reversible model: the time-zero value is an upper bound and zero speed attains it. The model and true held-control derivative fields have the reported analytic Lipschitz bound under the public speed/turn envelope. The common epsilon is L(C_x+1)Delta. Inactive intervals satisfy the direct clearance reachability bound.

| condition | Cx | Delta | L_min | L_max | epsilon_min | epsilon_max | representative_intervals | reconstructed_subinterval_checks | failures |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| aligned | 1.0 | 0.002 | 1.9479026063075147 | 2.335656917898536 | 0.007791610425230059 | 0.009342627671594144 | 118527 | 474108 | 0 |
| easy | 1.0 | 0.002 | 1.947919746432245 | 2.3874232269586724 | 0.007791678985728981 | 0.00954969290783469 | 162595 | 650380 | 0 |
| hard | 1.0 | 0.002 | 1.947902826148154 | 2.4337326469857263 | 0.007791611304592615 | 0.009734930587942905 | 167918 | 671672 | 0 |


90 preselected representative traces reconstructed 449040 complete intervals and 1796160 interior/end checks; all passed. True derivative violations in Hard uncalibrated trajectories are expected mechanism evidence, while the model QP remains feasible. CP true inter-sample lower bounds pass.

## Figures

- `/home/mars/Desktop/conformal shield/results_mechanism_v6/study_001/figures/safety_goal_comparison.pdf`: MAIN safety comparison: all three methods and all three conditions; mean and sample SD across five paired actors, 500 episodes per bar. PNG counterpart is in the same directory.
- `/home/mars/Desktop/conformal shield/results_mechanism_v6/study_001/figures/matched_trajectories.pdf`: First predeclared evaluation episode of the first final actor; all conditions and methods. Circle includes robot inflation. PNG counterpart is in the same directory.
- `/home/mars/Desktop/conformal shield/results_mechanism_v6/study_001/figures/calibration_buffers_coverage.pdf`: Regional buffer mean and SD plus all 15 fresh held-out reference coverage counts; Aligned buffers are zero. PNG counterpart is in the same directory.
- `/home/mars/Desktop/conformal shield/results_mechanism_v6/study_001/figures/shared_actor_training.pdf`: Logged nominal-model training returns for five shared residual actors; descriptive training trace, not evidence of learning convergence. PNG counterpart is in the same directory.
- `/home/mars/Desktop/conformal shield/results_mechanism_v6/study_001/figures/feasibility_intersample_verification.pdf`: Hard CP first predeclared trajectory: safety clearance, derivative tightening, positive available QP margin and analytic true inter-sample lower bound. PNG counterpart is in the same directory.
- `/home/mars/Desktop/conformal shield/results_mechanism_v6/study_001/figures/uncalibrated_mismatch_mechanism.pdf`: Hard uncalibrated first predeclared trajectory: the model constraint remains satisfied while the true derivative condition becomes negative before collision. PNG counterpart is in the same directory.

## Acceptance

| Criterion | Result |
| --- | --- |
| frozen_hashes | PASS |
| final_45_runs | PASS |
| five_fresh_shared_actors | PASS |
| all_condition_seed_acceptance | PASS |
| zero_exhaustive_infeasibility | PASS |
| strictly_positive_feasibility | PASS |
| heldout_coverage | PASS |
| independent_interval_reconstruction | PASS |
| hard_uncal_separation | PASS |
| goal_success | PASS |
| nontrivial_mismatch_buffers | PASS |
| zero_aligned_buffers | PASS |
| paper_figures | PASS |


## Scope

- Finite-grid feasibility plus strict runtime checks is not a global continuous-state feasibility theorem.
- Split-conformal coverage refers to uncalibrated-filter reference rollouts; deployment coverage is separately empirical.
- Learned policy is a bounded residual on a specified shared reference; this is not a learning-efficiency comparison.
- Zero observed collisions is an empirical outcome, not proof of zero failure probability under arbitrary deployment distributions.

## Executed-node feasibility and interval checks

| condition | method | rollouts | active_node_checks | minimum_available_margin | minimum_achieved_constraint_residual | minimum_true_interval_bound |
| --- | --- | --- | --- | --- | --- | --- |
| aligned | cbvf | 1995 | 5386943 | 0.11586930343509558 | -1.9081958235744878e-17 | 0.004483446961825093 |
| aligned | cp | 1000 | 2700247 | 0.11682235754211506 | -1.9081958235744878e-17 | 0.004485468249737477 |
| easy | cbvf | 1995 | 8285303 | 0.04696905328188494 | -4.163336342344337e-17 | 0.001593948615363429 |
| easy | cp | 1000 | 4229347 | 0.026039346825785466 | -4.163336342344337e-17 | 0.023134486719441332 |
| hard | cbvf | 1995 | 4278597 | 0.045459220960471475 | -4.163336342344337e-17 | -0.01317508337299181 |
| hard | cp | 1000 | 4795322 | 0.05300512211166384 | -4.163336342344337e-17 | 0.03258568012840559 |

The achieved QP residual may be zero at a binding projection; residuals of order 1e-16 are floating-point roundoff. Available control margins are strictly positive. Negative true derivative bounds for Hard uncalibrated CBVF are the model-mismatch mechanism, not a model-QP infeasibility.

## FILES TO OPEN FOR WRITING THE CASE STUDY

- `/home/mars/Desktop/conformal shield/results_mechanism_v6/study_001/PAPER_HANDOFF.md`
- `/home/mars/Desktop/conformal shield/results_mechanism_v6/study_001/effective_configuration.json`
- `/home/mars/Desktop/conformal shield/results_mechanism_v6/study_001/figures/safety_goal_comparison.pdf`
- `/home/mars/Desktop/conformal shield/results_mechanism_v6/study_001/figures/matched_trajectories.pdf`
- `/home/mars/Desktop/conformal shield/results_mechanism_v6/study_001/figures/calibration_buffers_coverage.pdf`
- `/home/mars/Desktop/conformal shield/results_mechanism_v6/study_001/figures/shared_actor_training.pdf`
- `/home/mars/Desktop/conformal shield/results_mechanism_v6/study_001/figures/feasibility_intersample_verification.pdf`
- `/home/mars/Desktop/conformal shield/results_mechanism_v6/study_001/figures/uncalibrated_mismatch_mechanism.pdf`
- `/home/mars/Desktop/conformal shield/results_mechanism_v6/study_001/tables/buffer_summary.csv`
- `/home/mars/Desktop/conformal shield/results_mechanism_v6/study_001/tables/coverage_and_buffers.csv`
- `/home/mars/Desktop/conformal shield/results_mechanism_v6/study_001/tables/feasibility.csv`
- `/home/mars/Desktop/conformal shield/results_mechanism_v6/study_001/tables/intersample_summary.csv`
- `/home/mars/Desktop/conformal shield/results_mechanism_v6/study_001/tables/main_comparison.csv`
- `/home/mars/Desktop/conformal shield/results_mechanism_v6/study_001/tables/main_comparison.md`
- `/home/mars/Desktop/conformal shield/results_mechanism_v6/study_001/tables/main_comparison.tex`
- `/home/mars/Desktop/conformal shield/results_mechanism_v6/study_001/tables/representative_intervals.csv`
- `/home/mars/Desktop/conformal shield/results_mechanism_v6/study_001/aggregate/episodes.csv`
- `/home/mars/Desktop/conformal shield/results_mechanism_v6/study_001/aggregate/hard_same_state_mechanism.csv`
- `/home/mars/Desktop/conformal shield/results_mechanism_v6/study_001/aggregate/per_seed.csv`
- `/home/mars/Desktop/conformal shield/results_mechanism_v6/study_001/aggregate/summary.csv`
- `/home/mars/Desktop/conformal shield/results_mechanism_v6/study_001/tables/runtime_feasibility.csv`
