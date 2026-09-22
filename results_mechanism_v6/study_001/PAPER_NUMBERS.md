# Accepted study: paper numbers

/home/mars/Desktop/conformal shield/results_mechanism_v6/study_001

## Frozen case-study parameters

With normalized controls (a,b) in [-1,1]^2, p_dot=v(a)(cos(theta),sin(theta)):

- Aligned/model: v=0.2+0.4a; theta_dot=0.8b.
- Easy: v=0.2125+0.3875a; theta_dot=0.625b.
- Hard: v=0.225+0.375a; theta_dot=0.43b.
- Reset: independent uniforms centered at (-0.9,-3,0.7), half-width 0.02 in x, y, and heading.
- Obstacle: origin, radius 0.5 m; robot radius 0.2 m; inflated radius 0.7 m. Goal: (0,1.8), radius 0.5 m. Audit box: [-4,4]^2.
- Exact analytic CBVF B=h=hypot(x,y)-0.7 on the safe domain; gamma=1.5. Delta=0.002 s; horizon=25 s (12,500 intervals); activation B<=1 for both filters; no release hysteresis.
- Public bounds: |v|<=0.6, |omega|<=0.8, C_x=1. Exact held-control flow and exact minimum clearance over every continuous arc.
- Reference before 8 s: waypoint (-0.9,0.65) until y>=0.45, then goal. From 8 s: waypoint (-1.4,-1.2) while x>-1.2 and y<0.45; otherwise (-1.4,0.65) while y<0.45, then goal. Throttle is -1 during [8,10) s and +1 otherwise; yaw=clip(1.5*wrapped_heading_error/0.8,-1,1). No orbit branch.
- Nominal actor: this shared reference plus 0.02 times a deterministic frozen learned residual, with actuator clipping before either QP. The QP constraint itself is never clipped.
- Five independently initialized SAC-Lag residual actors trained for 20,000 steps on Aligned/no-filter dynamics; hidden layers (256,256), ReLU; tanh mean at evaluation. Warmup/reference proposals 1,000 steps; batch 256; update ratio 1; all learning rates 0.0003; discount 0.99; tau 0.005; initial temperature 0.1; cost limit 1. Full defaults are in effective_configuration.json.
- Reward per step: -Delta*distance_to_goal(next state)+100*goal-100*collision; exact-arc collision cost=1; goal/collision terminate.
- Calibration n=199, delta=0.01, two regions h<0.75 and h>=0.75 with delta/2 each. Quantile order ceil((n+1)(1-delta/2)).
- Final actor seeds: 111000001, 112000001, 113000001, 114000001, 115000001. Each is reused across all three conditions and all three filters.
- 45 evaluations x 100 episodes = 4,500 primary episodes. Fresh reference coverage: 15 x 100 rollouts, plus a separate CP deployment diagnostic on those held-out starts.
- Calibration seed = actor seed + 200000 + condition_index*100000 (Aligned=0, Easy=1, Hard=2); primary starts +10000; coverage starts +20000. Training reset RNG = actor seed +100000. No final seeds were used in design.

## Main numerical comparison

Mean ± sample SD across five actor seeds; unsafe/goal counts pooled over 500 episodes. Intervention rate is episode-averaged.

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


### Minimum physical safety margins

| Condition | Method | Per-seed minimum mean | Per-seed minimum SD | Pooled minimum |
| --- | --- | --- | --- | --- |
| aligned | nominal | 0.09532618391134055 | 0.009288250008664118 | 0.08404358343542129 |
| aligned | cbvf | 0.09533931797110962 | 0.009270175349347069 | 0.08409857846870694 |
| aligned | cp | 0.09533931797110962 | 0.009270175349347069 | 0.08409857846870694 |
| easy | nominal | 0.04752348073311998 | 0.009117316855168332 | 0.03637859035115354 |
| easy | cbvf | 0.048671390279008755 | 0.008598093183087106 | 0.03823521262560581 |
| easy | cp | 0.04926356731780344 | 0.008413702805189037 | 0.039080593961213994 |
| hard | nominal | -0.0005668984971028523 | 2.2447567823700647e-05 | -0.0005975905353250432 |
| hard | cbvf | -1.1919329021892189e-05 | 1.5537124538627014e-06 | -1.3656755025071021e-05 |
| hard | cp | 0.026224166439152374 | 0.00033710777474232055 | 0.025829482359072897 |

## Feasibility

30 exhaustive audits; 968226000 node/interval combinations; zero calibrated and uncalibrated infeasible combinations. There are 2,700 grid nodes per method-specific audit; the stationary margin is reused across its exact eligible interval multiplicity. These are finite-grid and executed-trajectory checks, not a continuous-state global feasibility theorem.

| Condition | Audits | Node_interval_checks | Uncalibrated_min_margin | CP_min_margin | Infeasible |
| --- | --- | --- | --- | --- | --- |
| aligned | 10 | 322742000 | 0.07815763928849538 | 0.07815763928849538 | 0 |
| easy | 10 | 322742000 | 0.07815763928849538 | 0.05594966891276503 | 0 |
| hard | 10 | 322742000 | 0.07815763928849538 | 0.030349799485069155 | 0 |

## All 15 held-out coverage results and regional buffers

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

Coverage range: 96/100 to 100/100; formal target 0.99; empirical acceptance threshold 95/100. Aligned zero buffers are exact. Reference-rollout coverage is the conformal population; deployment coverage is reported separately.

| condition | region | mean | std |
| --- | --- | --- | --- |
| aligned | h<0.75 | 0.0 | 0.0 |
| aligned | h>=0.75 | 0.0 | 0.0 |
| easy | h<0.75 | 0.02207279975837288 | 0.00011242538057048419 |
| easy | h>=0.75 | 0.025053969697973875 | 2.3647938456095268e-08 |
| hard | h<0.75 | 0.047680505767916444 | 0.00012473019065869977 |
| hard | h>=0.75 | 0.05011050974553618 | 9.268032305407944e-08 |

## Inter-sample quantities

L_Psi=hypot(gamma+0.6/(r-0.6*Delta),0.6); epsilon=L_Psi(C_x+1)Delta=0.004 L_Psi. The ranges below cover the active intervals of all primary representative trajectories.

| condition | Cx | Delta | L_min | L_max | epsilon_min | epsilon_max | representative_intervals | reconstructed_subinterval_checks | failures |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| aligned | 1.0 | 0.002 | 1.9479026063075147 | 2.335656917898536 | 0.007791610425230059 | 0.009342627671594144 | 118527 | 474108 | 0 |
| easy | 1.0 | 0.002 | 1.947919746432245 | 2.3874232269586724 | 0.007791678985728981 | 0.00954969290783469 | 162595 | 650380 | 0 |
| hard | 1.0 | 0.002 | 1.947902826148154 | 2.4337326469857263 | 0.007791611304592615 | 0.009734930587942905 | 167918 | 671672 | 0 |

90 predeclared representative traces; 449040 reconstructed intervals; 1796160 interior/end checks; all passed.

## Paper-ready sentences

1. We evaluated five frozen residual actors under three dynamics conditions and three filters, using 100 paired episodes per condition–actor block (4,500 primary episodes).
2. Under Hard Mismatch, Nominal and uncalibrated CBVF had 500/500 and 500/500 unsafe episodes, respectively, whereas conformal-CBVF had 0/500.
3. Hard-Mismatch goal counts were 0/500, 0/500, and 500/500 for Nominal, uncalibrated CBVF, and conformal-CBVF, respectively.
4. Both filters used identical actors, activation, gamma, actuator bounds, and inter-sample corrections; the calibrated regional mismatch buffer was the only difference in their QP constraints.
5. All 30 exhaustive method-specific grid audits passed with zero infeasible node/interval combinations, and all 449040 reconstructed representative intervals passed verification.
6. Fresh held-out reference coverage ranged from 96/100 to 100/100 across the 15 condition–actor blocks; deployment coverage was assessed separately, without claiming automatic transfer of the reference-population guarantee.

## Final figures

- `/home/mars/Desktop/conformal shield/results_mechanism_v6/study_001/figures/safety_goal_comparison.pdf`: MAIN safety comparison: all three methods and all three conditions; mean and sample SD across five paired actors, 500 episodes per bar.
- `/home/mars/Desktop/conformal shield/results_mechanism_v6/study_001/figures/matched_trajectories.pdf`: First predeclared evaluation episode of the first final actor; all conditions and methods. Circle includes robot inflation.
- `/home/mars/Desktop/conformal shield/results_mechanism_v6/study_001/figures/calibration_buffers_coverage.pdf`: Regional buffer mean and SD plus all 15 fresh held-out reference coverage counts; Aligned buffers are zero.
- `/home/mars/Desktop/conformal shield/results_mechanism_v6/study_001/figures/shared_actor_training.pdf`: Logged nominal-model training returns for five shared residual actors; descriptive training trace, not evidence of learning convergence.
- `/home/mars/Desktop/conformal shield/results_mechanism_v6/study_001/figures/feasibility_intersample_verification.pdf`: Hard CP first predeclared trajectory: safety clearance, derivative tightening, positive available QP margin and analytic true inter-sample lower bound.
- `/home/mars/Desktop/conformal shield/results_mechanism_v6/study_001/figures/uncalibrated_mismatch_mechanism.pdf`: Hard uncalibrated first predeclared trajectory: the model constraint remains satisfied while the true derivative condition becomes negative before collision.

## Final tables

- `/home/mars/Desktop/conformal shield/results_mechanism_v6/study_001/tables/buffer_summary.csv`
- `/home/mars/Desktop/conformal shield/results_mechanism_v6/study_001/tables/coverage_and_buffers.csv`
- `/home/mars/Desktop/conformal shield/results_mechanism_v6/study_001/tables/feasibility.csv`
- `/home/mars/Desktop/conformal shield/results_mechanism_v6/study_001/tables/intersample_summary.csv`
- `/home/mars/Desktop/conformal shield/results_mechanism_v6/study_001/tables/main_comparison.csv`
- `/home/mars/Desktop/conformal shield/results_mechanism_v6/study_001/tables/main_comparison.md`
- `/home/mars/Desktop/conformal shield/results_mechanism_v6/study_001/tables/main_comparison.tex`
- `/home/mars/Desktop/conformal shield/results_mechanism_v6/study_001/tables/representative_intervals.csv`

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

- `/home/mars/Desktop/conformal shield/results_mechanism_v6/study_001/PAPER_NUMBERS.md`
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
- `/home/mars/Desktop/conformal shield/results_mechanism_v6/study_001/aggregate/hard_same_state_mechanism.csv`
- `/home/mars/Desktop/conformal shield/results_mechanism_v6/study_001/tables/runtime_feasibility.csv`
