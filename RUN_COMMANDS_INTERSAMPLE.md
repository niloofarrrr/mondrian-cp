# Inter-sample experiment commands

```bash
cd ~/Desktop/"conformal shield"
```

## Isolated resumable wrapper

```bash
bash experiments/launch_full_suite_intersample.sh --retry-failed
```

The strict feasibility gate currently stops conformal runs before training; these are the literal configured commands and must not be run with an infeasibility override.

## aligned_nominal_seed_7

```bash
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
  --integration-substeps 10 \
  --intersample-protection \
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
  --speed-min -0.2 \
  --speed 0.6 \
  --beta-u 0.8 \
  --cbvf-model-speed-min -0.2 \
  --cbvf-model-speed 0.6 \
  --cbvf-model-beta-u 0.8 \
  --training-shield-method nominal \
  --seed 7 \
  --checkpoint-dir '/home/mars/Desktop/conformal shield/results_intersample/runs/aligned_nominal_seed_7' \
  --dense-safety-cost \
  --no-tqdm \
  --recompute-cbvf \
  --terminate-on-collision \
  --cache-dir '/home/mars/Desktop/conformal shield/cbvf_cache_intersample' \
  --odp-root '/home/mars/Desktop/conformal shield' \
  --solver-file '/home/mars/Desktop/conformal shield/solver_cbvf.py' \
  --eval-jsonl '/home/mars/Desktop/conformal shield/results_intersample/runs/aligned_nominal_seed_7/training_eval_metrics.jsonl' \
  --action-log-jsonl '/home/mars/Desktop/conformal shield/results_intersample/runs/aligned_nominal_seed_7/training_actions.jsonl' \
  --qp-diagnostics-jsonl '/home/mars/Desktop/conformal shield/results_intersample/runs/aligned_nominal_seed_7/qp_feasibility_diagnostics.jsonl' \
  --results-json '/home/mars/Desktop/conformal shield/results_intersample/runs/aligned_nominal_seed_7/training_and_evaluation_results.json' \
  --plot-dir '/home/mars/Desktop/conformal shield/results_intersample/runs/aligned_nominal_seed_7/plots'
```

## aligned_nominal_seed_42

```bash
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
  --integration-substeps 10 \
  --intersample-protection \
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
  --speed-min -0.2 \
  --speed 0.6 \
  --beta-u 0.8 \
  --cbvf-model-speed-min -0.2 \
  --cbvf-model-speed 0.6 \
  --cbvf-model-beta-u 0.8 \
  --training-shield-method nominal \
  --seed 42 \
  --checkpoint-dir '/home/mars/Desktop/conformal shield/results_intersample/runs/aligned_nominal_seed_42' \
  --dense-safety-cost \
  --no-tqdm \
  --recompute-cbvf \
  --terminate-on-collision \
  --cache-dir '/home/mars/Desktop/conformal shield/cbvf_cache_intersample' \
  --odp-root '/home/mars/Desktop/conformal shield' \
  --solver-file '/home/mars/Desktop/conformal shield/solver_cbvf.py' \
  --eval-jsonl '/home/mars/Desktop/conformal shield/results_intersample/runs/aligned_nominal_seed_42/training_eval_metrics.jsonl' \
  --action-log-jsonl '/home/mars/Desktop/conformal shield/results_intersample/runs/aligned_nominal_seed_42/training_actions.jsonl' \
  --qp-diagnostics-jsonl '/home/mars/Desktop/conformal shield/results_intersample/runs/aligned_nominal_seed_42/qp_feasibility_diagnostics.jsonl' \
  --results-json '/home/mars/Desktop/conformal shield/results_intersample/runs/aligned_nominal_seed_42/training_and_evaluation_results.json' \
  --plot-dir '/home/mars/Desktop/conformal shield/results_intersample/runs/aligned_nominal_seed_42/plots'
```

## aligned_nominal_seed_123

```bash
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
  --integration-substeps 10 \
  --intersample-protection \
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
  --speed-min -0.2 \
  --speed 0.6 \
  --beta-u 0.8 \
  --cbvf-model-speed-min -0.2 \
  --cbvf-model-speed 0.6 \
  --cbvf-model-beta-u 0.8 \
  --training-shield-method nominal \
  --seed 123 \
  --checkpoint-dir '/home/mars/Desktop/conformal shield/results_intersample/runs/aligned_nominal_seed_123' \
  --dense-safety-cost \
  --no-tqdm \
  --recompute-cbvf \
  --terminate-on-collision \
  --cache-dir '/home/mars/Desktop/conformal shield/cbvf_cache_intersample' \
  --odp-root '/home/mars/Desktop/conformal shield' \
  --solver-file '/home/mars/Desktop/conformal shield/solver_cbvf.py' \
  --eval-jsonl '/home/mars/Desktop/conformal shield/results_intersample/runs/aligned_nominal_seed_123/training_eval_metrics.jsonl' \
  --action-log-jsonl '/home/mars/Desktop/conformal shield/results_intersample/runs/aligned_nominal_seed_123/training_actions.jsonl' \
  --qp-diagnostics-jsonl '/home/mars/Desktop/conformal shield/results_intersample/runs/aligned_nominal_seed_123/qp_feasibility_diagnostics.jsonl' \
  --results-json '/home/mars/Desktop/conformal shield/results_intersample/runs/aligned_nominal_seed_123/training_and_evaluation_results.json' \
  --plot-dir '/home/mars/Desktop/conformal shield/results_intersample/runs/aligned_nominal_seed_123/plots'
```

## aligned_nominal_seed_0

```bash
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
  --integration-substeps 10 \
  --intersample-protection \
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
  --speed-min -0.2 \
  --speed 0.6 \
  --beta-u 0.8 \
  --cbvf-model-speed-min -0.2 \
  --cbvf-model-speed 0.6 \
  --cbvf-model-beta-u 0.8 \
  --training-shield-method nominal \
  --seed 0 \
  --checkpoint-dir '/home/mars/Desktop/conformal shield/results_intersample/runs/aligned_nominal_seed_0' \
  --dense-safety-cost \
  --no-tqdm \
  --recompute-cbvf \
  --terminate-on-collision \
  --cache-dir '/home/mars/Desktop/conformal shield/cbvf_cache_intersample' \
  --odp-root '/home/mars/Desktop/conformal shield' \
  --solver-file '/home/mars/Desktop/conformal shield/solver_cbvf.py' \
  --eval-jsonl '/home/mars/Desktop/conformal shield/results_intersample/runs/aligned_nominal_seed_0/training_eval_metrics.jsonl' \
  --action-log-jsonl '/home/mars/Desktop/conformal shield/results_intersample/runs/aligned_nominal_seed_0/training_actions.jsonl' \
  --qp-diagnostics-jsonl '/home/mars/Desktop/conformal shield/results_intersample/runs/aligned_nominal_seed_0/qp_feasibility_diagnostics.jsonl' \
  --results-json '/home/mars/Desktop/conformal shield/results_intersample/runs/aligned_nominal_seed_0/training_and_evaluation_results.json' \
  --plot-dir '/home/mars/Desktop/conformal shield/results_intersample/runs/aligned_nominal_seed_0/plots'
```

## aligned_nominal_seed_99

```bash
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
  --integration-substeps 10 \
  --intersample-protection \
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
  --speed-min -0.2 \
  --speed 0.6 \
  --beta-u 0.8 \
  --cbvf-model-speed-min -0.2 \
  --cbvf-model-speed 0.6 \
  --cbvf-model-beta-u 0.8 \
  --training-shield-method nominal \
  --seed 99 \
  --checkpoint-dir '/home/mars/Desktop/conformal shield/results_intersample/runs/aligned_nominal_seed_99' \
  --dense-safety-cost \
  --no-tqdm \
  --recompute-cbvf \
  --terminate-on-collision \
  --cache-dir '/home/mars/Desktop/conformal shield/cbvf_cache_intersample' \
  --odp-root '/home/mars/Desktop/conformal shield' \
  --solver-file '/home/mars/Desktop/conformal shield/solver_cbvf.py' \
  --eval-jsonl '/home/mars/Desktop/conformal shield/results_intersample/runs/aligned_nominal_seed_99/training_eval_metrics.jsonl' \
  --action-log-jsonl '/home/mars/Desktop/conformal shield/results_intersample/runs/aligned_nominal_seed_99/training_actions.jsonl' \
  --qp-diagnostics-jsonl '/home/mars/Desktop/conformal shield/results_intersample/runs/aligned_nominal_seed_99/qp_feasibility_diagnostics.jsonl' \
  --results-json '/home/mars/Desktop/conformal shield/results_intersample/runs/aligned_nominal_seed_99/training_and_evaluation_results.json' \
  --plot-dir '/home/mars/Desktop/conformal shield/results_intersample/runs/aligned_nominal_seed_99/plots'
```

## aligned_cbvf_seed_7

```bash
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
  --integration-substeps 10 \
  --intersample-protection \
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
  --speed-min -0.2 \
  --speed 0.6 \
  --beta-u 0.8 \
  --cbvf-model-speed-min -0.2 \
  --cbvf-model-speed 0.6 \
  --cbvf-model-beta-u 0.8 \
  --training-shield-method cbvf \
  --seed 7 \
  --checkpoint-dir '/home/mars/Desktop/conformal shield/results_intersample/runs/aligned_cbvf_seed_7' \
  --dense-safety-cost \
  --no-tqdm \
  --recompute-cbvf \
  --terminate-on-collision \
  --cache-dir '/home/mars/Desktop/conformal shield/cbvf_cache_intersample' \
  --odp-root '/home/mars/Desktop/conformal shield' \
  --solver-file '/home/mars/Desktop/conformal shield/solver_cbvf.py' \
  --eval-jsonl '/home/mars/Desktop/conformal shield/results_intersample/runs/aligned_cbvf_seed_7/training_eval_metrics.jsonl' \
  --action-log-jsonl '/home/mars/Desktop/conformal shield/results_intersample/runs/aligned_cbvf_seed_7/training_actions.jsonl' \
  --qp-diagnostics-jsonl '/home/mars/Desktop/conformal shield/results_intersample/runs/aligned_cbvf_seed_7/qp_feasibility_diagnostics.jsonl' \
  --results-json '/home/mars/Desktop/conformal shield/results_intersample/runs/aligned_cbvf_seed_7/training_and_evaluation_results.json' \
  --plot-dir '/home/mars/Desktop/conformal shield/results_intersample/runs/aligned_cbvf_seed_7/plots'
```

## aligned_cbvf_seed_42

```bash
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
  --integration-substeps 10 \
  --intersample-protection \
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
  --speed-min -0.2 \
  --speed 0.6 \
  --beta-u 0.8 \
  --cbvf-model-speed-min -0.2 \
  --cbvf-model-speed 0.6 \
  --cbvf-model-beta-u 0.8 \
  --training-shield-method cbvf \
  --seed 42 \
  --checkpoint-dir '/home/mars/Desktop/conformal shield/results_intersample/runs/aligned_cbvf_seed_42' \
  --dense-safety-cost \
  --no-tqdm \
  --recompute-cbvf \
  --terminate-on-collision \
  --cache-dir '/home/mars/Desktop/conformal shield/cbvf_cache_intersample' \
  --odp-root '/home/mars/Desktop/conformal shield' \
  --solver-file '/home/mars/Desktop/conformal shield/solver_cbvf.py' \
  --eval-jsonl '/home/mars/Desktop/conformal shield/results_intersample/runs/aligned_cbvf_seed_42/training_eval_metrics.jsonl' \
  --action-log-jsonl '/home/mars/Desktop/conformal shield/results_intersample/runs/aligned_cbvf_seed_42/training_actions.jsonl' \
  --qp-diagnostics-jsonl '/home/mars/Desktop/conformal shield/results_intersample/runs/aligned_cbvf_seed_42/qp_feasibility_diagnostics.jsonl' \
  --results-json '/home/mars/Desktop/conformal shield/results_intersample/runs/aligned_cbvf_seed_42/training_and_evaluation_results.json' \
  --plot-dir '/home/mars/Desktop/conformal shield/results_intersample/runs/aligned_cbvf_seed_42/plots'
```

## aligned_cbvf_seed_123

```bash
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
  --integration-substeps 10 \
  --intersample-protection \
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
  --speed-min -0.2 \
  --speed 0.6 \
  --beta-u 0.8 \
  --cbvf-model-speed-min -0.2 \
  --cbvf-model-speed 0.6 \
  --cbvf-model-beta-u 0.8 \
  --training-shield-method cbvf \
  --seed 123 \
  --checkpoint-dir '/home/mars/Desktop/conformal shield/results_intersample/runs/aligned_cbvf_seed_123' \
  --dense-safety-cost \
  --no-tqdm \
  --recompute-cbvf \
  --terminate-on-collision \
  --cache-dir '/home/mars/Desktop/conformal shield/cbvf_cache_intersample' \
  --odp-root '/home/mars/Desktop/conformal shield' \
  --solver-file '/home/mars/Desktop/conformal shield/solver_cbvf.py' \
  --eval-jsonl '/home/mars/Desktop/conformal shield/results_intersample/runs/aligned_cbvf_seed_123/training_eval_metrics.jsonl' \
  --action-log-jsonl '/home/mars/Desktop/conformal shield/results_intersample/runs/aligned_cbvf_seed_123/training_actions.jsonl' \
  --qp-diagnostics-jsonl '/home/mars/Desktop/conformal shield/results_intersample/runs/aligned_cbvf_seed_123/qp_feasibility_diagnostics.jsonl' \
  --results-json '/home/mars/Desktop/conformal shield/results_intersample/runs/aligned_cbvf_seed_123/training_and_evaluation_results.json' \
  --plot-dir '/home/mars/Desktop/conformal shield/results_intersample/runs/aligned_cbvf_seed_123/plots'
```

## aligned_cbvf_seed_0

```bash
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
  --integration-substeps 10 \
  --intersample-protection \
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
  --speed-min -0.2 \
  --speed 0.6 \
  --beta-u 0.8 \
  --cbvf-model-speed-min -0.2 \
  --cbvf-model-speed 0.6 \
  --cbvf-model-beta-u 0.8 \
  --training-shield-method cbvf \
  --seed 0 \
  --checkpoint-dir '/home/mars/Desktop/conformal shield/results_intersample/runs/aligned_cbvf_seed_0' \
  --dense-safety-cost \
  --no-tqdm \
  --recompute-cbvf \
  --terminate-on-collision \
  --cache-dir '/home/mars/Desktop/conformal shield/cbvf_cache_intersample' \
  --odp-root '/home/mars/Desktop/conformal shield' \
  --solver-file '/home/mars/Desktop/conformal shield/solver_cbvf.py' \
  --eval-jsonl '/home/mars/Desktop/conformal shield/results_intersample/runs/aligned_cbvf_seed_0/training_eval_metrics.jsonl' \
  --action-log-jsonl '/home/mars/Desktop/conformal shield/results_intersample/runs/aligned_cbvf_seed_0/training_actions.jsonl' \
  --qp-diagnostics-jsonl '/home/mars/Desktop/conformal shield/results_intersample/runs/aligned_cbvf_seed_0/qp_feasibility_diagnostics.jsonl' \
  --results-json '/home/mars/Desktop/conformal shield/results_intersample/runs/aligned_cbvf_seed_0/training_and_evaluation_results.json' \
  --plot-dir '/home/mars/Desktop/conformal shield/results_intersample/runs/aligned_cbvf_seed_0/plots'
```

## aligned_cbvf_seed_99

```bash
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
  --integration-substeps 10 \
  --intersample-protection \
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
  --speed-min -0.2 \
  --speed 0.6 \
  --beta-u 0.8 \
  --cbvf-model-speed-min -0.2 \
  --cbvf-model-speed 0.6 \
  --cbvf-model-beta-u 0.8 \
  --training-shield-method cbvf \
  --seed 99 \
  --checkpoint-dir '/home/mars/Desktop/conformal shield/results_intersample/runs/aligned_cbvf_seed_99' \
  --dense-safety-cost \
  --no-tqdm \
  --recompute-cbvf \
  --terminate-on-collision \
  --cache-dir '/home/mars/Desktop/conformal shield/cbvf_cache_intersample' \
  --odp-root '/home/mars/Desktop/conformal shield' \
  --solver-file '/home/mars/Desktop/conformal shield/solver_cbvf.py' \
  --eval-jsonl '/home/mars/Desktop/conformal shield/results_intersample/runs/aligned_cbvf_seed_99/training_eval_metrics.jsonl' \
  --action-log-jsonl '/home/mars/Desktop/conformal shield/results_intersample/runs/aligned_cbvf_seed_99/training_actions.jsonl' \
  --qp-diagnostics-jsonl '/home/mars/Desktop/conformal shield/results_intersample/runs/aligned_cbvf_seed_99/qp_feasibility_diagnostics.jsonl' \
  --results-json '/home/mars/Desktop/conformal shield/results_intersample/runs/aligned_cbvf_seed_99/training_and_evaluation_results.json' \
  --plot-dir '/home/mars/Desktop/conformal shield/results_intersample/runs/aligned_cbvf_seed_99/plots'
```

## aligned_cp_seed_7

```bash
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
  --integration-substeps 10 \
  --intersample-protection \
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
  --speed-min -0.2 \
  --speed 0.6 \
  --beta-u 0.8 \
  --cbvf-model-speed-min -0.2 \
  --cbvf-model-speed 0.6 \
  --cbvf-model-beta-u 0.8 \
  --training-shield-method cp \
  --seed 7 \
  --checkpoint-dir '/home/mars/Desktop/conformal shield/results_intersample/runs/aligned_cp_seed_7' \
  --dense-safety-cost \
  --no-tqdm \
  --recompute-cbvf \
  --terminate-on-collision \
  --cache-dir '/home/mars/Desktop/conformal shield/cbvf_cache_intersample' \
  --odp-root '/home/mars/Desktop/conformal shield' \
  --solver-file '/home/mars/Desktop/conformal shield/solver_cbvf.py' \
  --eval-jsonl '/home/mars/Desktop/conformal shield/results_intersample/runs/aligned_cp_seed_7/training_eval_metrics.jsonl' \
  --action-log-jsonl '/home/mars/Desktop/conformal shield/results_intersample/runs/aligned_cp_seed_7/training_actions.jsonl' \
  --qp-diagnostics-jsonl '/home/mars/Desktop/conformal shield/results_intersample/runs/aligned_cp_seed_7/qp_feasibility_diagnostics.jsonl' \
  --results-json '/home/mars/Desktop/conformal shield/results_intersample/runs/aligned_cp_seed_7/training_and_evaluation_results.json' \
  --plot-dir '/home/mars/Desktop/conformal shield/results_intersample/runs/aligned_cp_seed_7/plots'
```

## aligned_cp_seed_42

```bash
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
  --integration-substeps 10 \
  --intersample-protection \
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
  --speed-min -0.2 \
  --speed 0.6 \
  --beta-u 0.8 \
  --cbvf-model-speed-min -0.2 \
  --cbvf-model-speed 0.6 \
  --cbvf-model-beta-u 0.8 \
  --training-shield-method cp \
  --seed 42 \
  --checkpoint-dir '/home/mars/Desktop/conformal shield/results_intersample/runs/aligned_cp_seed_42' \
  --dense-safety-cost \
  --no-tqdm \
  --recompute-cbvf \
  --terminate-on-collision \
  --cache-dir '/home/mars/Desktop/conformal shield/cbvf_cache_intersample' \
  --odp-root '/home/mars/Desktop/conformal shield' \
  --solver-file '/home/mars/Desktop/conformal shield/solver_cbvf.py' \
  --eval-jsonl '/home/mars/Desktop/conformal shield/results_intersample/runs/aligned_cp_seed_42/training_eval_metrics.jsonl' \
  --action-log-jsonl '/home/mars/Desktop/conformal shield/results_intersample/runs/aligned_cp_seed_42/training_actions.jsonl' \
  --qp-diagnostics-jsonl '/home/mars/Desktop/conformal shield/results_intersample/runs/aligned_cp_seed_42/qp_feasibility_diagnostics.jsonl' \
  --results-json '/home/mars/Desktop/conformal shield/results_intersample/runs/aligned_cp_seed_42/training_and_evaluation_results.json' \
  --plot-dir '/home/mars/Desktop/conformal shield/results_intersample/runs/aligned_cp_seed_42/plots'
```

## aligned_cp_seed_123

```bash
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
  --integration-substeps 10 \
  --intersample-protection \
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
  --speed-min -0.2 \
  --speed 0.6 \
  --beta-u 0.8 \
  --cbvf-model-speed-min -0.2 \
  --cbvf-model-speed 0.6 \
  --cbvf-model-beta-u 0.8 \
  --training-shield-method cp \
  --seed 123 \
  --checkpoint-dir '/home/mars/Desktop/conformal shield/results_intersample/runs/aligned_cp_seed_123' \
  --dense-safety-cost \
  --no-tqdm \
  --recompute-cbvf \
  --terminate-on-collision \
  --cache-dir '/home/mars/Desktop/conformal shield/cbvf_cache_intersample' \
  --odp-root '/home/mars/Desktop/conformal shield' \
  --solver-file '/home/mars/Desktop/conformal shield/solver_cbvf.py' \
  --eval-jsonl '/home/mars/Desktop/conformal shield/results_intersample/runs/aligned_cp_seed_123/training_eval_metrics.jsonl' \
  --action-log-jsonl '/home/mars/Desktop/conformal shield/results_intersample/runs/aligned_cp_seed_123/training_actions.jsonl' \
  --qp-diagnostics-jsonl '/home/mars/Desktop/conformal shield/results_intersample/runs/aligned_cp_seed_123/qp_feasibility_diagnostics.jsonl' \
  --results-json '/home/mars/Desktop/conformal shield/results_intersample/runs/aligned_cp_seed_123/training_and_evaluation_results.json' \
  --plot-dir '/home/mars/Desktop/conformal shield/results_intersample/runs/aligned_cp_seed_123/plots'
```

## aligned_cp_seed_0

```bash
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
  --integration-substeps 10 \
  --intersample-protection \
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
  --speed-min -0.2 \
  --speed 0.6 \
  --beta-u 0.8 \
  --cbvf-model-speed-min -0.2 \
  --cbvf-model-speed 0.6 \
  --cbvf-model-beta-u 0.8 \
  --training-shield-method cp \
  --seed 0 \
  --checkpoint-dir '/home/mars/Desktop/conformal shield/results_intersample/runs/aligned_cp_seed_0' \
  --dense-safety-cost \
  --no-tqdm \
  --recompute-cbvf \
  --terminate-on-collision \
  --cache-dir '/home/mars/Desktop/conformal shield/cbvf_cache_intersample' \
  --odp-root '/home/mars/Desktop/conformal shield' \
  --solver-file '/home/mars/Desktop/conformal shield/solver_cbvf.py' \
  --eval-jsonl '/home/mars/Desktop/conformal shield/results_intersample/runs/aligned_cp_seed_0/training_eval_metrics.jsonl' \
  --action-log-jsonl '/home/mars/Desktop/conformal shield/results_intersample/runs/aligned_cp_seed_0/training_actions.jsonl' \
  --qp-diagnostics-jsonl '/home/mars/Desktop/conformal shield/results_intersample/runs/aligned_cp_seed_0/qp_feasibility_diagnostics.jsonl' \
  --results-json '/home/mars/Desktop/conformal shield/results_intersample/runs/aligned_cp_seed_0/training_and_evaluation_results.json' \
  --plot-dir '/home/mars/Desktop/conformal shield/results_intersample/runs/aligned_cp_seed_0/plots'
```

## aligned_cp_seed_99

```bash
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
  --integration-substeps 10 \
  --intersample-protection \
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
  --speed-min -0.2 \
  --speed 0.6 \
  --beta-u 0.8 \
  --cbvf-model-speed-min -0.2 \
  --cbvf-model-speed 0.6 \
  --cbvf-model-beta-u 0.8 \
  --training-shield-method cp \
  --seed 99 \
  --checkpoint-dir '/home/mars/Desktop/conformal shield/results_intersample/runs/aligned_cp_seed_99' \
  --dense-safety-cost \
  --no-tqdm \
  --recompute-cbvf \
  --terminate-on-collision \
  --cache-dir '/home/mars/Desktop/conformal shield/cbvf_cache_intersample' \
  --odp-root '/home/mars/Desktop/conformal shield' \
  --solver-file '/home/mars/Desktop/conformal shield/solver_cbvf.py' \
  --eval-jsonl '/home/mars/Desktop/conformal shield/results_intersample/runs/aligned_cp_seed_99/training_eval_metrics.jsonl' \
  --action-log-jsonl '/home/mars/Desktop/conformal shield/results_intersample/runs/aligned_cp_seed_99/training_actions.jsonl' \
  --qp-diagnostics-jsonl '/home/mars/Desktop/conformal shield/results_intersample/runs/aligned_cp_seed_99/qp_feasibility_diagnostics.jsonl' \
  --results-json '/home/mars/Desktop/conformal shield/results_intersample/runs/aligned_cp_seed_99/training_and_evaluation_results.json' \
  --plot-dir '/home/mars/Desktop/conformal shield/results_intersample/runs/aligned_cp_seed_99/plots'
```

## easy_nominal_seed_7

```bash
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
  --integration-substeps 10 \
  --intersample-protection \
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
  --speed-min -0.1 \
  --speed 0.6 \
  --beta-u 0.7 \
  --cbvf-model-speed-min -0.2 \
  --cbvf-model-speed 0.6 \
  --cbvf-model-beta-u 0.8 \
  --training-shield-method nominal \
  --seed 7 \
  --checkpoint-dir '/home/mars/Desktop/conformal shield/results_intersample/runs/easy_nominal_seed_7' \
  --dense-safety-cost \
  --no-tqdm \
  --recompute-cbvf \
  --terminate-on-collision \
  --cache-dir '/home/mars/Desktop/conformal shield/cbvf_cache_intersample' \
  --odp-root '/home/mars/Desktop/conformal shield' \
  --solver-file '/home/mars/Desktop/conformal shield/solver_cbvf.py' \
  --eval-jsonl '/home/mars/Desktop/conformal shield/results_intersample/runs/easy_nominal_seed_7/training_eval_metrics.jsonl' \
  --action-log-jsonl '/home/mars/Desktop/conformal shield/results_intersample/runs/easy_nominal_seed_7/training_actions.jsonl' \
  --qp-diagnostics-jsonl '/home/mars/Desktop/conformal shield/results_intersample/runs/easy_nominal_seed_7/qp_feasibility_diagnostics.jsonl' \
  --results-json '/home/mars/Desktop/conformal shield/results_intersample/runs/easy_nominal_seed_7/training_and_evaluation_results.json' \
  --plot-dir '/home/mars/Desktop/conformal shield/results_intersample/runs/easy_nominal_seed_7/plots'
```

## easy_nominal_seed_42

```bash
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
  --integration-substeps 10 \
  --intersample-protection \
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
  --speed-min -0.1 \
  --speed 0.6 \
  --beta-u 0.7 \
  --cbvf-model-speed-min -0.2 \
  --cbvf-model-speed 0.6 \
  --cbvf-model-beta-u 0.8 \
  --training-shield-method nominal \
  --seed 42 \
  --checkpoint-dir '/home/mars/Desktop/conformal shield/results_intersample/runs/easy_nominal_seed_42' \
  --dense-safety-cost \
  --no-tqdm \
  --recompute-cbvf \
  --terminate-on-collision \
  --cache-dir '/home/mars/Desktop/conformal shield/cbvf_cache_intersample' \
  --odp-root '/home/mars/Desktop/conformal shield' \
  --solver-file '/home/mars/Desktop/conformal shield/solver_cbvf.py' \
  --eval-jsonl '/home/mars/Desktop/conformal shield/results_intersample/runs/easy_nominal_seed_42/training_eval_metrics.jsonl' \
  --action-log-jsonl '/home/mars/Desktop/conformal shield/results_intersample/runs/easy_nominal_seed_42/training_actions.jsonl' \
  --qp-diagnostics-jsonl '/home/mars/Desktop/conformal shield/results_intersample/runs/easy_nominal_seed_42/qp_feasibility_diagnostics.jsonl' \
  --results-json '/home/mars/Desktop/conformal shield/results_intersample/runs/easy_nominal_seed_42/training_and_evaluation_results.json' \
  --plot-dir '/home/mars/Desktop/conformal shield/results_intersample/runs/easy_nominal_seed_42/plots'
```

## easy_nominal_seed_123

```bash
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
  --integration-substeps 10 \
  --intersample-protection \
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
  --speed-min -0.1 \
  --speed 0.6 \
  --beta-u 0.7 \
  --cbvf-model-speed-min -0.2 \
  --cbvf-model-speed 0.6 \
  --cbvf-model-beta-u 0.8 \
  --training-shield-method nominal \
  --seed 123 \
  --checkpoint-dir '/home/mars/Desktop/conformal shield/results_intersample/runs/easy_nominal_seed_123' \
  --dense-safety-cost \
  --no-tqdm \
  --recompute-cbvf \
  --terminate-on-collision \
  --cache-dir '/home/mars/Desktop/conformal shield/cbvf_cache_intersample' \
  --odp-root '/home/mars/Desktop/conformal shield' \
  --solver-file '/home/mars/Desktop/conformal shield/solver_cbvf.py' \
  --eval-jsonl '/home/mars/Desktop/conformal shield/results_intersample/runs/easy_nominal_seed_123/training_eval_metrics.jsonl' \
  --action-log-jsonl '/home/mars/Desktop/conformal shield/results_intersample/runs/easy_nominal_seed_123/training_actions.jsonl' \
  --qp-diagnostics-jsonl '/home/mars/Desktop/conformal shield/results_intersample/runs/easy_nominal_seed_123/qp_feasibility_diagnostics.jsonl' \
  --results-json '/home/mars/Desktop/conformal shield/results_intersample/runs/easy_nominal_seed_123/training_and_evaluation_results.json' \
  --plot-dir '/home/mars/Desktop/conformal shield/results_intersample/runs/easy_nominal_seed_123/plots'
```

## easy_nominal_seed_0

```bash
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
  --integration-substeps 10 \
  --intersample-protection \
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
  --speed-min -0.1 \
  --speed 0.6 \
  --beta-u 0.7 \
  --cbvf-model-speed-min -0.2 \
  --cbvf-model-speed 0.6 \
  --cbvf-model-beta-u 0.8 \
  --training-shield-method nominal \
  --seed 0 \
  --checkpoint-dir '/home/mars/Desktop/conformal shield/results_intersample/runs/easy_nominal_seed_0' \
  --dense-safety-cost \
  --no-tqdm \
  --recompute-cbvf \
  --terminate-on-collision \
  --cache-dir '/home/mars/Desktop/conformal shield/cbvf_cache_intersample' \
  --odp-root '/home/mars/Desktop/conformal shield' \
  --solver-file '/home/mars/Desktop/conformal shield/solver_cbvf.py' \
  --eval-jsonl '/home/mars/Desktop/conformal shield/results_intersample/runs/easy_nominal_seed_0/training_eval_metrics.jsonl' \
  --action-log-jsonl '/home/mars/Desktop/conformal shield/results_intersample/runs/easy_nominal_seed_0/training_actions.jsonl' \
  --qp-diagnostics-jsonl '/home/mars/Desktop/conformal shield/results_intersample/runs/easy_nominal_seed_0/qp_feasibility_diagnostics.jsonl' \
  --results-json '/home/mars/Desktop/conformal shield/results_intersample/runs/easy_nominal_seed_0/training_and_evaluation_results.json' \
  --plot-dir '/home/mars/Desktop/conformal shield/results_intersample/runs/easy_nominal_seed_0/plots'
```

## easy_nominal_seed_99

```bash
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
  --integration-substeps 10 \
  --intersample-protection \
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
  --speed-min -0.1 \
  --speed 0.6 \
  --beta-u 0.7 \
  --cbvf-model-speed-min -0.2 \
  --cbvf-model-speed 0.6 \
  --cbvf-model-beta-u 0.8 \
  --training-shield-method nominal \
  --seed 99 \
  --checkpoint-dir '/home/mars/Desktop/conformal shield/results_intersample/runs/easy_nominal_seed_99' \
  --dense-safety-cost \
  --no-tqdm \
  --recompute-cbvf \
  --terminate-on-collision \
  --cache-dir '/home/mars/Desktop/conformal shield/cbvf_cache_intersample' \
  --odp-root '/home/mars/Desktop/conformal shield' \
  --solver-file '/home/mars/Desktop/conformal shield/solver_cbvf.py' \
  --eval-jsonl '/home/mars/Desktop/conformal shield/results_intersample/runs/easy_nominal_seed_99/training_eval_metrics.jsonl' \
  --action-log-jsonl '/home/mars/Desktop/conformal shield/results_intersample/runs/easy_nominal_seed_99/training_actions.jsonl' \
  --qp-diagnostics-jsonl '/home/mars/Desktop/conformal shield/results_intersample/runs/easy_nominal_seed_99/qp_feasibility_diagnostics.jsonl' \
  --results-json '/home/mars/Desktop/conformal shield/results_intersample/runs/easy_nominal_seed_99/training_and_evaluation_results.json' \
  --plot-dir '/home/mars/Desktop/conformal shield/results_intersample/runs/easy_nominal_seed_99/plots'
```

## easy_cbvf_seed_7

```bash
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
  --integration-substeps 10 \
  --intersample-protection \
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
  --speed-min -0.1 \
  --speed 0.6 \
  --beta-u 0.7 \
  --cbvf-model-speed-min -0.2 \
  --cbvf-model-speed 0.6 \
  --cbvf-model-beta-u 0.8 \
  --training-shield-method cbvf \
  --seed 7 \
  --checkpoint-dir '/home/mars/Desktop/conformal shield/results_intersample/runs/easy_cbvf_seed_7' \
  --dense-safety-cost \
  --no-tqdm \
  --recompute-cbvf \
  --terminate-on-collision \
  --cache-dir '/home/mars/Desktop/conformal shield/cbvf_cache_intersample' \
  --odp-root '/home/mars/Desktop/conformal shield' \
  --solver-file '/home/mars/Desktop/conformal shield/solver_cbvf.py' \
  --eval-jsonl '/home/mars/Desktop/conformal shield/results_intersample/runs/easy_cbvf_seed_7/training_eval_metrics.jsonl' \
  --action-log-jsonl '/home/mars/Desktop/conformal shield/results_intersample/runs/easy_cbvf_seed_7/training_actions.jsonl' \
  --qp-diagnostics-jsonl '/home/mars/Desktop/conformal shield/results_intersample/runs/easy_cbvf_seed_7/qp_feasibility_diagnostics.jsonl' \
  --results-json '/home/mars/Desktop/conformal shield/results_intersample/runs/easy_cbvf_seed_7/training_and_evaluation_results.json' \
  --plot-dir '/home/mars/Desktop/conformal shield/results_intersample/runs/easy_cbvf_seed_7/plots'
```

## easy_cbvf_seed_42

```bash
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
  --integration-substeps 10 \
  --intersample-protection \
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
  --speed-min -0.1 \
  --speed 0.6 \
  --beta-u 0.7 \
  --cbvf-model-speed-min -0.2 \
  --cbvf-model-speed 0.6 \
  --cbvf-model-beta-u 0.8 \
  --training-shield-method cbvf \
  --seed 42 \
  --checkpoint-dir '/home/mars/Desktop/conformal shield/results_intersample/runs/easy_cbvf_seed_42' \
  --dense-safety-cost \
  --no-tqdm \
  --recompute-cbvf \
  --terminate-on-collision \
  --cache-dir '/home/mars/Desktop/conformal shield/cbvf_cache_intersample' \
  --odp-root '/home/mars/Desktop/conformal shield' \
  --solver-file '/home/mars/Desktop/conformal shield/solver_cbvf.py' \
  --eval-jsonl '/home/mars/Desktop/conformal shield/results_intersample/runs/easy_cbvf_seed_42/training_eval_metrics.jsonl' \
  --action-log-jsonl '/home/mars/Desktop/conformal shield/results_intersample/runs/easy_cbvf_seed_42/training_actions.jsonl' \
  --qp-diagnostics-jsonl '/home/mars/Desktop/conformal shield/results_intersample/runs/easy_cbvf_seed_42/qp_feasibility_diagnostics.jsonl' \
  --results-json '/home/mars/Desktop/conformal shield/results_intersample/runs/easy_cbvf_seed_42/training_and_evaluation_results.json' \
  --plot-dir '/home/mars/Desktop/conformal shield/results_intersample/runs/easy_cbvf_seed_42/plots'
```

## easy_cbvf_seed_123

```bash
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
  --integration-substeps 10 \
  --intersample-protection \
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
  --speed-min -0.1 \
  --speed 0.6 \
  --beta-u 0.7 \
  --cbvf-model-speed-min -0.2 \
  --cbvf-model-speed 0.6 \
  --cbvf-model-beta-u 0.8 \
  --training-shield-method cbvf \
  --seed 123 \
  --checkpoint-dir '/home/mars/Desktop/conformal shield/results_intersample/runs/easy_cbvf_seed_123' \
  --dense-safety-cost \
  --no-tqdm \
  --recompute-cbvf \
  --terminate-on-collision \
  --cache-dir '/home/mars/Desktop/conformal shield/cbvf_cache_intersample' \
  --odp-root '/home/mars/Desktop/conformal shield' \
  --solver-file '/home/mars/Desktop/conformal shield/solver_cbvf.py' \
  --eval-jsonl '/home/mars/Desktop/conformal shield/results_intersample/runs/easy_cbvf_seed_123/training_eval_metrics.jsonl' \
  --action-log-jsonl '/home/mars/Desktop/conformal shield/results_intersample/runs/easy_cbvf_seed_123/training_actions.jsonl' \
  --qp-diagnostics-jsonl '/home/mars/Desktop/conformal shield/results_intersample/runs/easy_cbvf_seed_123/qp_feasibility_diagnostics.jsonl' \
  --results-json '/home/mars/Desktop/conformal shield/results_intersample/runs/easy_cbvf_seed_123/training_and_evaluation_results.json' \
  --plot-dir '/home/mars/Desktop/conformal shield/results_intersample/runs/easy_cbvf_seed_123/plots'
```

## easy_cbvf_seed_0

```bash
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
  --integration-substeps 10 \
  --intersample-protection \
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
  --speed-min -0.1 \
  --speed 0.6 \
  --beta-u 0.7 \
  --cbvf-model-speed-min -0.2 \
  --cbvf-model-speed 0.6 \
  --cbvf-model-beta-u 0.8 \
  --training-shield-method cbvf \
  --seed 0 \
  --checkpoint-dir '/home/mars/Desktop/conformal shield/results_intersample/runs/easy_cbvf_seed_0' \
  --dense-safety-cost \
  --no-tqdm \
  --recompute-cbvf \
  --terminate-on-collision \
  --cache-dir '/home/mars/Desktop/conformal shield/cbvf_cache_intersample' \
  --odp-root '/home/mars/Desktop/conformal shield' \
  --solver-file '/home/mars/Desktop/conformal shield/solver_cbvf.py' \
  --eval-jsonl '/home/mars/Desktop/conformal shield/results_intersample/runs/easy_cbvf_seed_0/training_eval_metrics.jsonl' \
  --action-log-jsonl '/home/mars/Desktop/conformal shield/results_intersample/runs/easy_cbvf_seed_0/training_actions.jsonl' \
  --qp-diagnostics-jsonl '/home/mars/Desktop/conformal shield/results_intersample/runs/easy_cbvf_seed_0/qp_feasibility_diagnostics.jsonl' \
  --results-json '/home/mars/Desktop/conformal shield/results_intersample/runs/easy_cbvf_seed_0/training_and_evaluation_results.json' \
  --plot-dir '/home/mars/Desktop/conformal shield/results_intersample/runs/easy_cbvf_seed_0/plots'
```

## easy_cbvf_seed_99

```bash
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
  --integration-substeps 10 \
  --intersample-protection \
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
  --speed-min -0.1 \
  --speed 0.6 \
  --beta-u 0.7 \
  --cbvf-model-speed-min -0.2 \
  --cbvf-model-speed 0.6 \
  --cbvf-model-beta-u 0.8 \
  --training-shield-method cbvf \
  --seed 99 \
  --checkpoint-dir '/home/mars/Desktop/conformal shield/results_intersample/runs/easy_cbvf_seed_99' \
  --dense-safety-cost \
  --no-tqdm \
  --recompute-cbvf \
  --terminate-on-collision \
  --cache-dir '/home/mars/Desktop/conformal shield/cbvf_cache_intersample' \
  --odp-root '/home/mars/Desktop/conformal shield' \
  --solver-file '/home/mars/Desktop/conformal shield/solver_cbvf.py' \
  --eval-jsonl '/home/mars/Desktop/conformal shield/results_intersample/runs/easy_cbvf_seed_99/training_eval_metrics.jsonl' \
  --action-log-jsonl '/home/mars/Desktop/conformal shield/results_intersample/runs/easy_cbvf_seed_99/training_actions.jsonl' \
  --qp-diagnostics-jsonl '/home/mars/Desktop/conformal shield/results_intersample/runs/easy_cbvf_seed_99/qp_feasibility_diagnostics.jsonl' \
  --results-json '/home/mars/Desktop/conformal shield/results_intersample/runs/easy_cbvf_seed_99/training_and_evaluation_results.json' \
  --plot-dir '/home/mars/Desktop/conformal shield/results_intersample/runs/easy_cbvf_seed_99/plots'
```

## easy_cp_seed_7

```bash
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
  --integration-substeps 10 \
  --intersample-protection \
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
  --speed-min -0.1 \
  --speed 0.6 \
  --beta-u 0.7 \
  --cbvf-model-speed-min -0.2 \
  --cbvf-model-speed 0.6 \
  --cbvf-model-beta-u 0.8 \
  --training-shield-method cp \
  --seed 7 \
  --checkpoint-dir '/home/mars/Desktop/conformal shield/results_intersample/runs/easy_cp_seed_7' \
  --dense-safety-cost \
  --no-tqdm \
  --recompute-cbvf \
  --terminate-on-collision \
  --cache-dir '/home/mars/Desktop/conformal shield/cbvf_cache_intersample' \
  --odp-root '/home/mars/Desktop/conformal shield' \
  --solver-file '/home/mars/Desktop/conformal shield/solver_cbvf.py' \
  --eval-jsonl '/home/mars/Desktop/conformal shield/results_intersample/runs/easy_cp_seed_7/training_eval_metrics.jsonl' \
  --action-log-jsonl '/home/mars/Desktop/conformal shield/results_intersample/runs/easy_cp_seed_7/training_actions.jsonl' \
  --qp-diagnostics-jsonl '/home/mars/Desktop/conformal shield/results_intersample/runs/easy_cp_seed_7/qp_feasibility_diagnostics.jsonl' \
  --results-json '/home/mars/Desktop/conformal shield/results_intersample/runs/easy_cp_seed_7/training_and_evaluation_results.json' \
  --plot-dir '/home/mars/Desktop/conformal shield/results_intersample/runs/easy_cp_seed_7/plots'
```

## easy_cp_seed_42

```bash
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
  --integration-substeps 10 \
  --intersample-protection \
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
  --speed-min -0.1 \
  --speed 0.6 \
  --beta-u 0.7 \
  --cbvf-model-speed-min -0.2 \
  --cbvf-model-speed 0.6 \
  --cbvf-model-beta-u 0.8 \
  --training-shield-method cp \
  --seed 42 \
  --checkpoint-dir '/home/mars/Desktop/conformal shield/results_intersample/runs/easy_cp_seed_42' \
  --dense-safety-cost \
  --no-tqdm \
  --recompute-cbvf \
  --terminate-on-collision \
  --cache-dir '/home/mars/Desktop/conformal shield/cbvf_cache_intersample' \
  --odp-root '/home/mars/Desktop/conformal shield' \
  --solver-file '/home/mars/Desktop/conformal shield/solver_cbvf.py' \
  --eval-jsonl '/home/mars/Desktop/conformal shield/results_intersample/runs/easy_cp_seed_42/training_eval_metrics.jsonl' \
  --action-log-jsonl '/home/mars/Desktop/conformal shield/results_intersample/runs/easy_cp_seed_42/training_actions.jsonl' \
  --qp-diagnostics-jsonl '/home/mars/Desktop/conformal shield/results_intersample/runs/easy_cp_seed_42/qp_feasibility_diagnostics.jsonl' \
  --results-json '/home/mars/Desktop/conformal shield/results_intersample/runs/easy_cp_seed_42/training_and_evaluation_results.json' \
  --plot-dir '/home/mars/Desktop/conformal shield/results_intersample/runs/easy_cp_seed_42/plots'
```

## easy_cp_seed_123

```bash
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
  --integration-substeps 10 \
  --intersample-protection \
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
  --speed-min -0.1 \
  --speed 0.6 \
  --beta-u 0.7 \
  --cbvf-model-speed-min -0.2 \
  --cbvf-model-speed 0.6 \
  --cbvf-model-beta-u 0.8 \
  --training-shield-method cp \
  --seed 123 \
  --checkpoint-dir '/home/mars/Desktop/conformal shield/results_intersample/runs/easy_cp_seed_123' \
  --dense-safety-cost \
  --no-tqdm \
  --recompute-cbvf \
  --terminate-on-collision \
  --cache-dir '/home/mars/Desktop/conformal shield/cbvf_cache_intersample' \
  --odp-root '/home/mars/Desktop/conformal shield' \
  --solver-file '/home/mars/Desktop/conformal shield/solver_cbvf.py' \
  --eval-jsonl '/home/mars/Desktop/conformal shield/results_intersample/runs/easy_cp_seed_123/training_eval_metrics.jsonl' \
  --action-log-jsonl '/home/mars/Desktop/conformal shield/results_intersample/runs/easy_cp_seed_123/training_actions.jsonl' \
  --qp-diagnostics-jsonl '/home/mars/Desktop/conformal shield/results_intersample/runs/easy_cp_seed_123/qp_feasibility_diagnostics.jsonl' \
  --results-json '/home/mars/Desktop/conformal shield/results_intersample/runs/easy_cp_seed_123/training_and_evaluation_results.json' \
  --plot-dir '/home/mars/Desktop/conformal shield/results_intersample/runs/easy_cp_seed_123/plots'
```

## easy_cp_seed_0

```bash
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
  --integration-substeps 10 \
  --intersample-protection \
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
  --speed-min -0.1 \
  --speed 0.6 \
  --beta-u 0.7 \
  --cbvf-model-speed-min -0.2 \
  --cbvf-model-speed 0.6 \
  --cbvf-model-beta-u 0.8 \
  --training-shield-method cp \
  --seed 0 \
  --checkpoint-dir '/home/mars/Desktop/conformal shield/results_intersample/runs/easy_cp_seed_0' \
  --dense-safety-cost \
  --no-tqdm \
  --recompute-cbvf \
  --terminate-on-collision \
  --cache-dir '/home/mars/Desktop/conformal shield/cbvf_cache_intersample' \
  --odp-root '/home/mars/Desktop/conformal shield' \
  --solver-file '/home/mars/Desktop/conformal shield/solver_cbvf.py' \
  --eval-jsonl '/home/mars/Desktop/conformal shield/results_intersample/runs/easy_cp_seed_0/training_eval_metrics.jsonl' \
  --action-log-jsonl '/home/mars/Desktop/conformal shield/results_intersample/runs/easy_cp_seed_0/training_actions.jsonl' \
  --qp-diagnostics-jsonl '/home/mars/Desktop/conformal shield/results_intersample/runs/easy_cp_seed_0/qp_feasibility_diagnostics.jsonl' \
  --results-json '/home/mars/Desktop/conformal shield/results_intersample/runs/easy_cp_seed_0/training_and_evaluation_results.json' \
  --plot-dir '/home/mars/Desktop/conformal shield/results_intersample/runs/easy_cp_seed_0/plots'
```

## easy_cp_seed_99

```bash
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
  --integration-substeps 10 \
  --intersample-protection \
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
  --speed-min -0.1 \
  --speed 0.6 \
  --beta-u 0.7 \
  --cbvf-model-speed-min -0.2 \
  --cbvf-model-speed 0.6 \
  --cbvf-model-beta-u 0.8 \
  --training-shield-method cp \
  --seed 99 \
  --checkpoint-dir '/home/mars/Desktop/conformal shield/results_intersample/runs/easy_cp_seed_99' \
  --dense-safety-cost \
  --no-tqdm \
  --recompute-cbvf \
  --terminate-on-collision \
  --cache-dir '/home/mars/Desktop/conformal shield/cbvf_cache_intersample' \
  --odp-root '/home/mars/Desktop/conformal shield' \
  --solver-file '/home/mars/Desktop/conformal shield/solver_cbvf.py' \
  --eval-jsonl '/home/mars/Desktop/conformal shield/results_intersample/runs/easy_cp_seed_99/training_eval_metrics.jsonl' \
  --action-log-jsonl '/home/mars/Desktop/conformal shield/results_intersample/runs/easy_cp_seed_99/training_actions.jsonl' \
  --qp-diagnostics-jsonl '/home/mars/Desktop/conformal shield/results_intersample/runs/easy_cp_seed_99/qp_feasibility_diagnostics.jsonl' \
  --results-json '/home/mars/Desktop/conformal shield/results_intersample/runs/easy_cp_seed_99/training_and_evaluation_results.json' \
  --plot-dir '/home/mars/Desktop/conformal shield/results_intersample/runs/easy_cp_seed_99/plots'
```

## hard_nominal_seed_7

```bash
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
  --integration-substeps 10 \
  --intersample-protection \
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
  --speed-min 0.0 \
  --speed 0.6 \
  --beta-u 0.6 \
  --cbvf-model-speed-min -0.2 \
  --cbvf-model-speed 0.6 \
  --cbvf-model-beta-u 0.8 \
  --training-shield-method nominal \
  --seed 7 \
  --checkpoint-dir '/home/mars/Desktop/conformal shield/results_intersample/runs/hard_nominal_seed_7' \
  --dense-safety-cost \
  --no-tqdm \
  --recompute-cbvf \
  --terminate-on-collision \
  --cache-dir '/home/mars/Desktop/conformal shield/cbvf_cache_intersample' \
  --odp-root '/home/mars/Desktop/conformal shield' \
  --solver-file '/home/mars/Desktop/conformal shield/solver_cbvf.py' \
  --eval-jsonl '/home/mars/Desktop/conformal shield/results_intersample/runs/hard_nominal_seed_7/training_eval_metrics.jsonl' \
  --action-log-jsonl '/home/mars/Desktop/conformal shield/results_intersample/runs/hard_nominal_seed_7/training_actions.jsonl' \
  --qp-diagnostics-jsonl '/home/mars/Desktop/conformal shield/results_intersample/runs/hard_nominal_seed_7/qp_feasibility_diagnostics.jsonl' \
  --results-json '/home/mars/Desktop/conformal shield/results_intersample/runs/hard_nominal_seed_7/training_and_evaluation_results.json' \
  --plot-dir '/home/mars/Desktop/conformal shield/results_intersample/runs/hard_nominal_seed_7/plots'
```

## hard_nominal_seed_42

```bash
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
  --integration-substeps 10 \
  --intersample-protection \
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
  --speed-min 0.0 \
  --speed 0.6 \
  --beta-u 0.6 \
  --cbvf-model-speed-min -0.2 \
  --cbvf-model-speed 0.6 \
  --cbvf-model-beta-u 0.8 \
  --training-shield-method nominal \
  --seed 42 \
  --checkpoint-dir '/home/mars/Desktop/conformal shield/results_intersample/runs/hard_nominal_seed_42' \
  --dense-safety-cost \
  --no-tqdm \
  --recompute-cbvf \
  --terminate-on-collision \
  --cache-dir '/home/mars/Desktop/conformal shield/cbvf_cache_intersample' \
  --odp-root '/home/mars/Desktop/conformal shield' \
  --solver-file '/home/mars/Desktop/conformal shield/solver_cbvf.py' \
  --eval-jsonl '/home/mars/Desktop/conformal shield/results_intersample/runs/hard_nominal_seed_42/training_eval_metrics.jsonl' \
  --action-log-jsonl '/home/mars/Desktop/conformal shield/results_intersample/runs/hard_nominal_seed_42/training_actions.jsonl' \
  --qp-diagnostics-jsonl '/home/mars/Desktop/conformal shield/results_intersample/runs/hard_nominal_seed_42/qp_feasibility_diagnostics.jsonl' \
  --results-json '/home/mars/Desktop/conformal shield/results_intersample/runs/hard_nominal_seed_42/training_and_evaluation_results.json' \
  --plot-dir '/home/mars/Desktop/conformal shield/results_intersample/runs/hard_nominal_seed_42/plots'
```

## hard_nominal_seed_123

```bash
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
  --integration-substeps 10 \
  --intersample-protection \
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
  --speed-min 0.0 \
  --speed 0.6 \
  --beta-u 0.6 \
  --cbvf-model-speed-min -0.2 \
  --cbvf-model-speed 0.6 \
  --cbvf-model-beta-u 0.8 \
  --training-shield-method nominal \
  --seed 123 \
  --checkpoint-dir '/home/mars/Desktop/conformal shield/results_intersample/runs/hard_nominal_seed_123' \
  --dense-safety-cost \
  --no-tqdm \
  --recompute-cbvf \
  --terminate-on-collision \
  --cache-dir '/home/mars/Desktop/conformal shield/cbvf_cache_intersample' \
  --odp-root '/home/mars/Desktop/conformal shield' \
  --solver-file '/home/mars/Desktop/conformal shield/solver_cbvf.py' \
  --eval-jsonl '/home/mars/Desktop/conformal shield/results_intersample/runs/hard_nominal_seed_123/training_eval_metrics.jsonl' \
  --action-log-jsonl '/home/mars/Desktop/conformal shield/results_intersample/runs/hard_nominal_seed_123/training_actions.jsonl' \
  --qp-diagnostics-jsonl '/home/mars/Desktop/conformal shield/results_intersample/runs/hard_nominal_seed_123/qp_feasibility_diagnostics.jsonl' \
  --results-json '/home/mars/Desktop/conformal shield/results_intersample/runs/hard_nominal_seed_123/training_and_evaluation_results.json' \
  --plot-dir '/home/mars/Desktop/conformal shield/results_intersample/runs/hard_nominal_seed_123/plots'
```

## hard_nominal_seed_0

```bash
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
  --integration-substeps 10 \
  --intersample-protection \
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
  --speed-min 0.0 \
  --speed 0.6 \
  --beta-u 0.6 \
  --cbvf-model-speed-min -0.2 \
  --cbvf-model-speed 0.6 \
  --cbvf-model-beta-u 0.8 \
  --training-shield-method nominal \
  --seed 0 \
  --checkpoint-dir '/home/mars/Desktop/conformal shield/results_intersample/runs/hard_nominal_seed_0' \
  --dense-safety-cost \
  --no-tqdm \
  --recompute-cbvf \
  --terminate-on-collision \
  --cache-dir '/home/mars/Desktop/conformal shield/cbvf_cache_intersample' \
  --odp-root '/home/mars/Desktop/conformal shield' \
  --solver-file '/home/mars/Desktop/conformal shield/solver_cbvf.py' \
  --eval-jsonl '/home/mars/Desktop/conformal shield/results_intersample/runs/hard_nominal_seed_0/training_eval_metrics.jsonl' \
  --action-log-jsonl '/home/mars/Desktop/conformal shield/results_intersample/runs/hard_nominal_seed_0/training_actions.jsonl' \
  --qp-diagnostics-jsonl '/home/mars/Desktop/conformal shield/results_intersample/runs/hard_nominal_seed_0/qp_feasibility_diagnostics.jsonl' \
  --results-json '/home/mars/Desktop/conformal shield/results_intersample/runs/hard_nominal_seed_0/training_and_evaluation_results.json' \
  --plot-dir '/home/mars/Desktop/conformal shield/results_intersample/runs/hard_nominal_seed_0/plots'
```

## hard_nominal_seed_99

```bash
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
  --integration-substeps 10 \
  --intersample-protection \
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
  --speed-min 0.0 \
  --speed 0.6 \
  --beta-u 0.6 \
  --cbvf-model-speed-min -0.2 \
  --cbvf-model-speed 0.6 \
  --cbvf-model-beta-u 0.8 \
  --training-shield-method nominal \
  --seed 99 \
  --checkpoint-dir '/home/mars/Desktop/conformal shield/results_intersample/runs/hard_nominal_seed_99' \
  --dense-safety-cost \
  --no-tqdm \
  --recompute-cbvf \
  --terminate-on-collision \
  --cache-dir '/home/mars/Desktop/conformal shield/cbvf_cache_intersample' \
  --odp-root '/home/mars/Desktop/conformal shield' \
  --solver-file '/home/mars/Desktop/conformal shield/solver_cbvf.py' \
  --eval-jsonl '/home/mars/Desktop/conformal shield/results_intersample/runs/hard_nominal_seed_99/training_eval_metrics.jsonl' \
  --action-log-jsonl '/home/mars/Desktop/conformal shield/results_intersample/runs/hard_nominal_seed_99/training_actions.jsonl' \
  --qp-diagnostics-jsonl '/home/mars/Desktop/conformal shield/results_intersample/runs/hard_nominal_seed_99/qp_feasibility_diagnostics.jsonl' \
  --results-json '/home/mars/Desktop/conformal shield/results_intersample/runs/hard_nominal_seed_99/training_and_evaluation_results.json' \
  --plot-dir '/home/mars/Desktop/conformal shield/results_intersample/runs/hard_nominal_seed_99/plots'
```

## hard_cbvf_seed_7

```bash
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
  --integration-substeps 10 \
  --intersample-protection \
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
  --speed-min 0.0 \
  --speed 0.6 \
  --beta-u 0.6 \
  --cbvf-model-speed-min -0.2 \
  --cbvf-model-speed 0.6 \
  --cbvf-model-beta-u 0.8 \
  --training-shield-method cbvf \
  --seed 7 \
  --checkpoint-dir '/home/mars/Desktop/conformal shield/results_intersample/runs/hard_cbvf_seed_7' \
  --dense-safety-cost \
  --no-tqdm \
  --recompute-cbvf \
  --terminate-on-collision \
  --cache-dir '/home/mars/Desktop/conformal shield/cbvf_cache_intersample' \
  --odp-root '/home/mars/Desktop/conformal shield' \
  --solver-file '/home/mars/Desktop/conformal shield/solver_cbvf.py' \
  --eval-jsonl '/home/mars/Desktop/conformal shield/results_intersample/runs/hard_cbvf_seed_7/training_eval_metrics.jsonl' \
  --action-log-jsonl '/home/mars/Desktop/conformal shield/results_intersample/runs/hard_cbvf_seed_7/training_actions.jsonl' \
  --qp-diagnostics-jsonl '/home/mars/Desktop/conformal shield/results_intersample/runs/hard_cbvf_seed_7/qp_feasibility_diagnostics.jsonl' \
  --results-json '/home/mars/Desktop/conformal shield/results_intersample/runs/hard_cbvf_seed_7/training_and_evaluation_results.json' \
  --plot-dir '/home/mars/Desktop/conformal shield/results_intersample/runs/hard_cbvf_seed_7/plots'
```

## hard_cbvf_seed_42

```bash
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
  --integration-substeps 10 \
  --intersample-protection \
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
  --speed-min 0.0 \
  --speed 0.6 \
  --beta-u 0.6 \
  --cbvf-model-speed-min -0.2 \
  --cbvf-model-speed 0.6 \
  --cbvf-model-beta-u 0.8 \
  --training-shield-method cbvf \
  --seed 42 \
  --checkpoint-dir '/home/mars/Desktop/conformal shield/results_intersample/runs/hard_cbvf_seed_42' \
  --dense-safety-cost \
  --no-tqdm \
  --recompute-cbvf \
  --terminate-on-collision \
  --cache-dir '/home/mars/Desktop/conformal shield/cbvf_cache_intersample' \
  --odp-root '/home/mars/Desktop/conformal shield' \
  --solver-file '/home/mars/Desktop/conformal shield/solver_cbvf.py' \
  --eval-jsonl '/home/mars/Desktop/conformal shield/results_intersample/runs/hard_cbvf_seed_42/training_eval_metrics.jsonl' \
  --action-log-jsonl '/home/mars/Desktop/conformal shield/results_intersample/runs/hard_cbvf_seed_42/training_actions.jsonl' \
  --qp-diagnostics-jsonl '/home/mars/Desktop/conformal shield/results_intersample/runs/hard_cbvf_seed_42/qp_feasibility_diagnostics.jsonl' \
  --results-json '/home/mars/Desktop/conformal shield/results_intersample/runs/hard_cbvf_seed_42/training_and_evaluation_results.json' \
  --plot-dir '/home/mars/Desktop/conformal shield/results_intersample/runs/hard_cbvf_seed_42/plots'
```

## hard_cbvf_seed_123

```bash
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
  --integration-substeps 10 \
  --intersample-protection \
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
  --speed-min 0.0 \
  --speed 0.6 \
  --beta-u 0.6 \
  --cbvf-model-speed-min -0.2 \
  --cbvf-model-speed 0.6 \
  --cbvf-model-beta-u 0.8 \
  --training-shield-method cbvf \
  --seed 123 \
  --checkpoint-dir '/home/mars/Desktop/conformal shield/results_intersample/runs/hard_cbvf_seed_123' \
  --dense-safety-cost \
  --no-tqdm \
  --recompute-cbvf \
  --terminate-on-collision \
  --cache-dir '/home/mars/Desktop/conformal shield/cbvf_cache_intersample' \
  --odp-root '/home/mars/Desktop/conformal shield' \
  --solver-file '/home/mars/Desktop/conformal shield/solver_cbvf.py' \
  --eval-jsonl '/home/mars/Desktop/conformal shield/results_intersample/runs/hard_cbvf_seed_123/training_eval_metrics.jsonl' \
  --action-log-jsonl '/home/mars/Desktop/conformal shield/results_intersample/runs/hard_cbvf_seed_123/training_actions.jsonl' \
  --qp-diagnostics-jsonl '/home/mars/Desktop/conformal shield/results_intersample/runs/hard_cbvf_seed_123/qp_feasibility_diagnostics.jsonl' \
  --results-json '/home/mars/Desktop/conformal shield/results_intersample/runs/hard_cbvf_seed_123/training_and_evaluation_results.json' \
  --plot-dir '/home/mars/Desktop/conformal shield/results_intersample/runs/hard_cbvf_seed_123/plots'
```

## hard_cbvf_seed_0

```bash
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
  --integration-substeps 10 \
  --intersample-protection \
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
  --speed-min 0.0 \
  --speed 0.6 \
  --beta-u 0.6 \
  --cbvf-model-speed-min -0.2 \
  --cbvf-model-speed 0.6 \
  --cbvf-model-beta-u 0.8 \
  --training-shield-method cbvf \
  --seed 0 \
  --checkpoint-dir '/home/mars/Desktop/conformal shield/results_intersample/runs/hard_cbvf_seed_0' \
  --dense-safety-cost \
  --no-tqdm \
  --recompute-cbvf \
  --terminate-on-collision \
  --cache-dir '/home/mars/Desktop/conformal shield/cbvf_cache_intersample' \
  --odp-root '/home/mars/Desktop/conformal shield' \
  --solver-file '/home/mars/Desktop/conformal shield/solver_cbvf.py' \
  --eval-jsonl '/home/mars/Desktop/conformal shield/results_intersample/runs/hard_cbvf_seed_0/training_eval_metrics.jsonl' \
  --action-log-jsonl '/home/mars/Desktop/conformal shield/results_intersample/runs/hard_cbvf_seed_0/training_actions.jsonl' \
  --qp-diagnostics-jsonl '/home/mars/Desktop/conformal shield/results_intersample/runs/hard_cbvf_seed_0/qp_feasibility_diagnostics.jsonl' \
  --results-json '/home/mars/Desktop/conformal shield/results_intersample/runs/hard_cbvf_seed_0/training_and_evaluation_results.json' \
  --plot-dir '/home/mars/Desktop/conformal shield/results_intersample/runs/hard_cbvf_seed_0/plots'
```

## hard_cbvf_seed_99

```bash
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
  --integration-substeps 10 \
  --intersample-protection \
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
  --speed-min 0.0 \
  --speed 0.6 \
  --beta-u 0.6 \
  --cbvf-model-speed-min -0.2 \
  --cbvf-model-speed 0.6 \
  --cbvf-model-beta-u 0.8 \
  --training-shield-method cbvf \
  --seed 99 \
  --checkpoint-dir '/home/mars/Desktop/conformal shield/results_intersample/runs/hard_cbvf_seed_99' \
  --dense-safety-cost \
  --no-tqdm \
  --recompute-cbvf \
  --terminate-on-collision \
  --cache-dir '/home/mars/Desktop/conformal shield/cbvf_cache_intersample' \
  --odp-root '/home/mars/Desktop/conformal shield' \
  --solver-file '/home/mars/Desktop/conformal shield/solver_cbvf.py' \
  --eval-jsonl '/home/mars/Desktop/conformal shield/results_intersample/runs/hard_cbvf_seed_99/training_eval_metrics.jsonl' \
  --action-log-jsonl '/home/mars/Desktop/conformal shield/results_intersample/runs/hard_cbvf_seed_99/training_actions.jsonl' \
  --qp-diagnostics-jsonl '/home/mars/Desktop/conformal shield/results_intersample/runs/hard_cbvf_seed_99/qp_feasibility_diagnostics.jsonl' \
  --results-json '/home/mars/Desktop/conformal shield/results_intersample/runs/hard_cbvf_seed_99/training_and_evaluation_results.json' \
  --plot-dir '/home/mars/Desktop/conformal shield/results_intersample/runs/hard_cbvf_seed_99/plots'
```

## hard_cp_seed_7

```bash
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
  --integration-substeps 10 \
  --intersample-protection \
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
  --speed-min 0.0 \
  --speed 0.6 \
  --beta-u 0.6 \
  --cbvf-model-speed-min -0.2 \
  --cbvf-model-speed 0.6 \
  --cbvf-model-beta-u 0.8 \
  --training-shield-method cp \
  --seed 7 \
  --checkpoint-dir '/home/mars/Desktop/conformal shield/results_intersample/runs/hard_cp_seed_7' \
  --dense-safety-cost \
  --no-tqdm \
  --recompute-cbvf \
  --terminate-on-collision \
  --cache-dir '/home/mars/Desktop/conformal shield/cbvf_cache_intersample' \
  --odp-root '/home/mars/Desktop/conformal shield' \
  --solver-file '/home/mars/Desktop/conformal shield/solver_cbvf.py' \
  --eval-jsonl '/home/mars/Desktop/conformal shield/results_intersample/runs/hard_cp_seed_7/training_eval_metrics.jsonl' \
  --action-log-jsonl '/home/mars/Desktop/conformal shield/results_intersample/runs/hard_cp_seed_7/training_actions.jsonl' \
  --qp-diagnostics-jsonl '/home/mars/Desktop/conformal shield/results_intersample/runs/hard_cp_seed_7/qp_feasibility_diagnostics.jsonl' \
  --results-json '/home/mars/Desktop/conformal shield/results_intersample/runs/hard_cp_seed_7/training_and_evaluation_results.json' \
  --plot-dir '/home/mars/Desktop/conformal shield/results_intersample/runs/hard_cp_seed_7/plots'
```

## hard_cp_seed_42

```bash
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
  --integration-substeps 10 \
  --intersample-protection \
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
  --speed-min 0.0 \
  --speed 0.6 \
  --beta-u 0.6 \
  --cbvf-model-speed-min -0.2 \
  --cbvf-model-speed 0.6 \
  --cbvf-model-beta-u 0.8 \
  --training-shield-method cp \
  --seed 42 \
  --checkpoint-dir '/home/mars/Desktop/conformal shield/results_intersample/runs/hard_cp_seed_42' \
  --dense-safety-cost \
  --no-tqdm \
  --recompute-cbvf \
  --terminate-on-collision \
  --cache-dir '/home/mars/Desktop/conformal shield/cbvf_cache_intersample' \
  --odp-root '/home/mars/Desktop/conformal shield' \
  --solver-file '/home/mars/Desktop/conformal shield/solver_cbvf.py' \
  --eval-jsonl '/home/mars/Desktop/conformal shield/results_intersample/runs/hard_cp_seed_42/training_eval_metrics.jsonl' \
  --action-log-jsonl '/home/mars/Desktop/conformal shield/results_intersample/runs/hard_cp_seed_42/training_actions.jsonl' \
  --qp-diagnostics-jsonl '/home/mars/Desktop/conformal shield/results_intersample/runs/hard_cp_seed_42/qp_feasibility_diagnostics.jsonl' \
  --results-json '/home/mars/Desktop/conformal shield/results_intersample/runs/hard_cp_seed_42/training_and_evaluation_results.json' \
  --plot-dir '/home/mars/Desktop/conformal shield/results_intersample/runs/hard_cp_seed_42/plots'
```

## hard_cp_seed_123

```bash
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
  --integration-substeps 10 \
  --intersample-protection \
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
  --speed-min 0.0 \
  --speed 0.6 \
  --beta-u 0.6 \
  --cbvf-model-speed-min -0.2 \
  --cbvf-model-speed 0.6 \
  --cbvf-model-beta-u 0.8 \
  --training-shield-method cp \
  --seed 123 \
  --checkpoint-dir '/home/mars/Desktop/conformal shield/results_intersample/runs/hard_cp_seed_123' \
  --dense-safety-cost \
  --no-tqdm \
  --recompute-cbvf \
  --terminate-on-collision \
  --cache-dir '/home/mars/Desktop/conformal shield/cbvf_cache_intersample' \
  --odp-root '/home/mars/Desktop/conformal shield' \
  --solver-file '/home/mars/Desktop/conformal shield/solver_cbvf.py' \
  --eval-jsonl '/home/mars/Desktop/conformal shield/results_intersample/runs/hard_cp_seed_123/training_eval_metrics.jsonl' \
  --action-log-jsonl '/home/mars/Desktop/conformal shield/results_intersample/runs/hard_cp_seed_123/training_actions.jsonl' \
  --qp-diagnostics-jsonl '/home/mars/Desktop/conformal shield/results_intersample/runs/hard_cp_seed_123/qp_feasibility_diagnostics.jsonl' \
  --results-json '/home/mars/Desktop/conformal shield/results_intersample/runs/hard_cp_seed_123/training_and_evaluation_results.json' \
  --plot-dir '/home/mars/Desktop/conformal shield/results_intersample/runs/hard_cp_seed_123/plots'
```

## hard_cp_seed_0

```bash
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
  --integration-substeps 10 \
  --intersample-protection \
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
  --speed-min 0.0 \
  --speed 0.6 \
  --beta-u 0.6 \
  --cbvf-model-speed-min -0.2 \
  --cbvf-model-speed 0.6 \
  --cbvf-model-beta-u 0.8 \
  --training-shield-method cp \
  --seed 0 \
  --checkpoint-dir '/home/mars/Desktop/conformal shield/results_intersample/runs/hard_cp_seed_0' \
  --dense-safety-cost \
  --no-tqdm \
  --recompute-cbvf \
  --terminate-on-collision \
  --cache-dir '/home/mars/Desktop/conformal shield/cbvf_cache_intersample' \
  --odp-root '/home/mars/Desktop/conformal shield' \
  --solver-file '/home/mars/Desktop/conformal shield/solver_cbvf.py' \
  --eval-jsonl '/home/mars/Desktop/conformal shield/results_intersample/runs/hard_cp_seed_0/training_eval_metrics.jsonl' \
  --action-log-jsonl '/home/mars/Desktop/conformal shield/results_intersample/runs/hard_cp_seed_0/training_actions.jsonl' \
  --qp-diagnostics-jsonl '/home/mars/Desktop/conformal shield/results_intersample/runs/hard_cp_seed_0/qp_feasibility_diagnostics.jsonl' \
  --results-json '/home/mars/Desktop/conformal shield/results_intersample/runs/hard_cp_seed_0/training_and_evaluation_results.json' \
  --plot-dir '/home/mars/Desktop/conformal shield/results_intersample/runs/hard_cp_seed_0/plots'
```

## hard_cp_seed_99

```bash
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
  --integration-substeps 10 \
  --intersample-protection \
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
  --speed-min 0.0 \
  --speed 0.6 \
  --beta-u 0.6 \
  --cbvf-model-speed-min -0.2 \
  --cbvf-model-speed 0.6 \
  --cbvf-model-beta-u 0.8 \
  --training-shield-method cp \
  --seed 99 \
  --checkpoint-dir '/home/mars/Desktop/conformal shield/results_intersample/runs/hard_cp_seed_99' \
  --dense-safety-cost \
  --no-tqdm \
  --recompute-cbvf \
  --terminate-on-collision \
  --cache-dir '/home/mars/Desktop/conformal shield/cbvf_cache_intersample' \
  --odp-root '/home/mars/Desktop/conformal shield' \
  --solver-file '/home/mars/Desktop/conformal shield/solver_cbvf.py' \
  --eval-jsonl '/home/mars/Desktop/conformal shield/results_intersample/runs/hard_cp_seed_99/training_eval_metrics.jsonl' \
  --action-log-jsonl '/home/mars/Desktop/conformal shield/results_intersample/runs/hard_cp_seed_99/training_actions.jsonl' \
  --qp-diagnostics-jsonl '/home/mars/Desktop/conformal shield/results_intersample/runs/hard_cp_seed_99/qp_feasibility_diagnostics.jsonl' \
  --results-json '/home/mars/Desktop/conformal shield/results_intersample/runs/hard_cp_seed_99/training_and_evaluation_results.json' \
  --plot-dir '/home/mars/Desktop/conformal shield/results_intersample/runs/hard_cp_seed_99/plots'
```

## Fresh held-out coverage commands

```bash
/home/mars/miniconda3/envs/odp/bin/python3.8 experiments/evaluate_reference_coverage.py \
  '/home/mars/Desktop/conformal shield/results_intersample/runs/aligned_cp_seed_7' \
  --episodes 100 \
  --fresh-calibration \
  --output '/home/mars/Desktop/conformal shield/results_intersample/runs/aligned_cp_seed_7/heldout_reference_coverage_fresh_intersample.json'
```

```bash
/home/mars/miniconda3/envs/odp/bin/python3.8 experiments/evaluate_reference_coverage.py \
  '/home/mars/Desktop/conformal shield/results_intersample/runs/aligned_cp_seed_42' \
  --episodes 100 \
  --fresh-calibration \
  --output '/home/mars/Desktop/conformal shield/results_intersample/runs/aligned_cp_seed_42/heldout_reference_coverage_fresh_intersample.json'
```

```bash
/home/mars/miniconda3/envs/odp/bin/python3.8 experiments/evaluate_reference_coverage.py \
  '/home/mars/Desktop/conformal shield/results_intersample/runs/aligned_cp_seed_123' \
  --episodes 100 \
  --fresh-calibration \
  --output '/home/mars/Desktop/conformal shield/results_intersample/runs/aligned_cp_seed_123/heldout_reference_coverage_fresh_intersample.json'
```

```bash
/home/mars/miniconda3/envs/odp/bin/python3.8 experiments/evaluate_reference_coverage.py \
  '/home/mars/Desktop/conformal shield/results_intersample/runs/aligned_cp_seed_0' \
  --episodes 100 \
  --fresh-calibration \
  --output '/home/mars/Desktop/conformal shield/results_intersample/runs/aligned_cp_seed_0/heldout_reference_coverage_fresh_intersample.json'
```

```bash
/home/mars/miniconda3/envs/odp/bin/python3.8 experiments/evaluate_reference_coverage.py \
  '/home/mars/Desktop/conformal shield/results_intersample/runs/aligned_cp_seed_99' \
  --episodes 100 \
  --fresh-calibration \
  --output '/home/mars/Desktop/conformal shield/results_intersample/runs/aligned_cp_seed_99/heldout_reference_coverage_fresh_intersample.json'
```

```bash
/home/mars/miniconda3/envs/odp/bin/python3.8 experiments/evaluate_reference_coverage.py \
  '/home/mars/Desktop/conformal shield/results_intersample/runs/easy_cp_seed_7' \
  --episodes 100 \
  --fresh-calibration \
  --output '/home/mars/Desktop/conformal shield/results_intersample/runs/easy_cp_seed_7/heldout_reference_coverage_fresh_intersample.json'
```

```bash
/home/mars/miniconda3/envs/odp/bin/python3.8 experiments/evaluate_reference_coverage.py \
  '/home/mars/Desktop/conformal shield/results_intersample/runs/easy_cp_seed_42' \
  --episodes 100 \
  --fresh-calibration \
  --output '/home/mars/Desktop/conformal shield/results_intersample/runs/easy_cp_seed_42/heldout_reference_coverage_fresh_intersample.json'
```

```bash
/home/mars/miniconda3/envs/odp/bin/python3.8 experiments/evaluate_reference_coverage.py \
  '/home/mars/Desktop/conformal shield/results_intersample/runs/easy_cp_seed_123' \
  --episodes 100 \
  --fresh-calibration \
  --output '/home/mars/Desktop/conformal shield/results_intersample/runs/easy_cp_seed_123/heldout_reference_coverage_fresh_intersample.json'
```

```bash
/home/mars/miniconda3/envs/odp/bin/python3.8 experiments/evaluate_reference_coverage.py \
  '/home/mars/Desktop/conformal shield/results_intersample/runs/easy_cp_seed_0' \
  --episodes 100 \
  --fresh-calibration \
  --output '/home/mars/Desktop/conformal shield/results_intersample/runs/easy_cp_seed_0/heldout_reference_coverage_fresh_intersample.json'
```

```bash
/home/mars/miniconda3/envs/odp/bin/python3.8 experiments/evaluate_reference_coverage.py \
  '/home/mars/Desktop/conformal shield/results_intersample/runs/easy_cp_seed_99' \
  --episodes 100 \
  --fresh-calibration \
  --output '/home/mars/Desktop/conformal shield/results_intersample/runs/easy_cp_seed_99/heldout_reference_coverage_fresh_intersample.json'
```

```bash
/home/mars/miniconda3/envs/odp/bin/python3.8 experiments/evaluate_reference_coverage.py \
  '/home/mars/Desktop/conformal shield/results_intersample/runs/hard_cp_seed_7' \
  --episodes 100 \
  --fresh-calibration \
  --output '/home/mars/Desktop/conformal shield/results_intersample/runs/hard_cp_seed_7/heldout_reference_coverage_fresh_intersample.json'
```

```bash
/home/mars/miniconda3/envs/odp/bin/python3.8 experiments/evaluate_reference_coverage.py \
  '/home/mars/Desktop/conformal shield/results_intersample/runs/hard_cp_seed_42' \
  --episodes 100 \
  --fresh-calibration \
  --output '/home/mars/Desktop/conformal shield/results_intersample/runs/hard_cp_seed_42/heldout_reference_coverage_fresh_intersample.json'
```

```bash
/home/mars/miniconda3/envs/odp/bin/python3.8 experiments/evaluate_reference_coverage.py \
  '/home/mars/Desktop/conformal shield/results_intersample/runs/hard_cp_seed_123' \
  --episodes 100 \
  --fresh-calibration \
  --output '/home/mars/Desktop/conformal shield/results_intersample/runs/hard_cp_seed_123/heldout_reference_coverage_fresh_intersample.json'
```

```bash
/home/mars/miniconda3/envs/odp/bin/python3.8 experiments/evaluate_reference_coverage.py \
  '/home/mars/Desktop/conformal shield/results_intersample/runs/hard_cp_seed_0' \
  --episodes 100 \
  --fresh-calibration \
  --output '/home/mars/Desktop/conformal shield/results_intersample/runs/hard_cp_seed_0/heldout_reference_coverage_fresh_intersample.json'
```

```bash
/home/mars/miniconda3/envs/odp/bin/python3.8 experiments/evaluate_reference_coverage.py \
  '/home/mars/Desktop/conformal shield/results_intersample/runs/hard_cp_seed_99' \
  --episodes 100 \
  --fresh-calibration \
  --output '/home/mars/Desktop/conformal shield/results_intersample/runs/hard_cp_seed_99/heldout_reference_coverage_fresh_intersample.json'
```

## Aggregation and audit

These commands are intentionally not run unless all 45 strict runs complete.

```bash
/home/mars/miniconda3/envs/odp/bin/python3.8 experiments/aggregate_full_suite_intersample.py
```

```bash
/home/mars/miniconda3/envs/odp/bin/python3.8 experiments/audit_full_suite_intersample.py
```

```bash
/home/mars/miniconda3/envs/odp/bin/python3.8 experiments/build_scientific_report_intersample.py
```
