#!/usr/bin/env bash
set -euo pipefail

name="${1:?candidate name required}"
seed="${2:?seed required}"
activation="${3:?activation margin required}"
initial_x="${4:?initial x required}"
controller_dt="${5:-0.002}"
horizon="${6:-6000}"
model_beta="${7:-0.8}"
truth_beta="${8:-0.6}"
cache_dir="${9:-cbvf_cache_intersample_design/gamma_025_fresh}"
recompute="${10:-false}"
out="results_intersample_v2/design/${name}"
mkdir -p "$out"
if [[ "$recompute" == true ]]; then
  cbvf_mode=(--recompute-cbvf)
else
  cbvf_mode=(--reuse-cbvf)
fi

/home/mars/miniconda3/envs/odp/bin/python3.8 train/train_sac_lag.py \
  --algorithm sac_lag --max-steps 20 --start-training 20 \
  --warmup-policy reference --reference-proposal-steps 20 \
  --residual-reference-scale 0.02 --batch-size 256 --cost-limit 1 \
  --goal-bonus 750 --collision-terminal-penalty 2500 --unsafe-penalty 25 \
  --eval-interval 20 --save-interval 20 --eval-episodes 100 \
  --periodic-eval-episodes 20 --n-calib 40 --delta 0.05 \
  --mondrian-clearance-edges 0.75 --dt "$controller_dt" --integration-substeps 2 \
  --intersample-protection --horizon "$horizon" --cbvf-nx 31 --cbvf-ny 31 \
  --cbvf-nth 25 --cbvf-dt 0.10 --cbvf-terminal-guard 5 --gamma 0.25 \
  --target-shape signed_distance --cbvf-activate-margin "$activation" \
  --cp-activate-margin "$activation" --shield-release-margin 0 \
  --auto-certified-initial-B-floor --initial-x "$initial_x" --initial-y -3.0 \
  --initial-theta 0.7853981633974483 --start-jitter-xy 0.1 \
  --start-jitter-theta 0.1 --speed-min 0 --speed 0.6 --beta-u "$truth_beta" \
  --cbvf-model-speed-min -0.2 --cbvf-model-speed 0.6 \
  --cbvf-model-beta-u "$model_beta" --training-shield-method cp --seed "$seed" \
  --epsilon-inter 0 --cache-dir "$cache_dir" \
  "${cbvf_mode[@]}" \
  --no-tqdm --feasibility-only --checkpoint-dir "$out" \
  --eval-jsonl "$out/training_eval_metrics.jsonl" \
  --action-log-jsonl "$out/training_actions.jsonl" \
  --qp-diagnostics-jsonl "$out/qp_feasibility_diagnostics.jsonl" \
  --results-json "$out/training_and_evaluation_results.json" \
  --plot-dir "$out/plots" --odp-root "$(pwd)" --solver-file "$(pwd)/solver_cbvf.py"
