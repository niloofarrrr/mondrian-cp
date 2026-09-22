#!/usr/bin/env bash
set -euo pipefail

name="${1:?pilot name required}"
seed="${2:?pilot seed required}"
condition="${3:?condition required: aligned, easy, or hard}"
calibration_manifest="${4:?qualified calibration manifest required}"
episodes="${5:-100}"
activation="${6:-0.03}"
initial_x="${7:--1.45}"
model_beta="${8:-0.8}"

case "$condition" in
  aligned) speed_min=-0.2; beta_u=0.8 ;;
  easy)    speed_min=-0.1; beta_u=0.7 ;;
  hard)    speed_min=0.0;  beta_u=0.6 ;;
  *) echo "unknown condition: $condition" >&2; exit 2 ;;
esac

out="results_intersample_v2/pilots/${name}"
mkdir -p "$out"

/home/mars/miniconda3/envs/odp/bin/python3.8 train/train_sac_lag.py \
  --algorithm sac_lag --max-steps 20 --start-training 20 \
  --warmup-policy reference --reference-proposal-steps 20 \
  --residual-reference-scale 0.02 --batch-size 256 --cost-limit 1 \
  --goal-bonus 750 --collision-terminal-penalty 2500 --unsafe-penalty 25 \
  --eval-interval 20 --save-interval 20 --eval-episodes "$episodes" \
  --periodic-eval-episodes 20 --n-calib 40 --delta 0.05 \
  --mondrian-clearance-edges 0.75 --dt 0.001 --integration-substeps 2 \
  --intersample-protection --horizon 12000 --cbvf-nx 31 --cbvf-ny 31 \
  --cbvf-nth 25 --cbvf-dt 0.10 --cbvf-terminal-guard 5 --gamma 0.25 \
  --target-shape signed_distance --cbvf-activate-margin "$activation" \
  --cp-activate-margin "$activation" --shield-release-margin 0 \
  --auto-certified-initial-B-floor --initial-x "$initial_x" --initial-y -3.0 \
  --initial-theta 0.7853981633974483 --start-jitter-xy 0.1 \
  --start-jitter-theta 0.1 --speed-min "$speed_min" --speed 0.6 \
  --beta-u "$beta_u" --cbvf-model-speed-min -0.2 --cbvf-model-speed 0.6 \
  --cbvf-model-beta-u "$model_beta" --training-shield-method cp --seed "$seed" \
  --epsilon-inter 0 --cache-dir cbvf_cache_intersample_design/gamma_025_fresh \
  --reuse-cbvf --calibration-manifest-in "$calibration_manifest" \
  --reuse-zero-offline-audit --no-tqdm --checkpoint-dir "$out" \
  --eval-jsonl "$out/training_eval_metrics.jsonl" \
  --action-log-jsonl "$out/training_actions.jsonl" \
  --qp-diagnostics-jsonl "$out/qp_feasibility_diagnostics.jsonl" \
  --results-json "$out/training_and_evaluation_results.json" \
  --plot-dir "$out/plots" --odp-root "$(pwd)" --solver-file "$(pwd)/solver_cbvf.py"
