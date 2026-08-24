#!/usr/bin/env bash
set -euo pipefail

: "${MONDRIAN_ODP_ROOT:?Set MONDRIAN_ODP_ROOT to the directory containing the odp package}"
: "${MONDRIAN_SOLVER_FILE:?Set MONDRIAN_SOLVER_FILE to the local CBVF HJSolver file}"

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
  --odp-root "$MONDRIAN_ODP_ROOT" \
  --solver-file "$MONDRIAN_SOLVER_FILE" \
  --checkpoint-dir rl_checkpoints/reach_avoid_mondrian_shielded_from_scratch
