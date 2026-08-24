# Complete aligned-model shielded training run

Set the two local dependency paths, then run the command below. The CBVF is recomputed, the Mondrian buffers are calibrated before episode one, and all RL rollout actions are filtered.

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

The command also requires more than one effective Mondrian region and reports the near-obstacle score count and candidate-region activity. It fails rather than using an unshielded mode, a cached CBVF, post-training calibration, a silently collapsed global buffer, or an infeasible CP fallback. Gym's TimeLimit is set from `--horizon`, and all evaluation methods use identical seeded starts. The no-filter evaluation is a counterfactual diagnostic of the shield-trained actor, not a separately trained nominal baseline. The explicit zero inter-sample term is reproducible but must be replaced by an independently validated positive bound when the paper's inter-sample certificate requires it; the code does not estimate or tune that bound from evaluation outcomes.
