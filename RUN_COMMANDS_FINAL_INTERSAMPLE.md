# Reproduction status: no final experiment was authorized

The strict task-horizon feasibility gate failed before configuration freeze.
Consequently there are no scientifically valid commands for final training,
final evaluation, aggregation, or figures.  Running the 45-job suite would
violate the required pre-training gate.

The exact implemented entry point is `train/train_sac_lag.py`.  Design audits
must use `--intersample-protection --feasibility-only`, dimensional calibration
(now the code default), `--shield-release-margin 0`, and must not use
`--allow-infeasible-cp-diagnostic-run`.

The decisive raw manifests are:

- `results_intersample/design_sweeps/hard_dt002_long_two_regions_n40_seed31217_corrected2/pretraining_mondrian_calibration.json`
- `results_intersample/design_sweeps/hard_dt002_long_loaded_floor020_startm2m3_seed31220/pretraining_mondrian_calibration.json`

The complete literal task-horizon commands, including every numeric argument,
are preserved in the corresponding process logs and shell invocations recorded
with these result directories.  Final commands must only be generated after a
new theoretical feasibility mechanism passes the same exhaustive gate.
# Reachability gate added after the hard-mismatch infeasibility diagnosis

The current frozen configuration fails this gate, so commands below the gate
must not be run unless a newly calibrated configuration passes it.

```bash
PYTHONPATH=. MPLCONFIGDIR=/tmp/mpl-reach \
  /home/mars/miniconda3/envs/odp/bin/python3.8 \
  diagnostics/find_reachable_infeasible_node.py \
  --table cbvf_cache_reachability_frozen_v1/cbvf_table_vmin-0.200_vmax0.600_bu0.800_g0.100_cbvfdt0.100_lb17.0_nx31_ny31_nth25_targetsigned_distance_clip1.00_scale0.00.npz \
  --manifest results_intersample/reachability/frozen_v1_calibration_seed41201_cached/pretraining_mondrian_calibration.json \
  --output results_intersample/reachability/frozen_v1_geometric_B_envelope_intersection_witness.json \
  --initial-B-floor 0.2404419557358216 --dt 0.002 --horizon 6000 \
  --gamma 0.1 --activation 0.3
```
