#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export MPLBACKEND=Agg
export MPLCONFIGDIR="${TMPDIR:-/tmp}/mondrian-matplotlib-intersample"
mkdir -p "$MPLCONFIGDIR" results_intersample/final_suite/logs
exec /home/mars/miniconda3/envs/odp/bin/python3.8 \
  experiments/run_full_suite_intersample.py "$@"
