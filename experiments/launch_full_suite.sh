#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export MPLBACKEND=Agg
export MPLCONFIGDIR="${TMPDIR:-/tmp}/mondrian-matplotlib"
mkdir -p "$MPLCONFIGDIR" results/logs
exec /home/mars/miniconda3/envs/odp/bin/python3.8 experiments/run_full_suite.py "$@"
