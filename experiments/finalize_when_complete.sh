#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export MPLBACKEND=Agg
export MPLCONFIGDIR="${TMPDIR:-/tmp}/mondrian-matplotlib"
mkdir -p "$MPLCONFIGDIR" results/aggregate

while true; do
  status="$(python -c 'import json; print(json.load(open("results/run_state.json"))["status"])')"
  case "$status" in
    completed) break ;;
    failed|interrupted)
      echo "Suite ended with status: $status" >&2
      exit 1
      ;;
  esac
  sleep 60
done

python experiments/aggregate_full_suite.py
audit_status=0
python experiments/audit_full_suite.py || audit_status=$?
python experiments/build_scientific_report.py
exit "$audit_status"
