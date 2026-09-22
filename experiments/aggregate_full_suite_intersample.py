#!/usr/bin/env python3
"""Generate isolated, suffixed artifacts for a completed inter-sample suite."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments import aggregate_full_suite as aggregate

aggregate.STATE = ROOT / "results_intersample" / "final_suite" / "run_state.json"
aggregate.OUT = ROOT / "results_intersample" / "final_suite" / "aggregate"
aggregate.OUT.mkdir(parents=True, exist_ok=True)
aggregate.main()

for path in list(aggregate.OUT.iterdir()):
    if path.is_file() and "_intersample" not in path.stem:
        path.rename(path.with_name(path.stem + "_intersample" + path.suffix))
