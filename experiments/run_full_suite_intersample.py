#!/usr/bin/env python3
"""Isolated resumable driver for the inter-sample experiment suite."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments import run_full_suite as runner

if not any(arg == "--manifest" or arg.startswith("--manifest=") for arg in sys.argv[1:]):
    sys.argv.extend([
        "--manifest", str(ROOT / "experiments" / "full_suite_intersample.json")
    ])

raise SystemExit(runner.main())
