#!/usr/bin/env python3
"""Strict acceptance audit for the isolated inter-sample suite."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments import audit_full_suite as audit

audit.STATE = ROOT / "results_intersample" / "final_suite" / "run_state.json"
audit.OUT_JSON = ROOT / "results_intersample" / "final_suite" / "aggregate" / "acceptance_audit_intersample.json"
audit.OUT_MD = ROOT / "results_intersample" / "final_suite" / "aggregate" / "ACCEPTANCE_AUDIT_intersample.md"
raise SystemExit(audit.main())
