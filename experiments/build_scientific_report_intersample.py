#!/usr/bin/env python3
"""Build the isolated scientific report after successful inter-sample aggregation."""

from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments import build_scientific_report as report

report.RESULTS = ROOT / "results_intersample" / "final_suite"
report.AGG = report.RESULTS / "aggregate"
report.REPORT = report.RESULTS / "SCIENTIFIC_FINAL_REPORT_intersample.md"
temporary = []
for suffixed, plain in (
    (report.AGG / "final_metrics_intersample.csv", report.AGG / "final_metrics.csv"),
    (report.AGG / "acceptance_audit_intersample.json", report.AGG / "acceptance_audit.json"),
):
    if not suffixed.is_file():
        raise SystemExit(f"Missing required inter-sample artifact: {suffixed}")
    shutil.copyfile(suffixed, plain)
    temporary.append(plain)
try:
    report.main()
finally:
    for path in temporary:
        path.unlink(missing_ok=True)
