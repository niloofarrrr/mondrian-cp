"""Check freeze ordering and no retuning after a failed final study, with mocks."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'experiments'))
from freeze_v5_configuration import CORE


def check(pilot_passed, final_passed):
    spec = importlib.util.spec_from_file_location('continuation_test', ROOT / 'experiments/continue_v5_study.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    calls = []
    with tempfile.TemporaryDirectory() as temporary:
        module.RESULTS = Path(temporary); module.OUT = Path(temporary) / 'workflow'
        module.CANDIDATES = ['candidate']; module.EXISTING = dict(aligned='aligned', easy='easy')
        for name in ['candidate', 'aligned', 'easy']:
            folder = Path(temporary) / name; folder.mkdir()
            (folder / 'qualification_status.json').write_text(json.dumps({'passed': pilot_passed}))
            (folder / 'batch_configuration.json').write_text(json.dumps({'configuration': {key: 1 for key in CORE}}))
        module.run_replica = lambda name, seed, cfg, condition: (Path(temporary) / name, True)
        def command(args, log_name):
            calls.append(args[1])
            frozen = Path(temporary) / 'frozen_study'
            if args[1].endswith('freeze_v5_configuration.py'):
                frozen.mkdir(); (frozen / 'FREEZE.json').write_text('{}')
            if args[1].endswith('audit_full_suite.py'):
                out = frozen / 'final_suite/aggregate'; out.mkdir(parents=True)
                (out / 'acceptance_audit.json').write_text(json.dumps({'passed': final_passed}))
            return 0
        module.command = command
        code = module.main()
        if not pilot_passed:
            assert code == 2 and not calls
            assert not (Path(temporary) / 'frozen_study').exists()
        else:
            assert code == (0 if final_passed else 1)
            assert calls == ['experiments/freeze_v5_configuration.py', 'experiments/run_frozen_suite.py',
                             'experiments/aggregate_full_suite.py', 'experiments/audit_full_suite.py',
                             'experiments/build_v3_paper_report.py']
            status = json.loads((module.OUT / 'status.json').read_text())
            assert status['final_data_used_for_retuning'] is False


check(False, False)
check(True, True)
check(True, False)
print('PASS: rejected designs never freeze; accepted designs freeze first; failed final data never trigger retuning')
