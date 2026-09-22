"""Exercise rejection, independent qualification, and full-study seed retirement."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import autonomous_case_study as workflow


class WorkflowTests(unittest.TestCase):
    def run_scenario(self, reject_first=False, fail_first_final=False):
        with tempfile.TemporaryDirectory(prefix='autonomous-workflow-test-') as temporary:
            root = Path(temporary); out = root/'workflow'; out.mkdir()
            candidates = [dict(name='candidate%d'%i, seed=622000+4*i, configuration={},
                replica_seeds=[622001+4*i,622002+4*i,622003+4*i], rationale='test') for i in range(2)]
            (out/'design_plan.json').write_text(json.dumps({'candidates':candidates}))
            freezes, finals, pilots = [], [], []
            def ensure(name, seed, cfg, condition):
                pilots.append((name, condition))
                return root/name, not (reject_first and name == 'candidate0')
            def command(args, label):
                freezes.append(args)
                return 0
            def final(folder):
                finals.append(folder)
                return not (fail_first_final and len(finals) == 1)
            with patch.object(workflow, 'OUT', out), patch.object(workflow, 'BASE', root), \
                 patch.object(workflow, 'ensure_pilot', ensure), patch.object(workflow, 'command', command), \
                 patch.object(workflow, 'run_final', final):
                self.assertEqual(workflow.main(), 0)
                count = len(freezes)
                self.assertEqual(workflow.main(), 0)
                self.assertEqual(len(freezes), count, 'Restart of a completed study must not rerun it')
            return freezes, finals, pilots, json.loads((out/'progress.json').read_text())

    def test_failed_pilot_advances_without_freeze(self):
        freezes, finals, pilots, state = self.run_scenario(reject_first=True)
        self.assertEqual(len(freezes), 1)
        self.assertEqual(freezes[0].count('--pilot'), 4)
        self.assertEqual(len(finals), 1)
        self.assertEqual(len(pilots), 5)
        self.assertEqual(state['candidate_index'], 1)
        self.assertEqual(len(state['rejected']), 1)

    def test_failed_final_retires_full_seed_block(self):
        freezes, finals, pilots, state = self.run_scenario(fail_first_final=True)
        self.assertEqual(len(freezes), 2)
        blocks = [args[args.index('--final-seeds')+1:args.index('--final-seeds')+6] for args in freezes]
        self.assertFalse(set(blocks[0]) & set(blocks[1]))
        self.assertEqual(len(state['final_attempts']), 1)
        self.assertEqual(state['study_number'], 1)
        self.assertEqual(len(pilots), 8)


if __name__ == '__main__':
    unittest.main()
