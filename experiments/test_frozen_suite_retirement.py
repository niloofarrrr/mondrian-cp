"""Verify whole-study retirement preserves evidence and stops live work only."""
import io
import unittest
from unittest.mock import Mock
from run_frozen_suite import retire_failed_study, coverage_failure


class RetirementTests(unittest.TestCase):
    def test_coverage_gate_rejects_failed_or_incomplete_evidence(self):
        for value, episodes in [(0.94, 100), (float('nan'), 100), (1.0, 99)]:
            self.assertIsNotNone(coverage_failure({'episodes': episodes, 'empirical_simultaneous_coverage': value}))
        self.assertIsNone(coverage_failure({'episodes': 100, 'empirical_simultaneous_coverage': 0.95}))

    def test_failure_retires_all_work_without_relabeling_success(self):
        state = {'runs': {'bad': {'status': 'failed'}, 'done': {'status': 'completed'},
                          'live': {'status': 'running'}, 'next': {'status': 'pending'}}}
        process = Mock(); process.poll.return_value = None
        log = io.StringIO()
        self.assertTrue(retire_failed_study(state, {'live': (process, log, 'training')}))
        process.terminate.assert_called_once_with()
        self.assertEqual(state['runs']['done']['status'], 'completed')
        self.assertEqual(state['runs']['bad']['status'], 'failed')
        self.assertEqual(state['runs']['next']['status'], 'pending')
        self.assertEqual(state['whole_study_rejection']['trigger_runs'], ['bad'])
        self.assertFalse(state['whole_study_rejection']['final_data_used_for_parameter_selection'])
        self.assertTrue(retire_failed_study(state, {'live': (process, log, 'training')}))
        process.terminate.assert_called_once_with()

    def test_successful_progress_does_not_terminate_jobs(self):
        state = {'runs': {'done': {'status': 'completed'}, 'live': {'status': 'running'}}}
        process = Mock()
        self.assertFalse(retire_failed_study(state, {'live': (process, io.StringIO(), 'training')}))
        process.terminate.assert_not_called()


if __name__ == '__main__':
    unittest.main()
