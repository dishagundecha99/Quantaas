import unittest
from backend_server.prev_runs import best_run_exp

class PrevRunsTestCase(unittest.TestCase):
    def test_best_run_exp(self):
        runs = [
            {'exp_id': 'exp1', 'accuracy': '0.8'},
            {'exp_id': 'exp2', 'accuracy': '0.9'},
            {'exp_id': 'exp3', 'accuracy': '0.7'}
        ]
        result = best_run_exp('experiment', runs)
        self.assertEqual(result, 'exp2')

if __name__ == '__main__':
    unittest.main()