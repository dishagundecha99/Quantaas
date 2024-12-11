import unittest
from flask import Flask
from flask.testing import FlaskClient
from unittest.mock import patch
from backend_server import app, mongo

class CurrentRunTestCase(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    def test_current_run_no_user_id(self):
        response = self.app.get('/current-run')
        data = response.get_json()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(data['message'], 'No running jobs found')

    @patch('backend_server.mongo.get_all_runs')
    def test_current_run_with_user_id(self, mock_get_all_runs):
        mock_get_all_runs.return_value = {
            'experiment1': [
                {'training': True, 'epoch': 10},
                {'training': False, 'epoch': 5}
            ],
            'experiment2': [
                {'training': False, 'epoch': 3},
                {'training': False, 'epoch': 7}
            ]
        }

        response = self.app.get('/current-run?user-id=123')
        data = response.get_json()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(data['message'], 'experiment1 is currently running')
        self.assertEqual(data['current-run'], [
            {'training': True, 'epoch': 10},
            {'training': False, 'epoch': 5}
        ])

if __name__ == '__main__':
    unittest.main()