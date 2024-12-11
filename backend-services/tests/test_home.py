import unittest
from flask import Flask, session
from flask.testing import FlaskClient
from unittest.mock import patch
from backend_server import app

class HomeTestCase(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    def test_index_authenticated(self):
        with self.app as client:
            with client.session_transaction() as sess:
                sess['user_id'] = '123'

            response = client.get('/')
            data = response.get_json()

            self.assertEqual(response.status_code, 200)
            self.assertEqual(data['user_id'], '123')
            self.assertEqual(data['logged_in'], True)

    def test_index_unauthenticated(self):
        with self.app.test_client() as client:
            response = client.get('/')

            self.assertEqual(response.status_code, 200)
            self.assertIn('<a class="button" href="/auth">Google Login</a>', response.get_data(as_text=True))

if __name__ == '__main__':
    unittest.main()