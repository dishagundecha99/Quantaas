import unittest
from flask import Flask
from flask.testing import FlaskClient
from unittest.mock import patch
from auth import app

class AuthTestCase(unittest.TestCase):
    def setUp(self):
        self.app = auth.app.test_client()
        self.app.testing = True

    def test_auth_endpoint(self):
        response = self.app.get('/auth?user-id=123')
        data = response.get_json()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(data['user_id'], '123')
        self.assertEqual(data['logged_in'], True)

    @patch('backend_server.auth.requests.get')
    def test_login_endpoint(self, mock_get):
        mock_get.return_value.json.return_value = {
            "authorization_endpoint": "https://accounts.google.com/o/oauth2/auth",
            "token_endpoint": "https://oauth2.googleapis.com/token",
            "userinfo_endpoint": "https://openidconnect.googleapis.com/v1/userinfo"
        }

        response = self.app.get('/login')
        self.assertEqual(response.status_code, 302)
        self.assertIn('https://accounts.google.com/o/oauth2/auth', response.headers['Location'])

    @patch('backend_server.auth.requests.post')
    @patch('backend_server.auth.requests.get')
    def test_callback_endpoint(self, mock_get, mock_post):
        mock_get.return_value.json.return_value = {
            "email_verified": True,
            "email": "test@example.com",
            "given_name": "John"
        }
        mock_post.return_value.json.return_value = {
            "access_token": "ACCESS_TOKEN",
            "refresh_token": "REFRESH_TOKEN",
            "expires_in": 3600,
            "token_type": "Bearer"
        }

        response = self.app.get('/login/callback?code=AUTHORIZATION_CODE')
        self.assertEqual(response.status_code, 302)
        self.assertEqual(response.headers['Location'], 'http://localhost/')

if __name__ == '__main__':
    unittest.main()