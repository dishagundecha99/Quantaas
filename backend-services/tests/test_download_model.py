import unittest
from flask import Flask, session, request, jsonify
from flask.testing import FlaskClient
from unittest.mock import patch, MagicMock
from backend_server import download_model

class DownloadModelTestCase(unittest.TestCase):
    def setUp(self):
        self.app = Flask(__name__)
        self.app.testing = True

    def test_download_model_without_user_id(self):
        with self.app.test_request_context('/download_model?expid=123'):
            response = download_model()
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.get_json(), {'message': 'failed to get model details'})

    @patch('backend_server.download_model.clean_up')
    @patch('backend_server.download_model.mongo.get_job_details')
    @patch('backend_server.download_model.minio.download_model_file')
    @patch('backend_server.download_model.send_file')
    def test_download_model_with_valid_exp_id(self, mock_send_file, mock_download_model_file, mock_get_job_details, mock_clean_up):
        with self.app.test_request_context('/download_model?expid=123&user-id=456'):
            session['user-id'] = '456'
            request.args = {'expid': '123'}
            mock_get_job_details.return_value = {
                'exp_id': '123',
                'model_filename': 'model.h5',
                'minio_bucket': 'bucket'
            }
            mock_send_file.return_value = 'file_content'

            response = download_model()

            mock_clean_up.assert_called_once()
            mock_get_job_details.assert_called_once_with('456', '123', '123')
            mock_download_model_file.assert_called_once_with('bucket', '123/model.h5', 'backend_server/model.h5')
            mock_send_file.assert_called_once_with('backend_server/model.h5', as_attachment=True)
            self.assertEqual(response, 'file_content')

if __name__ == '__main__':
    unittest.main()