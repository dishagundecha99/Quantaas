import unittest
from flask import Flask
from flask.testing import FlaskClient
from unittest.mock import patch
from backend_server.submit_job import app, submit_job_paylod_validator
from werkzeug.datastructures import FileStorage

class SubmitJobTestCase(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    def test_submit_training_job_valid_payload(self):
        with patch('backend_server.submit_job.minio.upload_dataset') as mock_upload_dataset, \
             patch('backend_server.submit_job.kafka.push_to_topic') as mock_push_to_topic, \
             patch('backend_server.submit_job.mongo.record_train_meta_data') as mock_record_train_meta_data:
            
            # Create a valid payload
            payload = {
                'user-id': '123',
                'exp_name': 'experiment',
                'task_type': 'classification',
                'model_name': 'model',
                'hyperparams': '{"param1": [1, 2], "param2": [3, 4]}'
            }
            train_file = FileStorage(filename='train.csv')
            test_file = FileStorage(filename='test.csv')
            payload['train_user_file'] = train_file
            payload['test_user_file'] = test_file

            # Send a POST request with the payload
            response = self.app.post('/submit-job', data=payload, content_type='multipart/form-data')

            # Assert that the response is successful
            self.assertEqual(response.status_code, 200)

            # Assert that the necessary functions were called with the correct arguments
            mock_upload_dataset.assert_called_with('123', train_file)
            mock_upload_dataset.assert_called_with('123', test_file)
            mock_push_to_topic.assert_called()
            mock_record_train_meta_data.assert_called()

    def test_submit_training_job_invalid_payload(self):
        # Create an invalid payload
        payload = {
            'user-id': '123',
            'exp_name': 'experiment',
            'task_type': 'classification',
            'model_name': 'model',
            'hyperparams': '{"param1": [1, 2], "param2": [3, 4]}'
        }

        # Send a POST request with the invalid payload
        response = self.app.post('/submit-job', data=payload, content_type='multipart/form-data')

        # Assert that the response is a bad request
        self.assertEqual(response.status_code, 400)

if __name__ == '__main__':
    unittest.main()