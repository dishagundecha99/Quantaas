import unittest
from unittest.mock import patch, MagicMock
from werkzeug.datastructures import FileStorage
from backend_server import minio_utils

class MinioUtilsTestCase(unittest.TestCase):
    @patch('backend_server.minio_utils.Minio')
    def test_get_user_bucket_name(self, mock_minio):
        user_id = 'test@example.com'
        bucket_name = minio_utils.get_user_bucket_name(user_id)
        self.assertEqual(bucket_name, 'test.example.com')

    @patch('backend_server.minio_utils.minio_client')
    def test_prepare_user_bucket(self, mock_minio_client):
        user_bucket_name = 'test.example.com'
        mock_minio_client.bucket_exists.return_value = False

        minio_utils.prepare_user_bucket(user_bucket_name)

        mock_minio_client.make_bucket.assert_called_once_with(user_bucket_name)

    @patch('backend_server.minio_utils.minio_client')
    def test_upload_dataset(self, mock_minio_client):
        user_id = 'test@example.com'
        file = FileStorage(filename='test.txt', stream=MagicMock(), content_type='text/plain')
        user_bucket_name = 'test.example.com'
        file_size = 100

        minio_utils.prepare_user_bucket = MagicMock()
        minio_utils.upload_dataset(user_id, file)

        minio_utils.prepare_user_bucket.assert_called_once_with(user_bucket_name)
        mock_minio_client.put_object.assert_called_once_with(user_bucket_name, 'datasets/test.txt', file, file_size)

    @patch('backend_server.minio_utils.minio_client')
    def test_download_model_file(self, mock_minio_client):
        bucket = 'test.example.com'
        object_name = 'model.h5'
        file_path = '/path/to/model.h5'

        minio_utils.download_model_file(bucket, object_name, file_path)

        mock_minio_client.fget_object.assert_called_once_with(bucket, object_name, file_path)

if __name__ == '__main__':
    unittest.main()