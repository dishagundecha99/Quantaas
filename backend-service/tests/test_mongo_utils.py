import unittest
from unittest.mock import patch
from backend_server import mongo_utils

class MongoUtilsTestCase(unittest.TestCase):
    def setUp(self):
        self.user_id = '123'
        self.exp_name = 'experiment'
        self.exp_id = '456'
        self.train_meta_data = {
            'accuracy': 0.8,
            'training': True,
            'model_filename': 'model.pt'
        }

    @patch('backend_server.mongo_utils.mongo_collection')
    def test_create_user_doc(self, mock_collection):
        mongo_utils.create_user_doc(self.user_id)
        mock_collection.insert_one.assert_called_once_with({'user_id': self.user_id, 'runs': {}})

    @patch('backend_server.mongo_utils.mongo_collection')
    def test_find_or_create_user_doc_existing(self, mock_collection):
        mock_collection.find_one.return_value = {'user_id': self.user_id}
        mongo_utils.find_or_create_user_doc(self.user_id)
        mock_collection.find_one.assert_called_once_with({'user_id': self.user_id})
        mock_collection.insert_one.assert_not_called()

    @patch('backend_server.mongo_utils.mongo_collection')
    def test_find_or_create_user_doc_new(self, mock_collection):
        mock_collection.find_one.return_value = None
        mongo_utils.find_or_create_user_doc(self.user_id)
        mock_collection.find_one.assert_called_once_with({'user_id': self.user_id})
        mock_collection.insert_one.assert_called_once_with({'user_id': self.user_id, 'runs': {}})

    @patch('backend_server.mongo_utils.mongo_collection')
    def test_find_or_create_exp_doc_existing(self, mock_collection):
        mock_collection.find_one.return_value = {'user_id': self.user_id, 'runs': {self.exp_name: []}}
        mongo_utils.find_or_create_exp_doc(self.user_id, self.exp_name)
        mock_collection.find_one.assert_called_once_with({'user_id': self.user_id})
        mock_collection.update_one.assert_not_called()

    @patch('backend_server.mongo_utils.mongo_collection')
    def test_find_or_create_exp_doc_new(self, mock_collection):
        mock_collection.find_one.return_value = {'user_id': self.user_id, 'runs': {}}
        mongo_utils.find_or_create_exp_doc(self.user_id, self.exp_name)
        mock_collection.find_one.assert_called_once_with({'user_id': self.user_id})
        mock_collection.update_one.assert_called_once_with(
            {'user_id': self.user_id},
            {'$set': {'runs': {self.exp_name: []}}}
        )

    @patch('backend_server.mongo_utils.mongo_collection')
    def test_record_train_meta_data(self, mock_collection):
        mongo_utils.record_train_meta_data(self.user_id, self.train_meta_data, self.exp_name)
        mock_collection.update_one.assert_called_once_with(
            {'user_id': self.user_id},
            {'$push': {'runs.experiment': self.train_meta_data}}
        )

    @patch('backend_server.mongo_utils.mongo_collection')
    def test_get_all_runs_existing(self, mock_collection):
        mock_collection.find_one.return_value = {'user_id': self.user_id, 'runs': {'experiment': []}}
        runs = mongo_utils.get_all_runs(self.user_id)
        mock_collection.find_one.assert_called_once_with({'user_id': self.user_id})
        self.assertEqual(runs, {'experiment': []})

    @patch('backend_server.mongo_utils.mongo_collection')
    def test_get_all_runs_non_existing(self, mock_collection):
        mock_collection.find_one.return_value = None
        runs = mongo_utils.get_all_runs(self.user_id)
        mock_collection.find_one.assert_called_once_with({'user_id': self.user_id})
        self.assertEqual(runs, {})

    @patch('backend_server.mongo_utils.mongo_collection')
    def test_get_job_details_existing(self, mock_collection):
        exp_doc = {
            'user_id': self.user_id,
            'runs': {
                'experiment': [
                    {'exp_id': '123', 'accuracy': 0.9},
                    {'exp_id': self.exp_id, 'accuracy': 0.8}
                ]
            }
        }
        mock_collection.find_one.return_value = exp_doc
        job_details = mongo_utils.get_job_details(self.user_id, self.exp_name, self.exp_id)
        mock_collection.find_one.assert_called_once_with(
            {'user_id': self.user_id, 'runs.experiment.exp_id': self.exp_id}
        )
        self.assertEqual(job_details, {'exp_id': self.exp_id, 'accuracy': 0.8})

    @patch('backend_server.mongo_utils.mongo_collection')
    def test_get_job_details_non_existing(self, mock_collection):
        exp_doc = {
            'user_id': self.user_id,
            'runs': {
                'experiment': [
                    {'exp_id': '123', 'accuracy': 0.9},
                    {'exp_id': '789', 'accuracy': 0.7}
                ]
            }
        }
        mock_collection.find_one.return_value = exp_doc
        job_details = mongo_utils.get_job_details(self.user_id, self.exp_name, self.exp_id)
        mock_collection.find_one.assert_called_once_with(
            {'user_id': self.user_id, 'runs.experiment.exp_id': self.exp_id}
        )
        self.assertIsNone(job_details)

if __name__ == '__main__':
    unittest.main()