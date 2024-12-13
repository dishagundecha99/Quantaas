import constants
import pymongo
from pymongo import MongoClient

# Initialize MongoDB connection
mongo_cluster = MongoClient(constants.MONGO_CONNECTION)
mongo_db = mongo_cluster['QuanTAAS']
mongo_collection = mongo_db['userinfo']

def update_mongo_run(data):
    user_id = data['user_id']
    exp_name = '_'.join(data['exp_id'].split('_')[:-1])
    exp_id = data['exp_id']

    # Update data for evaluation, pruning, and quantization
    update_data = {
        'evaluation': {
            'accuracy': data.get('pre_trained_accuracy'),
            'size': data.get('pre_trained_size'),
            'status': 'completed' if data.get('pre_trained_accuracy') is not None else 'pending'
        },
        'pruning': {
            'accuracy': data.get('pruned_accuracy'),
            'size': data.get('pruned_size'),
            'path': data.get('pruned_model_filename'),
            'status': 'completed' if not data.get('pruned_training', True) else 'pending'
        },
        'quantization': {
            'accuracy': data.get('quantized_accuracy'),
            'size': data.get('quantized_size'),
            'path': data.get('quantized_model_filename'),
            'status': 'completed' if not data.get('quantized_training', True) else 'pending'
        }
    }

    # Update the MongoDB document
    mongo_collection.find_one_and_update(
        {'user_id': user_id, f'runs.{exp_name}.exp_id': exp_id},
        {'$set': {
            f'runs.{exp_name}.$.evaluation': update_data['evaluation'],
            f'runs.{exp_name}.$.pruning': update_data['pruning'],
            f'runs.{exp_name}.$.quantization': update_data['quantization']
        }}
    )
