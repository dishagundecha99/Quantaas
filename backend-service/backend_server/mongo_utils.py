from backend_server import constants
import pymongo
from pymongo import MongoClient


mongo_cluster = MongoClient(constants.MONGO_CONNECTION)
mongo_db = mongo_cluster['HypTAAS']
mongo_collection = mongo_db['userinfo']

def create_user_doc(user_id):
    mongo_collection.insert_one({'user_id': user_id,'runs': {}})

def find_or_create_user_doc(user_id):
    user_doc = mongo_collection.find_one({'user_id': user_id})
    if not user_doc:
        create_user_doc(user_id)

def find_or_create_exp_doc(user_id, exp_name):
    user_doc = mongo_collection.find_one({'user_id': user_id})
    runs = user_doc.get('runs')
    if exp_name not in runs:
        runs[exp_name] = []
        mongo_collection.update_one(
            {'user_id': user_id},
            {'$set': {
                'runs': runs
            }}
        )

def record_train_meta_data(user_id, train_meta_data, exp_name):
    find_or_create_user_doc(user_id)
    find_or_create_exp_doc(user_id, exp_name)
    train_meta_data['accuracy'] = 0.0
    train_meta_data['training'] = True
    train_meta_data['model_filename'] = ''
    update_key = f'runs.{exp_name}'
    mongo_collection.update_one(
        {'user_id': user_id},
        {'$push': {
            update_key: train_meta_data
        }}
    )

def get_all_runs(user_id):
    user_doc = mongo_collection.find_one({'user_id': user_id})
    if user_doc is not None:
        runs = user_doc.get('runs')
        return runs
    return {}

def get_job_details(user_id, exp_name, exp_id):
    exp_filter_key = f'runs.{exp_name}.exp_id'
    exp_doc = mongo_collection.find_one(
        {'user_id': user_id, exp_filter_key:exp_id}
    )
    for trial in exp_doc['runs'][exp_name]:
        if trial.get('exp_id') == exp_id:
            return trial
    return None