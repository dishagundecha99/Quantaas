import constants
import pymongo
from pymongo import MongoClient


mongo_cluster = MongoClient(constants.MONGO_CONNECTION)
mongo_db = mongo_cluster['QuanTAAS']
mongo_collection = mongo_db['userinfo']


def update_mongo_run(data):
    user_id = data['user_id']
    exp_name = '_'.join(data['exp_id'].split('_')[:-1])
    exp_id =  data['exp_id']
    accuracy = data['accuracy']
    training = False
    model_filename = data['model_filename']
    filter_key = f'runs.{exp_name}.exp_id'
    update_key_root = f'runs.{exp_name}.$.'
    acc_update = update_key_root + 'accuracy'
    training_update = update_key_root + 'training'
    model_filename_update = update_key_root + 'model_filename'
    mongo_collection.find_one_and_update(
        {'user_id':user_id,filter_key:exp_id},
        {'$set': {
            acc_update:accuracy,
            training_update:training,
            model_filename_update:model_filename
            }
        }
    )