import constants
import pymongo
from pymongo import MongoClient


mongo_cluster = MongoClient(constants.MONGO_CONNECTION)
mongo_db = mongo_cluster['QuanTAAS']
mongo_collection = mongo_db['userinfo']

def update_mongo_run(data):
    user_id = data['user_id']
    exp_name = '_'.join(data['exp_id'].split('_')[:-1])
    exp_id = data['exp_id']
    
    # Gather accuracy and size metrics
    pre_trained_accuracy = data['pre_trained_accuracy']
    pre_trained_size = data['pre_trained_size']
    
    pruned_accuracy = data['pruned_accuracy']
    pruned_size = data['pruned_size']
    pruned_model_filename = data['pruned_model_filename']
    pruned_training = data['pruned_training']  # Boolean flag

    quantized_accuracy = data['quantized_accuracy']
    quantized_size = data['quantized_size']
    quantized_model_filename = data['quantized_model_filename']
    quantized_training = data['quantized_training']  # Boolean flag
    
    # Update MongoDB with experiment details
    update_data = {
        f'runs.{exp_name}.pre_trained': {
            'accuracy': pre_trained_accuracy,
            'size': pre_trained_size,
            'model_filename': data['pre_trained_model_filename'],
            'training': False  # Pre-trained model is always complete
        },
        f'runs.{exp_name}.pruned': {
            'accuracy': pruned_accuracy,
            'size': pruned_size,
            'model_filename': pruned_model_filename,
            'training': pruned_training
        },
        f'runs.{exp_name}.quantized': {
            'accuracy': quantized_accuracy,
            'size': quantized_size,
            'model_filename': quantized_model_filename,
            'training': quantized_training
        }
    }
    
    mongo_collection.find_one_and_update(
        {'user_id': user_id, f'runs.{exp_name}.exp_id': exp_id},
        {'$set': update_data}
    )

    '''

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
'''
