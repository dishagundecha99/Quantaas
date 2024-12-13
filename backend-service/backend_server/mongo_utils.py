from backend_server import constants
import pymongo
from pymongo import MongoClient


mongo_cluster = MongoClient(constants.MONGO_CONNECTION)
mongo_db = mongo_cluster['QuanTAAS']
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
#TO_DO here we can add train_meta_data differently i am not sure what all to add yet, but we can have the file paths to pruned model and stuff or just keep it like this
def record_train_meta_data(user_id, model_meta_data, exp_name):
    find_or_create_user_doc(user_id)
    find_or_create_exp_doc(user_id, exp_name)
    model_meta_data['evaluation'] = {0.0,0,True}
    model_meta_data['pruning'] = {0.0,0,None,True}
    model_meta_data['quantization'] = {0.0,0,None, True}
    model_meta_data['model_name'] = ''
    update_key = f'runs.{exp_name}'
    mongo_collection.update_one(
        {'user_id': user_id},
        {'$push': {
            update_key: model_meta_data
        }}
    )

def get_all_runs(user_id):
    user_doc = mongo_collection.find_one({'user_id': user_id})
    if user_doc is not None:
        runs = user_doc.get('runs')
        return runs
    return {}

#TO_DO here make sure the trial returned has file paths to both pruned and quantized for now it looks ok
def get_job_details(user_id, exp_name, exp_id):
    exp_filter_key = f'runs.{exp_name}.exp_id'
    exp_doc = mongo_collection.find_one(
        {'user_id': user_id, exp_filter_key:exp_id}
    )
    for trial in exp_doc['runs'][exp_name]:
        if trial.get('exp_id') == exp_id:
            return trial
    return None


# this does not make sense as we need history but i am keeping the commented code if needed later but mostly need to just remove 

#TO_DO we don't need runs concept 
    '''from backend_server import constants
import pymongo
from pymongo import MongoClient

# MongoDB setup
mongo_cluster = MongoClient(constants.MONGO_CONNECTION)
mongo_db = mongo_cluster['HypTAAS']
mongo_collection = mongo_db['userinfo']

# Create or find user document
def create_user_doc(user_id):
    mongo_collection.insert_one({
        'user_id': user_id,
        'models': {
            'pretrained': None,
            'pruned': None,
            'quantized': None
        }
    })

def find_or_create_user_doc(user_id):
    user_doc = mongo_collection.find_one({'user_id': user_id})
    if not user_doc:
        create_user_doc(user_id)

# Record model paths for the user
def record_model_paths(user_id, pretrained_model_path, pruned_model_path, quantized_model_path):
    find_or_create_user_doc(user_id)
    mongo_collection.update_one(
        {'user_id': user_id},
        {'$set': {
            'models.pretrained': pretrained_model_path,
            'models.pruned': pruned_model_path,
            'models.quantized': quantized_model_path
        }}
    )

# Get model details for the user
def get_model_details(user_id):
    user_doc = mongo_collection.find_one({'user_id': user_id})
    if user_doc:
        return user_doc.get('models', {})
    return None

'''