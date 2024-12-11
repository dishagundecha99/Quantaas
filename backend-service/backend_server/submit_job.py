from backend_server import app, session, constants, minio, kafka, mongo
from werkzeug.utils import secure_filename
from flask import request, redirect, url_for, jsonify
from itertools import product
import os
import json

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in constants.ALLOWED_EXTENSIONS

def submit_job_paylod_validator(req):
    # check if user file is there in payload
    if not req.files:
        return 'Valied csv file needs to be uploaded', False
    if 'train_user_file' not in req.files or 'test_user_file' not in req.files:
        return 'Invalid key for uploading user file', False
    
    # check if it is a valid file type
    file = request.files['train_user_file']
    filename = secure_filename(file.filename)
    if not(allowed_file(filename)):
        return 'User can only upload valid csv files', False
    
    # check if it is a valid file type
    file = request.files['test_user_file']
    filename = secure_filename(file.filename)
    if not(allowed_file(filename)):
        return 'User can only upload valid csv files', False
    
    # check for validity form data
    model_meta_data = ['exp_name', 'task_type', 'hyperparams', 'model_name']
    for key in model_meta_data:
        if key not in request.form:
            return f'{key} not found in request payload', False
    return 'Valid Payload', True


@app.route('/submit-job', methods=['POST'])
def submit_training_job():
    if 'user-id' not in session:
        session['user-id'] = request.form.get('user-id')
    
    msg, is_payload_valid = submit_job_paylod_validator(request)
    if not is_payload_valid:
        return jsonify({'message':msg}), 400
    
    train_file = request.files['train_user_file']
    test_file = request.files['test_user_file']

    minio.upload_dataset(session.get('user-id'), train_file)
    minio.upload_dataset(session.get('user-id'), test_file)
    
    exp_name = request.form.get('exp_name')
    task_type = request.form.get('task_type')
    model_name = request.form.get('model_name')
    hyperparams = json.loads(request.form.get('hyperparams'))

    valid_hp_keys, valid_hps = list(), list()
    for key in hyperparams:
        if len(hyperparams.get(key)) > 0:
            valid_hp_keys.append(key)
            valid_hps.append(hyperparams.get(key))

    hyp_combs = list(product(*valid_hps))

    for i, hyp_comb in enumerate(hyp_combs):
        train_meta_data = dict()
        train_meta_data['exp_id'] = f'{exp_name}_{i}'
        train_meta_data['task_type'] = task_type
        train_meta_data['model_name'] = model_name
        train_meta_data['train_dataset'] = secure_filename(train_file.filename)
        train_meta_data['test_dataset'] = secure_filename(test_file.filename)
        train_meta_data['hyperparams'] = dict()
        train_meta_data['minio_bucket'] = minio.get_user_bucket_name(session.get('user-id'))
        train_meta_data['user_id'] = session.get('user-id')
        for j, hyp_name in enumerate(valid_hp_keys):
            train_meta_data['hyperparams'][hyp_name] = hyp_comb[j]
        kafka.push_to_topic(json.dumps(train_meta_data)) 
        train_meta_data.pop('user_id')  
        mongo.record_train_meta_data(session.get('user-id'), train_meta_data, exp_name)  

    return jsonify({'message':msg}), 200