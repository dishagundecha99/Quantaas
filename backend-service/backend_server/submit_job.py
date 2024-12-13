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
        return 'Valid file needs to be uploaded', False
    if 'test_user_file' not in req.files:
        return 'Invalid key for uploading user file', False
    
    # check if it is a valid file type
    file = request.files['test_user_file']
    filename = secure_filename(file.filename)
    if not(allowed_file(filename)):
        return 'User can only upload valid csv files', False
    
    # check for validity form data #TO_DO change this we don't need all this data
    model_meta_data = ['exp_name']
    for key in model_meta_data:
        if key not in request.form:
            return f'{key} not found in request payload', False
    return 'Valid Payload', True

@app.route('/submit-job', methods=['POST'])
def submit_model_processing_job():
    if 'user-id' not in session:
        session['user-id'] = request.form.get('user-id')
    
    # Validate payload
    msg, is_payload_valid = submit_job_paylod_validator(request)
    if not is_payload_valid:
        return jsonify({'message': msg}), 400
    
    test_file = request.files['test_user_file']
    minio.upload_dataset(session.get('user-id'), test_file)
    
    # Collect metadata from request
    exp_name = request.form.get('exp_name')
    model_name = request.form.get('model_name')
    task_type = request.form.get('task_type')

    # Prepare metadata for MongoDB and Kafka
    model_meta_data = {
        'exp_id': exp_name,
        'task_type': task_type,
        'model_name': model_name,
        'test_dataset': secure_filename(test_file.filename),
        'minio_bucket': minio.get_user_bucket_name(session.get('user-id')),
        'user_id': session.get('user-id'),
        'pruning': {
            'status': 'pending',
            'pruned_model_path': None,
            'pruned_model_accuracy': None,
            'pruned_model_size': None,
        },
        'quantization': {
            'status': 'pending',
            'quantized_model_path': None,
            'quantized_model_accuracy': None,
            'quantized_model_size': None,
        },
        'evaluation': {
            'original_model_accuracy': None,
            'original_model_size': None,
        },
    }

    # Push tasks to Kafka
    kafka.push_to_topic(json.dumps({
        'action': 'evaluate_original',
        'model_name': model_name,
        'exp_id': exp_name,
        'user_id': session.get('user-id'),
        'test_dataset': model_meta_data['test_dataset'],
    }))

    kafka.push_to_topic(json.dumps({
        'action': 'prune_model',
        'model_name': model_name,
        'exp_id': exp_name,
        'user_id': session.get('user-id'),
        'test_dataset': model_meta_data['test_dataset'],
    }))

    kafka.push_to_topic(json.dumps({
        'action': 'quantize_model',
        'model_name': model_name,
        'exp_id': exp_name,
        'user_id': session.get('user-id'),
        'test_dataset': model_meta_data['test_dataset'],
    }))

    # Record metadata in MongoDB
    mongo.record_train_meta_data(
        session.get('user-id'), 
        model_meta_data, 
        exp_name
    )

    return jsonify({'message': 'Job submitted successfully'}), 200

#TO_DO i am not sure how much we need this as we are not training the models only for testing we will use

'''
@app.route('/submit-job', methods=['POST'])
def submit_training_job():
    if 'user-id' not in session:
        session['user-id'] = request.form.get('user-id')
    
    msg, is_payload_valid = submit_job_paylod_validator(request)
    if not is_payload_valid:
        return jsonify({'message':msg}), 400
    
    test_file = request.files['test_user_file']

    minio.upload_dataset(session.get('user-id'), test_file)
    
    exp_name = request.form.get('exp_name')
    task_type = request.form.get('task_type')
    model_name = request.form.get('model_name')
  #  hyperparams = json.loads(request.form.get('hyperparams'))

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

from backend_server import app, session, constants, minio, kafka, mongo
from werkzeug.utils import secure_filename
from flask import request, jsonify
import json
from backend_server import model_utils  # (You would have a utility file to handle pruning and quantizing)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in constants.ALLOWED_EXTENSIONS

def submit_job_payload_validator(req):
    # Check if user file is there in payload
    if not req.files:
        return 'Valid model file needs to be uploaded', False
    if 'pretrained_model' not in req.files:
        return 'Invalid key for uploading pre-trained model file', False
    
    # Check if it is a valid file type (i.e., model file, can be .h5, .pth, etc.)
    file = request.files['pretrained_model']
    filename = secure_filename(file.filename)
    if not(allowed_file(filename)):
        return 'User can only upload valid model files', False
    
    # Check for validity of form data (parameters that are required)
    model_meta_data = ['exp_name', 'task_type', 'test_data']
    for key in model_meta_data:
        if key not in request.form:
            return f'{key} not found in request payload', False
    return 'Valid Payload', True


@app.route('/submit-job', methods=['POST'])
def submit_evaluation_job():
    if 'user-id' not in session:
        session['user-id'] = request.form.get('user-id')
    
    msg, is_payload_valid = submit_job_payload_validator(request)
    if not is_payload_valid:
        return jsonify({'message': msg}), 400

    # Get the pretrained model and test data from the payload
    pretrained_model_file = request.files['pretrained_model']
    test_data_file = request.files['test_data']  # Assuming test data is also uploaded

    # Upload files to MinIO
    minio.upload_model(session.get('user-id'), pretrained_model_file)
    minio.upload_dataset(session.get('user-id'), test_data_file)

    # Extract information from the request
    exp_name = request.form.get('exp_name')
    task_type = request.form.get('task_type')
    test_data_filename = secure_filename(test_data_file.filename)

    # Prepare the metadata for the models (original, pruned, quantized)
    model_metadata = {
        "exp_name": exp_name,
        "task_type": task_type,
        "test_data": test_data_filename,
        "user_id": session.get('user-id'),
        "pretrained_model_filename": pretrained_model_file.filename,
    }

    # Perform model pruning and quantization
    pruned_model = model_utils.prune_model(pretrained_model_file)
    quantized_model = model_utils.quantize_model(pretrained_model_file)

    # Save or upload pruned and quantized models
    pruned_model_filename = f"{exp_name}_pruned_model.h5"
    quantized_model_filename = f"{exp_name}_quantized_model.h5"
    minio.upload_model(session.get('user-id'), pruned_model, pruned_model_filename)
    minio.upload_model(session.get('user-id'), quantized_model, quantized_model_filename)

    # Evaluate models (original, pruned, quantized) on the test data
    evaluation_results = {}

    # Run the models on the test dataset and collect evaluation results
    evaluation_results['original'] = model_utils.evaluate_model(pretrained_model_file, test_data_file)
    evaluation_results['pruned'] = model_utils.evaluate_model(pruned_model, test_data_file)
    evaluation_results['quantized'] = model_utils.evaluate_model(quantized_model, test_data_file)

    # Save the evaluation results to MongoDB
    mongo.record_evaluation_results(session.get('user-id'), exp_name, evaluation_results)

    # Publish the results to Kafka (if required for asynchronous processing or logging)
    kafka.push_to_topic(json.dumps(evaluation_results))

    return jsonify({'message': 'Evaluation job submitted successfully', 'results': evaluation_results}), 200
'''