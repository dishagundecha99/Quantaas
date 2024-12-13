from backend_server import app, session, constants, minio, kafka, mongo
from werkzeug.utils import secure_filename
from flask import request, jsonify
import json
import os

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in constants.ALLOWED_EXTENSIONS

def submit_job_payload_validator(req):
    # Check if user file is in the payload
    if not req.files:
        return 'Valid file needs to be uploaded', False
    if 'test_user_file' not in req.files:
        return 'Invalid key for uploading user file', False

    # Check for valid file type (CSV)
    file = req.files['test_user_file']
    filename = secure_filename(file.filename)
    if not allowed_file(filename):
        return 'User can only upload valid CSV files', False

    # Check for required form data
    required_fields = ['exp_name', 'model_name', 'task_type']
    for field in required_fields:
        if field not in req.form:
            return f'{field} not found in request payload', False

    return 'Valid Payload', True

@app.route('/submit-job', methods=['POST'])
def submit_model_processing_job():
    if 'user_id' not in session:
        session['user_id'] = request.form.get('user-id')

    # Validate payload
    msg, is_payload_valid = submit_job_payload_validator(request)
    if not is_payload_valid:
        return jsonify({'message': msg}), 400

    # Get the test file and upload to MinIO
    test_file = request.files['test_user_file']
    minio.upload_dataset(session.get('user_id'), test_file)

    # Extract metadata from the request
    exp_name = request.form.get('exp_name')
    model_name = request.form.get('model_name')
    task_type = request.form.get('task_type')

    # Prepare metadata for the job
    model_meta_data = {
        'exp_id': exp_name,
        'task_type': task_type,
        'model_name': model_name,
        'test_dataset': secure_filename(test_file.filename),
        'minio_bucket': minio.get_user_bucket_name(session.get('user_id')),
        'user_id': session.get('user_id'),
        'evaluation': {
            'accuracy': None,
            'size': None,
            'status': 'pending'
        },
        'pruning': {
            'accuracy': None,
            'size': None,
            'path': None,
            'status': 'pending'
        },
        'quantization': {
            'accuracy': None,
            'size': None,
            'path': None,
            'status': 'pending'
        }
    }

    # Send job to Kafka for processing
    kafka.push_to_topic(json.dumps({
        'action': 'process_model',
        'job_id': f"job_{exp_name}_{session.get('user_id')}",
        'metadata': model_meta_data,
    }))

    # Record metadata in MongoDB
    mongo.record_train_meta_data(
        session.get('user_id'),
        model_meta_data,
        exp_name
    )

    return jsonify({'message': 'Job submitted successfully'}), 200
