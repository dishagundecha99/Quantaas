from backend_server import app, session, mongo, minio, constants
from flask import jsonify, send_file, request
import os
import shutil

@app.route('/download-pruned-model/<expid>', methods=['GET'])
def download_pruned_model(expid):
    if 'user_id' not in session:
        session['user_id'] = request.args.get('user-id')

    clean_up()
    exp_name = '_'.join(expid.split('_')[:-1])
    job_details = mongo.get_job_details(session.get('user_id'), exp_name, expid)
    if job_details is None:
        return jsonify({'message': 'Failed to get model details'}), 200

    # Get the pruned model path from MongoDB
    model_path = job_details.get("pruning", {}).get("path")
    if not model_path:
        return jsonify({'error': 'Pruned model not found'}), 404

    # MinIO paths
    minio_bucket = job_details.get("minio_bucket")
    minio_file_path = os.path.join('backend_server', constants.MODEL_SAVE_FOLDER, os.path.basename(model_path))
    res_file_path = os.path.join(constants.MODEL_SAVE_FOLDER, os.path.basename(model_path))

    # Download file from MinIO
    minio.download_model_file(minio_bucket, model_path, minio_file_path)
    return send_file(res_file_path, as_attachment=True)

@app.route('/download-quantized-model/<expid>', methods=['GET'])
def download_quantized_model(expid):
    if 'user_id' not in session:
        session['user_id'] = request.args.get('user-id')

    clean_up()
    exp_name = '_'.join(expid.split('_')[:-1])
    job_details = mongo.get_job_details(session.get('user_id'), exp_name, expid)
    if job_details is None:
        return jsonify({'message': 'Failed to get model details'}), 200

    # Get the quantized model path from MongoDB
    model_path = job_details.get("quantization", {}).get("path")
    if not model_path:
        return jsonify({'error': 'Quantized model not found'}), 404

    # MinIO paths
    minio_bucket = job_details.get("minio_bucket")
    minio_file_path = os.path.join('backend_server', constants.MODEL_SAVE_FOLDER, os.path.basename(model_path))
    res_file_path = os.path.join(constants.MODEL_SAVE_FOLDER, os.path.basename(model_path))

    # Download file from MinIO
    minio.download_model_file(minio_bucket, model_path, minio_file_path)
    return send_file(res_file_path, as_attachment=True)

def clean_up():
    path = os.path.join('backend_server', constants.MODEL_SAVE_FOLDER)
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)
