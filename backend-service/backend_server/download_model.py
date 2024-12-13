from backend_server import app, session, mongo, minio, constants
from flask import redirect, url_for, jsonify, send_file, request
import os
import shutil
'''
@app.route('/download-model/<expid>', methods=['GET'])
def download_model(expid):
    if 'user-id' not in session:
        session['user-id'] = request.args.get('user-id')
    
    clean_up()
    exp_name = '_'.join(expid.split('_')[:-1])
    job_details = mongo.get_job_details(session.get('user-id'), exp_name, expid)
    if job_details is None:
        msg = 'failed to get model details'
        return jsonify({'message':msg}), 200
    model_file = f'{job_details.get("exp_id")}/{job_details.get("model_filename")}'
    minio_file_path = os.path.join('backend_server', constants.MODEL_SAVE_FOLDER, job_details.get("model_filename"))
    res_file_path = os.path.join(constants.MODEL_SAVE_FOLDER, job_details.get("model_filename"))
    minio.download_model_file(job_details.get('minio_bucket'), model_file, minio_file_path)
    return send_file(res_file_path,as_attachment=True)
'''
#TO_DO frontend should route to the respective method based on what is expected to be downloaded. Also confirm that the file names and paths are correct in all places.

@app.route('/download-pruned-model/<expid>', methods=['GET'])
def download_pruned_model(expid):
    if 'user-id' not in session:
        session['user-id'] = request.args.get('user-id')
    
    clean_up()
    exp_name = '_'.join(expid.split('_')[:-1])
    job_details = mongo.get_job_details(session.get('user-id'), exp_name, expid)
    if job_details is None:
        msg = 'Failed to get model details'
        return jsonify({'message': msg}), 200

    # Get the pruned model filename
    model_filename = job_details.get("pruned_model_filename")
    if not model_filename:
        return jsonify({'error': 'Pruned model not found'}), 404

    # MinIO paths
    model_file = f'{job_details.get("exp_id")}/{model_filename}'
    minio_file_path = os.path.join('backend_server', constants.MODEL_SAVE_FOLDER, model_filename)
    res_file_path = os.path.join(constants.MODEL_SAVE_FOLDER, model_filename)

    # Download file from MinIO
    minio.download_model_file(job_details.get('minio_bucket'), model_file, minio_file_path)
    return send_file(res_file_path, as_attachment=True)

@app.route('/download-quantized-model/<expid>', methods=['GET'])
def download_quantized_model(expid):
    if 'user-id' not in session:
        session['user-id'] = request.args.get('user-id')
    
    clean_up()
    exp_name = '_'.join(expid.split('_')[:-1])
    job_details = mongo.get_job_details(session.get('user-id'), exp_name, expid)
    if job_details is None:
        msg = 'Failed to get model details'
        return jsonify({'message': msg}), 200

    # Get the quantized model filename
    model_filename = job_details.get("quantized_model_filename")
    if not model_filename:
        return jsonify({'error': 'Quantized model not found'}), 404

    # MinIO paths
    model_file = f'{job_details.get("exp_id")}/{model_filename}'
    minio_file_path = os.path.join('backend_server', constants.MODEL_SAVE_FOLDER, model_filename)
    res_file_path = os.path.join(constants.MODEL_SAVE_FOLDER, model_filename)

    # Download file from MinIO
    minio.download_model_file(job_details.get('minio_bucket'), model_file, minio_file_path)
    return send_file(res_file_path, as_attachment=True)


def clean_up():
    path = os.path.join('backend_server', constants.MODEL_SAVE_FOLDER)
    op_foler_exists = os.path.exists(path)
    if op_foler_exists:
        shutil.rmtree(path)


        
