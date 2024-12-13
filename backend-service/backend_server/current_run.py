from backend_server import app, session, mongo
from flask import redirect, url_for, jsonify, request


@app.route('/current-run', methods=['GET'])
def current_run():
    if 'user-id' not in session:
        session['user-id'] = request.args.get('user-id')

    prev_runs = mongo.get_all_runs(session.get('user-id'))
    for exp_name in prev_runs:
        for trial in prev_runs.get(exp_name):
            if trial.get('training'): #TO_DO this is ok, just make sure your training value is set when we are pruning or quantizing the model
                data = {'message': f'{exp_name} is currently running',
                        'current-run': prev_runs.get(exp_name)}
                return jsonify(data), 200
    data = {'message': 'No running jobs found'}
    return jsonify(data), 200

'''
def current_run():
    if 'user-id' not in session:
        session['user-id'] = request.args.get('user-id')

    prev_runs = mongo.get_all_runs(session.get('user-id'))
    for exp_name in prev_runs:
        for trial in prev_runs.get(exp_name):
            if trial.get('training'): #TO_DO this is ok, just make sure your training value is set when we are pruning or quantizing the model
                data = {'message': f'{exp_name} is currently running',
                        'current-run': prev_runs.get(exp_name)}
                return jsonify(data), 200
    data = {'message': 'No running jobs found'}
    return jsonify(data), 200'''