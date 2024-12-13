from backend_server import app, session, mongo
from flask import jsonify, request

@app.route('/current-run', methods=['GET'])
def current_run():
    if 'user_id' not in session:
        session['user_id'] = request.args.get('user-id')

    prev_runs = mongo.get_all_runs(session.get('user_id'))

    for exp_name, run in prev_runs.items():
        # Check if pruning or quantization is still in progress
        pruning_in_progress = not run.get('model_meta_data', {}).get('pruning', {}).get('completed', True)
        quantization_in_progress = not run.get('model_meta_data', {}).get('quantization', {}).get('completed', True)

        if pruning_in_progress or quantization_in_progress:
            data = {
                'message': f'Experiment "{exp_name}" is currently running',
                'current-run': run
            }
            return jsonify(data), 200

    return jsonify({'message': 'No running jobs found'}), 200
