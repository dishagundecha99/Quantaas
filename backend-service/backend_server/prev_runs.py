from backend_server import app, session, mongo
from flask import jsonify, request

def best_run(run):
    accuracies = {
        'original': run.get('model_meta_data', {}).get('evaluation', {}).get('accuracy', 0),
        'pruned': run.get('model_meta_data', {}).get('pruning', {}).get('accuracy', 0),
        'quantized': run.get('model_meta_data', {}).get('quantization', {}).get('accuracy', 0),
    }
    # Determine the model type with the highest accuracy
    best_model = max(accuracies, key=accuracies.get)
    return {'best_model': best_model, 'accuracy': accuracies[best_model]}

@app.route('/prev-runs', defaults={'expname': None}, methods=['GET'])
@app.route('/prev-runs/<expname>', methods=['GET'])
def prev_runs(expname):
    if 'user_id' not in session:
        session['user_id'] = request.args.get('user-id')

    prev_runs = mongo.get_all_runs(session.get('user_id'))

    # Filter out experiments where pruning or quantization is still in progress
    completed_runs = {
        exp_name: run for exp_name, run in prev_runs.items()
        if run.get('model_meta_data', {}).get('pruning', {}).get('completed', True) and
           run.get('model_meta_data', {}).get('quantization', {}).get('completed', True)
    }

    if not completed_runs:
        return jsonify({'message': 'No previous experiments were found'}), 200

    if expname is None:
        # Return all completed experiments and their best model evaluations
        data = {
            'message': f'Found {len(completed_runs)} experiments',
            'runs': completed_runs,
            'best_runs': {exp_name: best_run(run) for exp_name, run in completed_runs.items()}
        }
        return jsonify(data), 200
    else:
        # Return a specific experiment's data
        if expname not in completed_runs:
            return jsonify({'message': 'Given experiment name does not exist'}), 200
        else:
            data = {
                'message': 'Found experiment',
                'runs': completed_runs[expname],
                'best_runs': {expname: best_run(completed_runs[expname])}
            }
            return jsonify(data), 200
