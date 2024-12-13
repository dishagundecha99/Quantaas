from backend_server import app, session, mongo
from flask import redirect, url_for, jsonify, request

# Function to get the best evaluation (by accuracy) across different models (original, pruned, quantized) for each experiment
def best_runs_all(runs):
    best_runs = dict()
    for exp_name in runs:
        best_trial = best_run_exp(exp_name, runs.get(exp_name))
        best_runs[exp_name] = best_trial
    return best_runs

# Function to determine the best model based on accuracy (could be pruned, quantized, or original)
def best_run_exp(exp_name, runs):
    best_trial = f'{exp_name}_{0}'  # Default best trial is the first run
    best_accuracy = float(runs[0].get('accuracy', 0))  # Default to 0 if accuracy not found
    for trial in runs:
        accuracy = float(trial.get('accuracy', 0))  # Get accuracy or default to 0
        if accuracy > best_accuracy:
            best_trial = trial.get('exp_id')  # Get the exp_id for the best trial
            best_accuracy = accuracy
    return best_trial

@app.route('/prev-runs', defaults={'expname': None}, methods=['GET'])
@app.route('/prev-runs/<expname>', methods=['GET'])
def prev_runs(expname):
    if 'user-id' not in session:
        session['user-id'] = request.args.get('user-id')

    prev_runs = mongo.get_all_runs(session.get('user-id'))
    running_keys = list()

    # Remove experiments that are still in progress (if any)
    for exp in prev_runs:
        remove_key = False
        print(prev_runs)
        for trial in prev_runs.get(exp):
            print(trial)
            # No need to check for 'training', instead check if pruning or quantization is in progress
            if trial.get('pruning_in_progress') or trial.get('quantization_in_progress'):
                remove_key = True
                break
        if remove_key:
            running_keys.append(exp)

    # Remove the experiments that are still running from prev_runs
    for exp in running_keys:
        prev_runs.pop(exp)

    if len(prev_runs.keys()) == 0:
        data = {'message': 'No previous experiments were found'}
        return jsonify(data), 200
    elif expname is None:
        # Return all experiments and their evaluations
        data = {
            'message': f'found {len(prev_runs.keys())} experiments',
            'runs': prev_runs,
            'best_runs': best_runs_all(prev_runs)
        }
        return jsonify(data), 200
    else:
        # If an experiment name is provided, return only that experiment's data
        if expname not in prev_runs:
            data = {'message': 'Given experiment name does not exist'}
            return jsonify(data), 200
        else:
            # Return specific experiment with best evaluations for each model (original, pruned, quantized)
            data = {
                'message': 'Found experiment',
                'runs': prev_runs.get(expname),
                'best_runs': {expname: best_run_exp(expname, prev_runs.get(expname))}
            }
            return jsonify(data), 200
