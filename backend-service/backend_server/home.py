from backend_server import app, session, constants
from flask import request, redirect, url_for, jsonify

@app.route('/')
def index():
    if 'user_id' in session:
        data = {'user_id': session['user_id'],
                'logged_in': True}
        return jsonify(data), 200
    else:
        return '<a class="button" href="/auth">Google Login</a>'
    
