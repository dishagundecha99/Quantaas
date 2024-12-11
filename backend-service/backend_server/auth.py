from backend_server import app, session, constants
from flask import redirect, url_for, jsonify
from flask import request as flask_request
import json
from oauthlib.oauth2 import WebApplicationClient
import requests

client = WebApplicationClient(constants.GOOGLE_CLIENT_ID)

def get_google_provider_cfg():
    return requests.get(constants.GOOGLE_DISCOVERY_URL).json()

@app.route('/auth', methods=['GET'])
def auth():
    session['user_id'] = flask_request.args.get('user-id')
    data = {'user_id': session['user_id'],
                'logged_in': True}
    return jsonify(data), 200


@app.route('/login', methods=['GET', 'POST'])
def login():
    if flask_request.method == 'POST':
        session['username'] = flask_request.form['username']
        return redirect(url_for('index'))
    else:
        # Find out what URL to hit for Google login
        google_provider_cfg = get_google_provider_cfg()
        authorization_endpoint = google_provider_cfg["authorization_endpoint"]

        # Use library to construct the request for login and provide
        # scopes that let you retrieve user's profile from Google
        print(flask_request.base_url + "/callback",)
        request_uri = client.prepare_request_uri(
            authorization_endpoint,
            redirect_uri=flask_request.base_url + "/callback",
            scope=["openid", "email", "profile"],
        )
        return redirect(request_uri)
    

@app.route("/login/callback")
def callback():
    # Get authorization code Google sent back to you
    code = flask_request.args.get("code")

    # Find out what URL to hit to get tokens that allow you to ask for
    # things on behalf of a user
    google_provider_cfg = get_google_provider_cfg()
    token_endpoint = google_provider_cfg["token_endpoint"]

    # Prepare and send request to get tokens! Yay tokens!
    token_url, headers, body = client.prepare_token_request(
        token_endpoint,
        authorization_response=flask_request.url,
        redirect_url=flask_request.base_url,
        code=code,
    )
    token_response = requests.post(
        token_url,
        headers=headers,
        data=body,
        auth=(constants.GOOGLE_CLIENT_ID, constants.GOOGLE_CLIENT_SECRET),
    )

    # Parse the tokens!
    client.parse_request_body_response(json.dumps(token_response.json()))

    userinfo_endpoint = google_provider_cfg["userinfo_endpoint"]
    uri, headers, body = client.add_token(userinfo_endpoint)
    userinfo_response = requests.get(uri, headers=headers, data=body)

    if userinfo_response.json().get("email_verified"):
        user_email = userinfo_response.json()["email"]
        user_name = userinfo_response.json()["given_name"]
        session['user_id'] = user_email
        session['user_name'] = user_name
    else:
        return "User email not available or not verified by Google.", 400

    return redirect(url_for("index"))

