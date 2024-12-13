from flask import Flask, session
from flask_cors import CORS

from backend_server import constants
from backend_server import minio_utils as minio
from backend_server import kafka_utils as kafka
from backend_server import mongo_utils as mongo

app = Flask(__name__)
app.secret_key = constants.FLASK_SECRET
CORS(app)

from backend_server import home, auth, submit_job, prev_runs, current_run, download_pruned_model, download_quantized_model
#TO_DO see if we need more or less 