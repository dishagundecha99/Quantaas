import os
import secrets

# Flask configs
FLASK_SECRET = os.getenv('FLASK_SECRET') or secrets.token_hex()
ALLOWED_EXTENSIONS = ['csv']#TO_DO Change the format allowed to csv and the pretrained model format
MODEL_SAVE_FOLDER = 'models'

# Google Oauth configs
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET") 
GOOGLE_DISCOVERY_URL =  "https://accounts.google.com/.well-known/openid-configuration"

# Minio configs
minioHost = os.getenv("MINIO_HOST") or "localhost:9000"
minioUser = os.getenv("MINIO_USER") or "minioadmin"
minioPasswd = os.getenv("MINIO_PASSWD") or "minioadmin"

# Confluent Kafka configs
KAFKA_USERNAME = os.getenv("KAFKA_USERNAME")
KAFKA_SECRET = os.getenv("KAFKA_SECRET")
KAFKA_BOOTSTRAP_SERVER = "pkc-4r087.us-west2.gcp.confluent.cloud" #TO_DO this is to be done post creating the kafka cluster 
# KAFKA_SUBMIT_JOB_TOPIC = 'demo' #TO_DO i think they scammed by creating a new file here
KAFKA_SUBMIT_JOB_TOPIC = 'submit_job'

# Mongo configs
MONGO_ATLAS_SECRET = os.getenv("MONGO_ATLAS_SECRET")
MONGO_CONNECTION = f'mongodb+srv://HypTAASuser:{MONGO_ATLAS_SECRET}@hyptaas.wqjqmye.mongodb.net/?retryWrites=true&w=majority'#TO_DO I am creating new quantaas cluster we will change this after cluster creation

# Redis configs
REDISHOST = os.getenv("REDISTOGO_URL") or "localhost"
REDISPORT = os.getenv("REDISTOGO_PORT") or 6379
