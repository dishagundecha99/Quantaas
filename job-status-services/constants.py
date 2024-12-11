import os 

# Confluent Kafka configs
KAFKA_USERNAME = os.getenv("KAFKA_USERNAME")
KAFKA_SECRET = os.getenv("KAFKA_SECRET")
KAFKA_BOOTSTRAP_SERVER = "pkc-4r087.us-west2.gcp.confluent.cloud"
KAFKA_CONSUME_JOB_TOPIC = 'completed_job'
KAFKA_GROUP_ID = 'job_status'
KAFKA_OFFSET_RESET = 'earliest'
KAFKA_RESET_STATUS = True

# Mongo configs
MONGO_ATLAS_SECRET = os.getenv("MONGO_ATLAS_SECRET")
MONGO_CONNECTION = f'mongodb+srv://QuanTAASuser:{MONGO_ATLAS_SECRET}@quantaas.wqjqmye.mongodb.net/?retryWrites=true&w=majority'