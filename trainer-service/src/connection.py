from minio import Minio
import shutil
import os
from confluent_kafka import Consumer, Producer, OFFSET_BEGINNING

class readWriteMinioKafka():
    def __init__(self, reset=True) -> None:
        self.reset = reset
        kafka_consumer_config = {
            'bootstrap.servers': "pkc-4r087.us-west2.gcp.confluent.cloud", #TO_DO we need to change these whereever required
            'security.protocol': 'SASL_SSL',
            'sasl.mechanisms': 'PLAIN',
            'sasl.username': os.getenv("SASL_USERNAME") or "KRAW4JX76RIMK3Z6",
            'sasl.password': os.getenv("SASL_PASSWD") or "Yechd7PsQLv4ij46qRpO4utqQKhegZ/D3FoyoGnxkFVZgKYQQsgmPXOX8lErC0lD",
            'group.id': 'python_train_job_consumer',
            'auto.offset.reset': 'earliest'
        }
        self.CONSUMER = Consumer(kafka_consumer_config)

        self.topic_pull = "submit_job"
        self.CONSUMER.subscribe([self.topic_pull], on_assign=self.reset_offset)
        self.PRODUCER = Producer(kafka_consumer_config)
        self.topic_push = "completed_job"

        # establish minio connection
        minioHost = os.getenv("MINIO_HOST") or "localhost:9000"
        minioUser = os.getenv("MINIO_USER") or "minioadmin"
        minioPasswd = os.getenv("MINIO_PASSWD") or "minioadmin"

        self.MINIO_CLIENT = None
        try:
            self.MINIO_CLIENT = Minio(minioHost, access_key=minioUser, secret_key=minioPasswd, secure=False)
        except Exception as exp:
            print(f"Exception raised in worker loop: {str(exp)}")

        print("Established connections to both Kafka and Minio services")

    def get_clients(self):
        return self.CONSUMER, self.PRODUCER, self.MINIO_CLIENT
    
    def reset_offset(self, consumer, partitions):
        if self.reset == True:
            for p in partitions:
                p.offset = OFFSET_BEGINNING
            consumer.assign(partitions)

    def delivery_callback(self, err, msg):
        if err:
            print(f'ERROR: Message failed delivery: {err}')
        else:
            print(f'Produced event to topic {msg.topic()} value = {msg.value().decode("utf-8")}')

    def read_minio_data(self, pretrained_model_path, test_path, minio_bucket, save_path):
        print("Downloading the files", pretrained_model_path, test_path)

        # shutil.rmtree(save_path)
        os.makedirs(save_path, exist_ok=True)

        self.MINIO_CLIENT.fget_object(minio_bucket, "datasets/" + pretrained_model_path, save_path + pretrained_model_path)
        self.MINIO_CLIENT.fget_object(minio_bucket, "datasets/" + test_path, save_path + test_path)

        print("Placed file in temporary location", save_path)
        print("Files : ", os.listdir(save_path))   

    def write_to_minio(self, minio_bucket, file_name, repo_name):
        print("Storing the files", file_name, " to Minio")
        self.MINIO_CLIENT.fput_object(minio_bucket, repo_name, file_name)

        print("Saved the model to Minio")