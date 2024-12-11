from backend_server import constants
from confluent_kafka import Producer


kafka_producer_config = {
    'bootstrap.servers': constants.KAFKA_BOOTSTRAP_SERVER,
    'security.protocol': 'SASL_SSL',
    'sasl.mechanisms': 'PLAIN',
    'sasl.username': constants.KAFKA_USERNAME,
    'sasl.password': constants.KAFKA_SECRET
}

producer = Producer(kafka_producer_config)

def delivery_callback(err, msg):
        if err:
            print(f'ERROR: Message failed delivery: {err}')
        else:
            print(f'Produced event to topic {msg.topic()} value = {msg.value().decode("utf-8")}')

def push_to_topic(value):
    producer.produce(constants.KAFKA_SUBMIT_JOB_TOPIC, value=value, callback=delivery_callback)
    producer.poll(10000)
    producer.flush()