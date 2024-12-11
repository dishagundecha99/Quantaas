import constants
from confluent_kafka import Consumer, OFFSET_BEGINNING

kafka_consumer_config = {
    'bootstrap.servers': constants.KAFKA_BOOTSTRAP_SERVER,
    'security.protocol': 'SASL_SSL',
    'sasl.mechanisms': 'PLAIN',
    'sasl.username': constants.KAFKA_USERNAME,
    'sasl.password': constants.KAFKA_SECRET,
    'group.id': constants.KAFKA_GROUP_ID,
    'auto.offset.reset': constants.KAFKA_OFFSET_RESET

}

def reset_offset(consumer, partitions):
    if constants.KAFKA_RESET_STATUS:
        for p in partitions:
            p.offset = OFFSET_BEGINNING
        consumer.assign(partitions)


consumer = Consumer(kafka_consumer_config)
consumer.subscribe([constants.KAFKA_CONSUME_JOB_TOPIC], on_assign=reset_offset)
