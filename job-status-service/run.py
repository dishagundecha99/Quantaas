from kafka_util import consumer
from mongo_util import update_mongo_run
import json


def main():
    while True:
        msg = consumer.poll(1.0)
        if msg is None:
            print('Waiting...')
        elif msg.error():
            print(f'ERROR: {msg.error()}')
        else:
            topic, value = msg.topic(), msg.value().decode("utf-8")
            print(f'Consumed event from topic {msg.topic()} value = {msg.value().decode("utf-8")}')
            update_mongo_run(json.loads(value))
            


if __name__ == '__main__':
    main()