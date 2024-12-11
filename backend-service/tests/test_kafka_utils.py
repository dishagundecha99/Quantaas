import unittest
from unittest.mock import patch
from backend_server.kafka_utils import push_to_topic, delivery_callback


class KafkaUtilsTestCase(unittest.TestCase):
    @patch('backend_server.kafka_utils.producer')
    def test_push_to_topic(self, mock_producer):
        value = 'test_value'
        mock_producer.produce.return_value = None
        mock_producer.poll.return_value = None
        mock_producer.flush.return_value = None

        push_to_topic(value)

        mock_producer.produce.assert_called_once_with('submit_job_topic', value=value, callback=delivery_callback)
        mock_producer.poll.assert_called_once_with(10000)
        mock_producer.flush.assert_called_once()

    def test_delivery_callback_with_error(self):
        err = 'test_error'
        msg = 'test_message'

        with patch('builtins.print') as mock_print:
            delivery_callback(err, msg)

            mock_print.assert_called_once_with(f'ERROR: Message failed delivery: {err}')

    def test_delivery_callback_without_error(self):
        err = None
        msg = 'test_message'

        with patch('builtins.print') as mock_print:
            delivery_callback(err, msg)

            mock_print.assert_called_once_with(f'Produced event to topic {msg.topic()} value = {msg.value().decode("utf-8")}')


if __name__ == '__main__':
    unittest.main()