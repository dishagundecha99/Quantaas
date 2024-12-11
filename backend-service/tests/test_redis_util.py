import unittest
from unittest.mock import patch
from backend_server import redis_util

class RedisUtilTestCase(unittest.TestCase):
    @patch('backend_server.redis_util.redis.delete')
    def test_evict_key(self, mock_delete):
        exp_name = 'example_key'
        redis_util.evict_key(exp_name)
        mock_delete.assert_called_once_with(exp_name)

if __name__ == '__main__':
    unittest.main()