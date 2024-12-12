from backend_server import constants
import redis

redis = redis.Redis(host=constants.REDISHOST, port=constants.REDISPORT, db=0)

def set_best_models(exp_name, best_models):
    redis.set(exp_name, best_models)

def get_best_models(exp_name):
    redis.get(exp_name)

def check_if_exists(exp_name):
    return redis.exists(exp_name)

def evict_key(exp_name):
    redis.delete(exp_name)
#TO_DO 
    '''
import redis
import json
from backend_server import constants

# Create Redis connection
redis_client = redis.Redis(host=constants.REDISHOST, port=constants.REDISPORT, db=0)

# Function to store best models (original, pruned, quantized) for an experiment
def set_best_models(exp_name, best_models):
    """
    Stores the best models (original, pruned, quantized) in Redis under the given experiment name.
    `best_models` should be a dictionary containing model metadata (accuracy, size, filename).
    """
    # Convert best models to JSON string to store in Redis
    redis_client.set(exp_name, json.dumps(best_models))

# Function to retrieve the best models for a given experiment
def get_best_models(exp_name):
    """
    Retrieves the best models (original, pruned, quantized) for a given experiment from Redis.
    """
    # Fetch the JSON string from Redis and parse it
    best_models = redis_client.get(exp_name)
    if best_models:
        return json.loads(best_models)
    return None

# Check if an experiment's best models are stored in Redis
def check_if_exists(exp_name):
    """
    Checks if best models data exists for the given experiment in Redis.
    """
    return redis_client.exists(exp_name)

# Remove the best models data for a given experiment from Redis
def evict_key(exp_name):
    """
    Evicts the best models data for the given experiment from Redis.
    """
    redis_client.delete(exp_name)
'''