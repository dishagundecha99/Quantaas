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