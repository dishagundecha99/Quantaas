from backend_server import constants
from werkzeug.utils import secure_filename
from minio import Minio
import hashlib
import os

minio_client = None
try:
    minio_client = Minio(constants.minioHost, access_key=constants.minioUser, secret_key=constants.minioPasswd, secure=False)
except Exception as exp:
    print(f'Exception raised in while starting minio: {str(exp)}')

def get_user_bucket_name(user_id):
    return f'{user_id.replace("@",".")}'

def prepare_user_bucket(user_bucket_name):
    if minio_client.bucket_exists(user_bucket_name):
        return
    minio_client.make_bucket(user_bucket_name)

def upload_dataset(user_id, file):
    user_bucket_name =  get_user_bucket_name(user_id)
    prepare_user_bucket(user_bucket_name)
    filename = secure_filename(file.filename)
    user_bucket = get_user_bucket_name(user_id)
    size = os.fstat(file.fileno()).st_size
    minio_client.put_object(user_bucket, f'datasets/{filename}', file, size)

def download_model_file(bucket, object_name, file_path):
    minio_client.fget_object(bucket, object_name, file_path)