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

    #TO_DO these need to be added as we are also uploading pretrained model and for download we have 3 model dowbload options.

    '''
   def upload_pretrained_model(user_id, file):
    """
    Upload the pretrained model to the user's MinIO bucket.
    """
    user_bucket_name = get_user_bucket_name(user_id)
    prepare_user_bucket(user_bucket_name)
    
    # Secure the filename
    filename = secure_filename(file.filename)
    
    # Store the pretrained model in the "models/" folder
    file_path = f'models/{filename}'
    
    size = os.fstat(file.fileno()).st_size
    minio_client.put_object(user_bucket_name, file_path, file, size)

    def download_model(user_id, model_type, model_filename, local_file_path):
    """
    Download a specific model (pretrained, pruned, quantized) from MinIO to a local file.
    """
    user_bucket_name = get_user_bucket_name(user_id)
    
    # Define the model type path (pretrained, pruned, quantized)
    model_prefix = f'models/{model_type}/'
    
    # Full path to the model file in the bucket
    object_name = os.path.join(model_prefix, model_filename)
    
    # Download the model file from MinIO to the local path
    minio_client.fget_object(user_bucket_name, object_name, local_file_path)



'''