from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
import os
import json
import ast
import shutil
from minio import Minio
from confluent_kafka import Consumer, Producer, OFFSET_BEGINNING
from src.models import LoadModel
from dataset import LoadDataset

def connection(reset):
    kafka_consumer_config = {
        'bootstrap.servers': "pkc-4r087.us-west2.gcp.confluent.cloud",
        'security.protocol': 'SASL_SSL',
        'sasl.mechanisms': 'PLAIN',
        'sasl.username': "KRAW4JX76RIMK3Z6",
        'sasl.password': "Yechd7PsQLv4ij46qRpO4utqQKhegZ/D3FoyoGnxkFVZgKYQQsgmPXOX8lErC0lD",
        'group.id': 'python_eval_job_consumer',
        'auto.offset.reset': 'earliest'
    }

    # Create Consumer instance
    consumer = Consumer(kafka_consumer_config)
    producer = Producer(kafka_consumer_config)

    def reset_offset(consumer, partitions):
        if reset:
            for p in partitions:
                p.offset = OFFSET_BEGINNING
            consumer.assign(partitions)

    consumer.subscribe(["submit_job"], on_assign=reset_offset)
    topic_push = "completed_job"

    def delivery_callback(err, msg):
        if err:
            print(f'ERROR: Message failed delivery: {err}')
        else:
            print(f'Produced event to topic {msg.topic()} value = {msg.value().decode("utf-8")}')
            
    try:
        while True:
            msg = consumer.poll(1.0)
            if msg is None:
                print("Waiting...")
            elif msg.error():
                print(f"ERROR: {msg.error()}")
            else:
                value = ast.literal_eval(msg.value().decode('utf-8'))
                eval_results = evaluate_models(value)
                
                value.update(eval_results)
                producer.produce(topic_push, value=json.dumps(value), callback=delivery_callback)
                producer.poll(10000)
                producer.flush()
    except KeyboardInterrupt:
        pass
    finally:
        consumer.close()

def evaluate_models(value): #train(value)
    device = "cpu"
    task_type = value["task_type"]
    repo_name = value["exp_id"]
    model_name = value["model_name"]
    test_path = value["test_dataset"]
    minio_bucket = value["minio_bucket"]
    save_path = "./outputs/"
    tokenizer_name = "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis"

    # Load model and dataset
    model_c = LoadModel(device=device, tokenizer_name=tokenizer_name, model_name=model_name, task_type=task_type)
    read_minio_data(test_path, minio_bucket, save_path)
    dataset = LoadDataset(test_path=save_path + test_path, model=model_c, task_type=task_type)

    tokenizer, model = model_c.get_model()

    # Prune and quantize models
    pruned_model = model_c.apply_prune(model)
    quantized_model = model_c.apply_quant(model)

    # Evaluate models
    original_metrics = evaluate_model(model, dataset, tokenizer)
    pruned_metrics = evaluate_model(pruned_model, dataset, tokenizer)
    quantized_metrics = evaluate_model(quantized_model, dataset, tokenizer)

    # Save models
    save_models_and_metrics(original_metrics, pruned_metrics, quantized_metrics, model, pruned_model, quantized_model, save_path, repo_name)

    return {
        "original_accuracy": original_metrics["accuracy"],
        "pruned_accuracy": pruned_metrics["accuracy"],
        "quantized_accuracy": quantized_metrics["accuracy"],
        "model_filename": repo_name + ".zip"
    }

def evaluate_model(model, dataset, tokenizer):
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir="./outputs/",
            evaluation_strategy="no",
            per_device_eval_batch_size=4
        ),
        eval_dataset=dataset.tokenized_test,
        tokenizer=tokenizer
    )
    eval_results = trainer.evaluate()
    accuracy = eval_results.get("eval_accuracy", None)
    print(f"Model accuracy: {accuracy}")
    return {"accuracy": accuracy, "metrics": eval_results}

def save_models_and_metrics(original_metrics, pruned_metrics, quantized_metrics, original_model, pruned_model, quantized_model, save_path, repo_name):
    os.makedirs(os.path.join(save_path, repo_name), exist_ok=True)
    original_model.save_pretrained(os.path.join(save_path, repo_name, "original"))
    pruned_model.save_pretrained(os.path.join(save_path, repo_name, "pruned"))
    quantized_model.save_pretrained(os.path.join(save_path, repo_name, "quantized"))

    zip_model(os.path.join(save_path, repo_name), repo_name)
    metrics = {"original": original_metrics, "pruned": pruned_metrics, "quantized": quantized_metrics}
    with open(os.path.join(save_path, repo_name, "metrics.json"), "w") as f:
        json.dump(metrics, f)

def zip_model(save_path, repo_name):
    shutil.make_archive(save_path + "/" + repo_name, 'zip', save_path)
    print(f"Model saved and zipped at {save_path}/{repo_name}.zip")

def read_minio_data(test_path, minio_bucket, save_path):
    minioHost = os.getenv("MINIO_HOST") or "localhost:9000"
    minioUser = os.getenv("MINIO_USER") or "minioadmin"
    minioPasswd = os.getenv("MINIO_PASSWD") or "minioadmin"

    MINIO_CLIENT = Minio(minioHost, access_key=minioUser, secret_key=minioPasswd, secure=False)
    os.makedirs(save_path, exist_ok=True)

    MINIO_CLIENT.fget_object(minio_bucket, "datasets/" + test_path, save_path + test_path)

    print("Files placed in temporary location:", save_path)

if __name__ == "__main__":
    connection(reset=True)
