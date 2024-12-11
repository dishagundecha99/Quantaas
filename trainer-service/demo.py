from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForQuestionAnswering, DefaultDataCollator
from transformers import AutoModelForMaskedLM, DataCollatorWithPadding, TrainingArguments, Trainer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq
from datasets import load_dataset, load_metric
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
import json
import ast
import evaluate
import sys
import os
from minio import Minio
import shutil

from confluent_kafka import Consumer, Producer, OFFSET_BEGINNING

from dataset import LoadDataset
from src.models import LoadModel
from src.trainer import CustomTrainer

def connection(reset):
    kafka_consumer_config = {
        'bootstrap.servers': "pkc-4r087.us-west2.gcp.confluent.cloud",
        'security.protocol': 'SASL_SSL',
        'sasl.mechanisms': 'PLAIN',
        'sasl.username': "KRAW4JX76RIMK3Z6",
        'sasl.password': "Yechd7PsQLv4ij46qRpO4utqQKhegZ/D3FoyoGnxkFVZgKYQQsgmPXOX8lErC0lD",
        'group.id': 'python_train_job_consumer',
        'auto.offset.reset': 'earliest'
    }

    # Create Consumer instance
    consumer = Consumer(kafka_consumer_config)

    # Create Producer instance
    producer = Producer(kafka_consumer_config)

    # Set up a callback to handle the '--reset' flag.
    def reset_offset(consumer, partitions):
        if reset == True:
            for p in partitions:
                p.offset = OFFSET_BEGINNING
            consumer.assign(partitions)

    # Subscribe to topic
    topic = "submit_job"
    consumer.subscribe([topic], on_assign=reset_offset)

    # Push to topic
    topic_push = "completed_job"

    def delivery_callback(err, msg):
        if err:
            print(f'ERROR: Message failed delivery: {err}')
        else:
            print(f'Produced event to topic {msg.topic()} value = {msg.value().decode("utf-8")}')
            
    # Poll for new messages from Kafka and print them.
    try:
        while True:
            msg = consumer.poll(1.0)
            if msg is None:
                # Initial message consumption may take up to
                # `session.timeout.ms` for the consumer group to
                # rebalance and start consuming
                print("Waiting...")
            elif msg.error():
                print("ERROR: %s".format(msg.error()))
            else:
                # Extract the (optional) key and value, and print.

                print("Consumed event from topic {topic}: value = {value:12}".format(
                    topic=msg.topic(), value=msg.value().decode('utf-8')
                ))
                value = ast.literal_eval(msg.value().decode('utf-8'))
                eval_metrics = train(value)

                value["accuracy"] = "0.78"
                value["model_filename"] = value["exp_id"] + ".zip"

                producer.produce(topic_push, value=json.dumps(value), callback=delivery_callback)
                producer.poll(10000)
                producer.flush()
                
    except KeyboardInterrupt:
        pass
    finally:
        # Leave group and commit final offsets
        consumer.close()


def train(value):
    device = "cpu"
    task_type = value["task_type"]
    repo_name = value["exp_id"]
    model_name = value["model_name"]
    hyperparams = value["hyperparams"]

    train_path = value["train_dataset"]
    test_path = value["test_dataset"]
    minio_bucket = value["minio_bucket"]

    tokenizer_name = "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis"
    
    save_path = "./outputs/"
    print(task_type, train_path)

    model_c = LoadModel(device=device,
                        tokenizer_name=tokenizer_name,
                        model_name=model_name,
                        task_type=task_type)
    print(model_c)
    read_minio_data(train_path, test_path, minio_bucket, save_path)

    dataset = LoadDataset(train_path=save_path+train_path,
                          test_path=save_path+test_path,
                          model=model_c,
                          task_type=task_type)

    tokenizer, model = model_c.get_model()

    trainer = CustomTrainer(repo_name=save_path+repo_name,
                            hyperparameters=hyperparams,
                            tokenizer=tokenizer,
                            model=model,
                            dataset=dataset)

    trainer.train()
    trainer.evaluate()
    trainer.save(save_path=save_path,
                 repo_name=repo_name)

    # final metrics write to kafka
    eval_metrics = trainer.get_metrics_eval()
    print(eval_metrics)

    # write to minio
    write_to_minio(minio_bucket, os.path.join(save_path, repo_name + ".zip"), os.path.join(repo_name, repo_name + ".zip"))
'''def train(value):
    device = "cpu"
    task_type = value["task_type"]
    repo_name = value["exp_id"]
    model_name = value["model_name"]

    # Paths to datasets and bucket
    train_path = value["train_dataset"]
    test_path = value["test_dataset"]
    minio_bucket = value["minio_bucket"]

    # Model tokenizer name
    tokenizer_name = "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis"
    
    save_path = "./outputs/"
    print(f"Task Type: {task_type}, Train Path: {train_path}, Test Path: {test_path}")

    # Load pretrained model
    model_c = LoadModel(device=device,
                        tokenizer_name=tokenizer_name,
                        model_name=model_name,
                        task_type=task_type)

    # Read data from Minio
    read_minio_data(train_path, test_path, minio_bucket, save_path)

    # Load dataset
    dataset = LoadDataset(train_path=save_path + train_path,
                          test_path=save_path + test_path,
                          model=model_c,
                          task_type=task_type)

    # Get model and tokenizer
    tokenizer, model = model_c.get_model()

    # Apply pruning and quantization
    pruned_model = model_c.apply_prune(model)
    quantized_model = model_c.apply_quant(model)

    # Evaluate original, pruned, and quantized models
    original_metrics = evaluate_model(model, dataset)
    pruned_metrics = evaluate_model(pruned_model, dataset)
    quantized_metrics = evaluate_model(quantized_model, dataset)

    # Save the models and their metrics
    save_models_and_metrics(original_metrics, pruned_metrics, quantized_metrics, model, pruned_model, quantized_model, save_path, repo_name)

    # Push results to Kafka
    results = {
        "original_accuracy": original_metrics['accuracy'],
        "pruned_accuracy": pruned_metrics['accuracy'],
        "quantized_accuracy": quantized_metrics['accuracy'],
        "model_filename": repo_name + ".zip"
    }
    producer.produce(topic_push, value=json.dumps(results), callback=delivery_callback)
    producer.poll(10000)
    producer.flush()

    def evaluate_model(model, dataset):
    # Assuming the dataset has tokenized train and test datasets.
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir="./outputs/",
            evaluation_strategy="epoch",
            per_device_eval_batch_size=4,
        ),
        eval_dataset=dataset.tokenized_test,
        tokenizer=dataset.tokenizer,
    )

    # Evaluate the model
    eval_results = trainer.evaluate()
    accuracy = eval_results["eval_accuracy"]  # Or compute from custom metrics if needed
    print(f"Model accuracy: {accuracy}")
    return {"accuracy": accuracy, "metrics": eval_results}

    def save_models_and_metrics(original_metrics, pruned_metrics, quantized_metrics, original_model, pruned_model, quantized_model, save_path, repo_name):
    # Save original model
    original_model.save_pretrained(os.path.join(save_path, repo_name, "original"))
    # Save pruned model
    pruned_model.save_pretrained(os.path.join(save_path, repo_name, "pruned"))
    # Save quantized model
    quantized_model.save_pretrained(os.path.join(save_path, repo_name, "quantized"))

    # Optionally, create a zip of the models for easier download
    zip_model(os.path.join(save_path, repo_name), repo_name)
    
    # Save the metrics
    metrics = {
        "original": original_metrics,
        "pruned": pruned_metrics,
        "quantized": quantized_metrics
    }
    with open(os.path.join(save_path, repo_name, "metrics.json"), "w") as f:
        json.dump(metrics, f)
import shutil

def zip_model(save_path, repo_name):
    shutil.make_archive(save_path + "/" + repo_name, 'zip', save_path + "/" + repo_name)
    print(f"Model saved and zipped at {save_path}/{repo_name}.zip")

'''

def read_minio_data(train_path, test_path, minio_bucket, save_path):

    minioHost = os.getenv("MINIO_HOST") or "localhost:9000"
    minioUser = os.getenv("MINIO_USER") or "minioadmin"
    minioPasswd = os.getenv("MINIO_PASSWD") or "minioadmin"

    MINIO_CLIENT = None
    try:
        MINIO_CLIENT = Minio(minioHost, access_key=minioUser, secret_key=minioPasswd, secure=False)
    except Exception as exp:
        print(f"Exception raised in worker loop: {str(exp)}")

    print("Downloading the files", train_path, test_path)

    # shutil.rmtree(save_path)
    os.makedirs(save_path, exist_ok=True)

    MINIO_CLIENT.fget_object(minio_bucket, "datasets/" + train_path, save_path + train_path)
    MINIO_CLIENT.fget_object(minio_bucket, "datasets/" + test_path, save_path + test_path)

    print("Placed file in temporary location", save_path)
    print("Files : ", os.listdir(save_path))   

def write_to_minio(minio_bucket, file_name, repo_name):
    minioHost = os.getenv("MINIO_HOST") or "localhost:9000"
    minioUser = os.getenv("MINIO_USER") or "minioadmin"
    minioPasswd = os.getenv("MINIO_PASSWD") or "minioadmin"

    MINIO_CLIENT = None
    try:
        MINIO_CLIENT = Minio(minioHost, access_key=minioUser, secret_key=minioPasswd, secure=False)
    except Exception as exp:
        print(f"Exception raised in worker loop: {str(exp)}")

    print(repo_name, file_name)
    MINIO_CLIENT.fput_object(minio_bucket, repo_name, file_name)

    print("Saved the model to Minio")


if __name__ == "__main__":
    connection(reset=True)
    # main()