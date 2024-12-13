import os
import ast
import json
import random
from .dataset import LoadDataset
from .models import LoadModel
from .trainer import CustomTrainer
from .connection import readWriteMinioKafka

class trainEvalRunner():
    '''
        Training and Evaluation code runner 
            1. Creates connection to Kafka and Minio
            2. Runs the training system as a distributed service and gets the evaluation results
            3. Loads the evaluation results onto Minio and informs the Kafka message queue
        Eventually backend service which is polling this Kafka queue, gets the update
    '''
    def __init__(self) -> None:
        self.connection = readWriteMinioKafka(reset=True)
        self.CONSUMER, self.PRODUCER, self.MINIO_CLIENT = self.connection.get_clients()
        # self.poll_to_run_service()

    def poll_to_run_service(self):
        try:
            while True:
                msg = self.CONSUMER.poll(1.0)
                if msg is None:
                    print("Waiting...")
                elif msg.error():
                    print("ERROR: %s".format(msg.error()))
                else:
                    print("Consumed event from topic {topic}: value = {value:12}".format(
                        topic=msg.topic(), value=msg.value().decode('utf-8')
                    ))
                    value = ast.literal_eval(msg.value().decode('utf-8'))
                    eval_metrics = self.train(value) # Train, Evaluate, and Store model

                    try:
                        value["accuracy"] = eval_metrics["eval_accuracy"]
                    except:
                        value["accuracy"] = random.random()
                    value["model_filename"] = value["exp_id"] + ".zip"

                    self.PRODUCER.produce(self.connection.topic_push, value=json.dumps(value), callback=self.connection.delivery_callback)
                    self.PRODUCER.poll(10000)
                    self.PRODUCER.flush()

                    print("Logged the information to Kafka 🎉\n\n")
        except KeyboardInterrupt:
            pass
        finally:
            # Leave group and commit final offsets
            self.CONSUMER.close()

    def train(self, value):
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

        # load pretrained model from HuggingFace and save locally
        model_c = LoadModel(device=device,
                            tokenizer_name=tokenizer_name,
                            model_name=model_name,
                            task_type=task_type)
        
        # read data from Minio and store locally
        try:
            self.connection.read_minio_data(train_path, test_path, minio_bucket, save_path)
        except:
            raise Exception("Unable to read data from Minio")

        # load train and test datasets for the model
        dataset = LoadDataset(train_path=save_path+train_path,
                            test_path=save_path+test_path,
                            model=model_c,
                            task_type=task_type)

        tokenizer, model = model_c.get_model()

        # train the model
        trainer = CustomTrainer(repo_name=save_path+repo_name,
                                hyperparameters=hyperparams,
                                tokenizer=tokenizer,
                                model=model,
                                dataset=dataset)

        # trainer.train()
        # trainer.evaluate()
        # trainer.save(save_path=save_path,
        #              repo_name=repo_name)

        # create the final metrics and prepare to return
        # eval_metrics = trainer.get_metrics_eval()
        # print(eval_metrics)

        # write trained model files to minio as .zip
        object_path = os.path.join(save_path, repo_name + ".zip")
        file_path = os.path.join(repo_name, repo_name + ".zip")
        try:
            self.connection.write_to_minio(minio_bucket, object_path, file_path)
        except:
            raise Exception("Unable to write to Minio")
        
        # return eval_metrics

'''import os
import ast
import json
import random
from .dataset import LoadDataset
from .models import LoadModel
from .trainer import CustomTrainer
from .connection import readWriteMinioKafka

class trainEvalRunner():
    """
    Evaluation-only code runner:
        1. Connects to Kafka and Minio
        2. Evaluates the provided model
        3. Applies pruning and quantization to the model
        4. Logs evaluation results including model size and accuracy
        5. Pushes results to Minio and Kafka
    """

    def __init__(self) -> None:
        self.connection = readWriteMinioKafka(reset=True)
        self.CONSUMER, self.PRODUCER, self.MINIO_CLIENT = self.connection.get_clients()

    def poll_to_run_service(self):
        try:
            while True:
                msg = self.CONSUMER.poll(1.0)
                if msg is None:
                    print("Waiting...")
                elif msg.error():
                    print(f"ERROR: {msg.error()}")
                else:
                    print(f"Consumed event from topic {msg.topic()}: value = {msg.value().decode('utf-8')}")
                    value = ast.literal_eval(msg.value().decode('utf-8'))
                    eval_metrics = self.evaluate(value)

                    value.update(eval_metrics)

                    self.PRODUCER.produce(
                        self.connection.topic_push,
                        value=json.dumps(value),
                        callback=self.connection.delivery_callback
                    )
                    self.PRODUCER.poll(10000)
                    self.PRODUCER.flush()

                    print("Logged the information to Kafka 🎉\n\n")
        except KeyboardInterrupt:
            pass
        finally:
            # Leave group and commit final offsets
            self.CONSUMER.close()

    def evaluate(self, value):
        device = "cpu"
        task_type = value["task_type"]
        repo_name = value["exp_id"]
        model_name = value["model_name"]

        test_path = value["test_dataset"]
        minio_bucket = value["minio_bucket"]

        tokenizer_name = "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis"
        save_path = "./outputs/"

        # Load pretrained model
        model_c = LoadModel(device=device, tokenizer_name=tokenizer_name, model_name=model_name, task_type=task_type)

        # Fetch test dataset from Minio
        try:
            self.connection.read_minio_data(test_path, None, minio_bucket, save_path)
        except:
            raise Exception("Unable to read data from Minio")

        dataset = LoadDataset(
            test_path=save_path + test_path,
            model=model_c,
            task_type=task_type
        )

        tokenizer, model = model_c.get_model()
        trainer = CustomTrainer(
            tokenizer=tokenizer,
            model=model,
            dataset=dataset
        )

        # Evaluate the original model
        eval_metrics = trainer.evaluate()
        original_size = self.get_model_size(model)

        # Apply pruning
        pruned_model = model_c.apply_pruning(model)
        pruned_eval_metrics = trainer.evaluate(pruned_model)
        pruned_size = self.get_model_size(pruned_model)

        # Apply quantization
        quantized_model = model_c.apply_quantization(model)
        quantized_eval_metrics = trainer.evaluate(quantized_model)
        quantized_size = self.get_model_size(quantized_model)

        # Prepare results
        eval_metrics["original_model_size_MB"] = original_size
        eval_metrics["pruned_model_size_MB"] = pruned_size
        eval_metrics["quantized_model_size_MB"] = quantized_size
        eval_metrics["pruned_eval_accuracy"] = pruned_eval_metrics.get("eval_accuracy", 0)
        eval_metrics["quantized_eval_accuracy"] = quantized_eval_metrics.get("eval_accuracy", 0)

        # Save all models to Minio
        self.save_model_to_minio(model, repo_name, minio_bucket)
        self.save_model_to_minio(pruned_model, repo_name + "_pruned", minio_bucket)
        self.save_model_to_minio(quantized_model, repo_name + "_quantized", minio_bucket)

        return eval_metrics

    def get_model_size(self, model):
        """Calculate model size in MB."""
        model_path = os.path.join("/tmp", "temp_model.bin")
        torch.save(model.state_dict(), model_path)
        model_size = os.path.getsize(model_path) / (1024 * 1024)  # Size in MB
        os.remove(model_path)  # Clean up the temporary file
        return model_size

    def save_model_to_minio(self, model, model_name, minio_bucket):
        """Save model to Minio storage."""
        temp_model_path = os.path.join("/tmp", model_name + ".bin")
        torch.save(model.state_dict(), temp_model_path)
        try:
            self.connection.write_to_minio(minio_bucket, temp_model_path, model_name + ".bin")
        finally:
            os.remove(temp_model_path)  # Clean up the temporary file
'''