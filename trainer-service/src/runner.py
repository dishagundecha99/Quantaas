import os
import ast
import json
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
        self.save_path = "./outputs/"
        self.tokenizer_name = "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis"

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
            self.CONSUMER.close()

    def evaluate(self, value):
        device = "cpu"
        task_type = value["task_type"]
        repo_name = value["exp_id"]
        model_name = value["model_name"]
        test_path = value["test_dataset"]
        minio_bucket = value["minio_bucket"]

        # Load pretrained model
        model_loader = LoadModel(device=device, tokenizer_name=self.tokenizer_name, model_name=model_name, task_type=task_type)

        # Read test dataset from MinIO
        try:
            self.connection.read_minio_data(test_path, minio_bucket, self.save_path)
        except Exception as e:
            raise Exception(f"Unable to read data from MinIO: {e}")

        dataset = LoadDataset(test_path=os.path.join(self.save_path, test_path), model=model_loader, task_type=task_type)

        # Initialize trainer
        trainer = CustomTrainer(repo_name=repo_name, tokenizer=model_loader.tokenizer, model=model_loader.model, dataset=dataset)

        # Evaluate original model
        original_metrics = trainer.evaluate()
        original_size = model_loader.get_model_size()

        # Save original model
        original_filename = self.save_model_to_minio(model_loader.model, repo_name, minio_bucket, "original")

        # Apply pruning and evaluate
        pruned_model = model_loader.prune_model()
        pruned_metrics = trainer.evaluate(pruned_model)
        pruned_size = model_loader.get_model_size()
        pruned_filename = self.save_model_to_minio(pruned_model, repo_name, minio_bucket, "pruned")

        # Apply quantization and evaluate
        quantized_model = model_loader.quantize_model()
        quantized_metrics = trainer.evaluate(quantized_model)
        quantized_size = model_loader.get_model_size()
        quantized_filename = self.save_model_to_minio(quantized_model, repo_name, minio_bucket, "quantized")

        # Prepare and return results
        return {
            "original_accuracy": original_metrics["accuracy"],
            "original_size_MB": original_size,
            "original_model_filename": original_filename,
            "pruned_accuracy": pruned_metrics["accuracy"],
            "pruned_size_MB": pruned_size,
            "pruned_model_filename": pruned_filename,
            "quantized_accuracy": quantized_metrics["accuracy"],
            "quantized_size_MB": quantized_size,
            "quantized_model_filename": quantized_filename,
        }

    def save_model_to_minio(self, model, repo_name, minio_bucket, model_type):
        """
        Save the model to MinIO storage and return the filename.
        """
        model_filename = f"{repo_name}_{model_type}.bin"
        temp_model_path = os.path.join("/tmp", model_filename)

        # Save the model locally
        torch.save(model.state_dict(), temp_model_path)

        try:
            # Upload the model to MinIO
            self.connection.write_to_minio(minio_bucket, temp_model_path, model_filename)
            print(f"Saved {model_type} model to MinIO as {model_filename}")
        except Exception as e:
            raise Exception(f"Unable to write {model_type} model to MinIO: {e}")
        finally:
            # Clean up the temporary file
            os.remove(temp_model_path)

        return model_filename
