import os
import ast
import json
from .dataset import LoadDataset
from .models import LoadModel
from .trainer import CustomTrainer, run_job
from .connection import readWriteMinioKafka
from .mongo_util import update_mongo_run  # MongoDB utility to store results

class trainEvalRunner():
    """
    Evaluation and Model Processing Runner:
        1. Connects to Kafka and Minio.
        2. Loads and processes the provided model.
        3. Applies pruning and quantization to the model.
        4. Evaluates original, pruned, and quantized models.
        5. Saves the results to Minio and MongoDB.
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
                    eval_metrics = self.run_job(value)  # Call to the run_job() in trainer.py

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

    def run_job(self, value):
        # Here we call trainer.py's method for pruning, quantization, evaluation, and saving the models
        from .trainer import run_job  # Assuming run_job handles all of the operations
        eval_metrics = run_job(value)  # This will handle pruning, quantization, evaluation, saving to MinIO, etc.
        
        # After running the job, we can return the final metrics to store in MongoDB
        return eval_metrics

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
