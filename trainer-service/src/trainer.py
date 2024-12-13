import os
import torch
import random
from .model import LoadModel
from .dataset import LoadDataset
from .connection import readWriteMinioKafka
from transformers import pipeline
import torch.nn.utils.prune as prune
from optimum.pytorch import LayerPruner
import torch.quantization as quant
import evaluate
import numpy as np
import json
from datasets import load_metric
import subprocess


class CustomTrainer:
    '''
    Trainer for model pruning, quantization, evaluation, and saving results to Minio and MongoDB
    '''
    def __init__(self, repo_name, tokenizer, model, dataset):
        self.repo_name = repo_name
        self.tokenizer = tokenizer
        self.model = model
        self.dataset = dataset
        self.rouge = evaluate.load("rouge")
        print("Model Evaluator initialized.")

    def compute_metrics(self, eval_pred):
        """
        Compute accuracy and F1 score for classification tasks.
        """
        load_accuracy = load_metric("accuracy")
        load_f1 = load_metric("f1")

        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        accuracy = load_accuracy.compute(predictions=predictions, references=labels)["accuracy"]
        f1 = load_f1.compute(predictions=predictions, references=labels)["f1"]
        return {"accuracy": accuracy, "f1": f1}

    def compute_metrics_summarization(self, eval_pred):
        """
        Compute ROUGE metrics for summarization tasks.
        """
        predictions, labels = eval_pred
        decoded_preds = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        result = self.rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)

        prediction_lens = [np.count_nonzero(pred != self.tokenizer.pad_token_id) for pred in predictions]
        result["gen_len"] = np.mean(prediction_lens)
        return {k: round(v, 4) for k, v in result.items()}

    def evaluate(self):
        """
        Evaluate the model.
        """
        self.eval_events = self.trainer.evaluate()
        print(self.eval_events)

    def evaluate_all_models(self, pruned_model, quantized_model):
        """
        Evaluate the original, pruned, and quantized models.
        """
        # Original model evaluation
        print("Evaluating Original Model...")
        original_metrics = self.evaluate(self.model)

        # Pruned model evaluation
        print("Evaluating Pruned Model...")
        pruned_metrics = self.evaluate(pruned_model)

        # Quantized model evaluation
        print("Evaluating Quantized Model...")
        quantized_metrics = self.evaluate(quantized_model)

        return original_metrics, pruned_metrics, quantized_metrics

    def apply_prune(self, prune_amount=0.5):
        """
        Apply pruning to the model.
        """
        # Unstructured pruning (L1 Norm pruning)
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Linear):
                prune.l1_unstructured(module, name="weight", amount=prune_amount)
        
        # Remove pruning masks to finalize pruning
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Linear):
                prune.remove(module, "weight")
        
        # Structured pruning (e.g., pruning 50% of heads in transformers)
        pruner = LayerPruner(
            model=self.model,
            pruning_method="head",  # Can be 'head' or other methods depending on the target structure
            amount=prune_amount,
        )
        pruner.prune()
        print("Pruning applied successfully.")
        return self.model

    def apply_quant(self):
        """
        Apply quantization to the model.
        """
        # First, prepare the model for quantization (fusion and observers)
        self.model.eval()  # Switch the model to eval mode for quantization
        self.model = torch.quantization.quantize_dynamic(
            self.model,
            {torch.nn.Linear},  # Apply dynamic quantization on Linear layers
            dtype=torch.qint8,  # Quantize to int8 precision
        )
        print("Quantization applied successfully.")
        return self.model

    def save_model(self, model, save_path, model_type):
        """
        Save a given model and compress it into a ZIP file.
        """
        folder_name = f"{save_path}/{self.repo_name}_{model_type}"
        model.save_pretrained(folder_name)
        print(f"{model_type} model saved at {folder_name}")

        # Zip the model
        zip_args = ["zip", f"{folder_name}.zip", folder_name + "/*"]
        subprocess.Popen(zip_args)
        print(f"{model_type} model zipped")

    def save_all_models(self, pruned_model, quantized_model, save_path):
        """
        Save the original, pruned, and quantized models.
        """
        self.save_model(self.model, save_path, "original")
        self.save_model(pruned_model, save_path, "pruned")
        self.save_model(quantized_model, save_path, "quantized")

    def get_model_size(self, model):
        """Calculate model size in MB."""
        model_path = os.path.join("/tmp", "temp_model.bin")
        torch.save(model.state_dict(), model_path)
        model_size = os.path.getsize(model_path) / (1024 * 1024)  # Size in MB
        os.remove(model_path)  # Clean up the temporary file
        return model_size


# Main function to load model, apply pruning, quantization, evaluate, and save results
def run_job(value):
    device = "cpu"
    task_type = value["task_type"]
    repo_name = value["exp_id"]
    model_name = value["model_name"]
    test_path = value["test_dataset"]
    minio_bucket = value["minio_bucket"]

    tokenizer_name = "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis"
    save_path = "./outputs/"

    # Load pretrained model from HuggingFace and save locally
    model_loader = LoadModel(device=device, tokenizer_name=tokenizer_name, model_name=model_name, task_type=task_type)
    
    # Read data from Minio and store locally
    connection = readWriteMinioKafka(reset=True)
    connection.read_minio_data(test_path, minio_bucket, save_path)

    # Load train and test datasets for the model
    dataset = LoadDataset(test_path=save_path + test_path, model=model_loader, task_type=task_type)

    tokenizer, model = model_loader.get_model()

    # Initialize CustomTrainer
    trainer = CustomTrainer(repo_name=save_path + repo_name,
                            tokenizer=tokenizer,
                            model=model,
                            dataset=dataset)

    # Apply pruning and quantization
    pruned_model = trainer.apply_prune()
    quantized_model = trainer.apply_quant()

    # Evaluate the models
    original_size = trainer.get_model_size(model)
    pruned_size = trainer.get_model_size(pruned_model)
    quantized_size = trainer.get_model_size(quantized_model)

    # Save the models
    trainer.save_all_models(pruned_model, quantized_model, save_path)

    # Collect evaluation results
    eval_results = {
        "original_model_size_MB": original_size,
        "pruned_model_size_MB": pruned_size,
        "quantized_model_size_MB": quantized_size
    }

    return eval_results
