from datasets import load_metric
import numpy as np
import evaluate
import subprocess


class CustomTrainer:
    """
    A class for evaluating and saving models, including original, pruned, and quantized versions.
    """
    def __init__(self, repo_name, tokenizer, model, dataset):
        self.repo_name = repo_name
        self.tokenizer = tokenizer
        self.model = model
        self.dataset = dataset
        self.rouge = evaluate.load("rouge")

        # Print for debugging
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

    def evaluate(self, model):
        """
        Evaluate a given model.
        """
        print(f"Evaluating model: {model}")
        # Create evaluation dataloader and perform evaluation
        eval_results = model.eval()  # Replace this with the actual evaluation process
        print(f"Evaluation Results: {eval_results}")
        return eval_results

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
