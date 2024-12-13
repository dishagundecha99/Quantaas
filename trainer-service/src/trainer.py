from datasets import load_metric
import numpy as np
import evaluate
import subprocess
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.utils.prune as prune
from optimum.pytorch import LayerPruner
import torch.quantization as quant


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
    
    def evaluate(self):
        self.eval_events = self.trainer.evaluate()
        print(self.eval_events)

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
    
    def get_metrics_eval(self):
        return self.eval_events
    
    def apply_prune()
        
    def apply_quant()
        


# Function to load model and tokenizer
def load_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return tokenizer, model

# Function for unstructured pruning (L1 Norm pruning)
def unstructured_pruning(model, amount=0.5):
    # Unstructured pruning of 50% of weights in all Linear layers
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            prune.l1_unstructured(module, name="weight", amount=amount)
    
    # Remove pruning masks to finalize pruning
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            prune.remove(module, "weight")
    return model

# Function for structured pruning (e.g., pruning 50% of heads in transformers)
def structured_pruning(model, amount=0.5):
    # Define the pruner (for pruning attention heads or other parameters)
    pruner = LayerPruner(
        model=model,
        pruning_method="head",  # Can be 'head' or other methods depending on the target structure
        amount=amount,
    )
    # Apply pruning
    pruner.prune()
    return model

# Function to quantize the model
def quantize_model(model, backend='fbgemm'):
    # First, prepare the model for quantization (fusion and observers)
    model.eval()  # Switch the model to eval mode for quantization
    model = torch.quantization.quantize_dynamic(
        model,
        {torch.nn.Linear},  # Apply dynamic quantization on Linear layers
        dtype=torch.qint8,  # Quantize to int8 precision
    )
    return model

# Function to get a pruned model
def get_pruned_model(, prune_amount=0.5):
    # Load the model and tokenizer
    tokenizer, model = load_model(model_name)
    
    # Perform unstructured pruning
    model = unstructured_pruning(model, amount=prune_amount)
    
    # Perform structured pruning (e.g., pruning transformer attention heads)
    model = structured_pruning(model, amount=prune_amount)
    
    # Save the pruned model
    model.save_pretrained("pruned_model")
    tokenizer.save_pretrained("pruned_model")
    
    return model, tokenizer

# Function to get a quantized model
def get_quantized_model(model_name="mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis"):
    # Load the model and tokenizer
    tokenizer, model = load_model(model_name)
    
    # Quantize the model (dynamic quantization)
    model = quantize_model(model)
    
    # Save the quantized model
    model.save_pretrained("quantized_model")
    tokenizer.save_pretrained("quantized_model")
    
    return model, tokenizer

# Function to test the model (pruned or quantized)
def test_model(model, tokenizer, sample_text="The company reported record profits this quarter!"):
    from transformers import pipeline
    pipe = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
    print(pipe(sample_text))

# Running the pruning and quantization process separately
# Get the pruned model
pruned_model, pruned_tokenizer = get_prune

from datasets import load_metric
import numpy as np
import evaluate
import subprocess
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.utils.prune as prune
from optimum.pytorch import LayerPruner
import torch.quantization as quant


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
    
    def evaluate(self):
        self.eval_events = self.trainer.evaluate()
        print(self.eval_events)

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
    
    def get_metrics_eval(self):
        return self.eval_events
    
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
        # Apply pruning
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


# Function to load model and tokenizer
def load_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return tokenizer, model

# Function to get a pruned model
def get_pruned_model(model_name="mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis", prune_amount=0.5):
    # Load the model and tokenizer
    tokenizer, model = load_model(model_name)
    
    # Initialize the CustomTrainer with the loaded model and tokenizer
    trainer = CustomTrainer(repo_name="model_repo", tokenizer=tokenizer, model=model, dataset=None)
    
    # Apply pruning
    model = trainer.apply_prune(prune_amount=prune_amount)
    
    # Save the pruned model
    model.save_pretrained("pruned_model")
    tokenizer.save_pretrained("pruned_model")
    
    return model, tokenizer

# Function to get a quantized model
def get_quantized_model(model_name="mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis"):
    # Load the model and tokenizer
    tokenizer, model = load_model(model_name)
    
    # Initialize the CustomTrainer with the loaded model and tokenizer
    trainer = CustomTrainer(repo_name="model_repo", tokenizer=tokenizer, model=model, dataset=None)
    
    # Apply quantization
    model = trainer.apply_quant()
    
    # Save the quantized model
    model.save_pretrained("quantized_model")
    tokenizer.save_pretrained("quantized_model")
    
    return model, tokenizer

# Function to test the model (pruned or quantized)
def test_model(model, tokenizer, sample_text="The company reported record profits this quarter!"):
    from transformers import pipeline
    pipe = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
    print(pipe(sample_text))

# Running the pruning and quantization process separately
# Get the pruned model
pruned_model, pruned_tokenizer = get_pruned_model(model_name="mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis", prune_amount=0.5)

# Test the pruned model
test_model(pruned_model, pruned_tokenizer)

# Get the quantized model
quantized_model, quantized_tokenizer = get_quantized_model(model_name="mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis")

# Test the quantized model
test_model(quantized_model, quantized_tokenizer)
