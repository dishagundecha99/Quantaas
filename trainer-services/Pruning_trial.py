from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load the model and tokenizer
model_name = "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

import torch
import torch.nn.utils.prune as prune
#unstructured pruning
# Prune 50% of weights in all Linear layers
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Linear):
        prune.l1_unstructured(module, name="weight", amount=0.5)

# Remove pruning masks to finalize pruning
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Linear):
        prune.remove(module, "weight")

#structured pruning
from optimum.pytorch import LayerPruner

# Define the pruner
pruner = LayerPruner(
    model=model,
    pruning_method="head",
    amount=0.5,  # Prune 50% of heads
)

# Apply pruning
pruner.prune()

from transformers import pipeline

# Create a sentiment analysis pipeline
pipe = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

# Test on sample data
sample_text = "The company reported record profits this quarter!"
print(pipe(sample_text))

model.save_pretrained("pruned_model")
tokenizer.save_pretrained("pruned_model")
