# Trainer Service

The **Trainer Service** handles the fine-tuning, evaluation, pruning, and quantization of models. It takes a pre-trained HuggingFace model and applies various optimizations (such as pruning and quantization) to make the model more efficient for deployment in resource-constrained environments. The service also evaluates the performance of the original, pruned, and quantized models and saves them to the storage (MinIO and MongoDB).

## Features

- **Model Fine-tuning**: Fine-tune models on user-defined datasets.
- **Model Evaluation**: Evaluate the original, pruned, and quantized models based on accuracy and size.
- **Pruning**: Apply unstructured and structured pruning to reduce model size by removing unnecessary weights or parts of the network.
- **Quantization**: Apply dynamic quantization to reduce the precision of model weights and improve computational efficiency.
- **Model Saving**: Save models (original, pruned, quantized) and compress them into ZIP files.
- **Result Storage**: Store the evaluation results and models in MinIO and MongoDB.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/dishagundecha99/Quantaas.git
   cd Quantaas
