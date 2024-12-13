# Backend Service

The **Backend Service** is the core service built with Flask framework that facilitates interactions with users, submits jobs for model optimization, and manages model data. It is built using the **Flask** framework and is responsible for managing the user interactions such as submitting a job for pruning, quantization, and evaluation. The actual model optimization tasks (pruning, quantization, evaluation) are handled by the **Trainer Service**.

## Features

- **Job Submission**: Users can submit a job to apply pruning and quantization to pre-trained models.
- **Model Upload/Download**: Allows users to upload models to the platform and download optimized models.
- **Job Status Monitoring**: Monitors and provides the status of submitted jobs.
- **Authentication**: Manages user authentication and session handling.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/dishagundecha99/Quantaas.git
   cd Quantaas
