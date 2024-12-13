# **QuantTaaS: Quantization and Pruning as a Service**

## **Project Contributors**

**Final Project Team 92**

- Nidhi Sankhe
- Disha Gundecha
  
## **Project Overview**

**QuantTaaS** (Quantization and Pruning as a Service) is a cloud-based platform designed to optimize deep learning models by applying quantization and pruning techniques. The service allows users to upload pre-trained models and apply model compression strategies to reduce their size and improve deployment on resource-constrained environments, such as IoT devices, mobile platforms, and edge devices.

The primary goal of **QuantTaaS** is to automate model optimization with minimal manual effort, providing real-time insights into the impact of various optimization strategies on model performance. Through the service, users can upload pre-trained models, run optimizations (quantization and pruning), and evaluate the resulting models based on accuracy and size.

## **Key Features**

- **Pre-trained Model Upload**: Users can upload pre-trained models to the platform for optimization.
- **Pruning**: The model can be pruned to reduce unnecessary weights, enhancing both efficiency and speed.
- **Quantization**: The model can be quantized to reduce precision, making it more suitable for deployment on edge devices.
- **Model Evaluation**: Post-optimization, the platform evaluates the accuracy and model size of the original, pruned, and quantized models on a provided test dataset.
- **Real-Time Visualization**: Users can visualize the effects of pruning and quantization on the model’s accuracy and size.
- **Seamless Integration with MinIO and Kafka**: Results, including models, are stored in **MinIO** and the job status is tracked through **Kafka**.

## **Technologies Used**

### **Software Components:**

- **Frontend**: React (for a responsive, intuitive user interface)
- **Backend**: Flask (to handle user requests and API endpoints)
- **Cloud Storage**: MinIO (for storing models and results)
- **Messaging Queue**: Kafka (for job status tracking and notifications)
- **Model Optimization Libraries**: PyTorch (for pruning, quantization, and model evaluation)
- **Database**: MongoDB (for storing job and model metadata, including optimization details and performance metrics)
- **Experimentation Tools**: Ray/Dask (for parallel execution of pruning and quantization configurations)
- **Visualization**: Matplotlib (to visualize model performance changes after optimization)

### **Hardware Components:**

- **Cloud CPU Instances**: For model optimization tasks
- **Cloud GPU Instances**: For model optimization tasks that require high computational power.

## **Project Architecture**

The **QuantTaaS** architecture is divided into distinct services that work together to deliver the entire model optimization and evaluation pipeline:

1. **Frontend**: 
   - The frontend, built using **React**, provides an intuitive interface for users to upload pre-trained models, configure optimization settings (e.g., pruning percentage, quantization level), and visualize the results.
   
2. **Backend**:
   - The **Flask backend** manages user requests and coordinates tasks with other services.
   - Upon receiving requests, the backend triggers the model optimization process, which includes:
     - Loading the pre-trained model.
     - Applying pruning (both unstructured and structured) and quantization to the model.
     - Evaluating the model before and after optimization based on the user-provided test dataset.
   
3. **Trainer Service**:
   - This is the core service that handles:
     - **Model pruning and quantization**: Applies model compression techniques to reduce model size.
     - **Model evaluation**: Evaluates the model's performance, including accuracy and size.
     - **Saving models**: Stores the models in **MinIO** (both original, pruned, and quantized versions) and tracks their performance.
   
4. **Job Status Service**:
   - This service monitors the job status using **Kafka**, keeping track of the progress of each model optimization task. It sends updates to the frontend once the tasks are complete.
   
5. **MinIO Storage**:
   - **MinIO** is used for storing models and their evaluation results. The service saves the original, pruned, and quantized models along with their performance metrics.

6. **Kafka**:
   - **Kafka** is used for job tracking and status updates. It acts as a message queue that receives and sends job completion updates to the backend and frontend.

## **How to Deploy the Project**

### **Prerequisites**

Before deploying the project, ensure that you have the following setup:

- **MinIO**: For model storage and access.
- **Kafka**: For job status management.
- **Python 3.x**: For backend and model optimization tasks.
- **Docker**: For containerizing the services.
- **Google Cloud Account (optional)**: For GPU instances (if you want to run model optimization on cloud GPUs).
  
### **Deployment Steps**

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/dishagundecha99/Quantaas.git
   cd Quantaas

