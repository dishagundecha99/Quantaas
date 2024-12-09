# QuantTaaS

Project Proposal: QuantTaaS (Quantization
and Pruning as a Service)
#Team Members
• Nidhi Sankhe
• Disha Gundecha
#Project Goals
QuantTaaS aims to optimize deep learning models through quantization and pruning. By
enabling model compression with minimal manual effort, QuantTaaS is designed to make deep
learning models feasible to deploy on resource-constrained devices like IoT, mobile, and edge
devices. Our primary goals include:
1. Automate Model Compression:Provide a web based platform to upload trained models
and to optimize it by quantization and pruning.
2. Achieve Optimal Performance: Experiment with different compression levels to
balance model size, speed, and accuracy.
3. Real-Time Insights: Allow to visualize the impact of optimization strategies.
#Background
This project represents a real-world application of cloud resources to automate complex model
optimization tasks. It goes beyond simple cloud-based workflows, incorporating a multi-tiered
approach that enables both interactive user control and backend automation.
Using AI models for practical use cases is yet limited due to the lack of resources at the edge.
The project tries to fill a significant industry need for accessible model compression tools for
resource-constrained environments. The project is engaging for both developers and end-users,
with applications in sectors like IoT, mobile computing, and edge AI.
Software and Hardware Components
Software Components:
1. Frontend:
◦ React: For a responsive, intuitive UI that allows users to upload models, select
optimization parameters, and view results.
2. Backend:
◦ Flask: To manage user requests.
◦ Google Cloud Storage: For storing uploaded models and optimized model
artifacts.
◦ Google Cloud Functions: To execute specific tasks related to model processing,
including invoking model optimization workflows.
3. Model Optimization Libraries:
◦ TensorRT: For quantization and model optimization.
◦ OR/AND
◦ PyTorch Quantization Toolkit: For additional quantization and pruning
functions.
4. Experimentation Tools:
◦ Ray or Dask: For parallel experimentation with multiple pruning and
quantization configurations (We are yet checking if we can use something else or
stick to this one)
5. Visualization Tools:
◦ Matplotlib : To visualize the results.
Hardware Components:
1. Compute Resources:
◦ Google Cloud GPU Instances: To speed up model optimization processes,
particularly beneficial for larger models.
Architectural Diagram
Interaction of Software and Hardware Components
1. Frontend Interaction: Users interact with the React-based UI to upload models,
configure optimization parameters (such as pruning percentage and quantization level),
and monitor optimization results.
2. Backend and Cloud Integration:
◦ The Flask backend manages user requests and communicates with Google Cloud
Functions for on-demand model processing. It initiates quantization and pruning
workflows through TensorRT and/or PyTorch Quantization Toolkit.
◦ Google Cloud Storage serves as the central repository for storing both the original
and optimized model files.
3. Model Optimization:
◦ Model optimization workflows leverage TensorRT and PyTorch Quantization
Toolkit libraries on Google Cloud GPU instances for fast and efficient processing.
◦ Ray (or Dask) enables parallel execution of various pruning and quantization
configurations, enabling users to find the best balance between model size and
accuracy.
4. Evaluation and Visualization:
◦ The system measures and compares the model size and accuracy before and after
optimization.
◦ Matplotlib is used to generate visualizations.
Debugging Strategy
1. Logging and Error Tracking:
◦ Implement extensive logging within the Flask backend and Google Cloud
Functions to capture errors in real-time.
◦ Use Google Cloud Logging to monitor backend processes and quickly detect any
bottlenecks or failures in the model optimization workflows.
2. Testing Mechanisms:
◦ Unit Tests for individual functions within the Flask backend, particularly for
model upload, optimization, and result retrieval.
◦ Integration Testing to ensure all system components (frontend, backend, cloud
storage) work seamlessly.
