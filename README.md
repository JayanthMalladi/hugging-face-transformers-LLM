# hugging-face-transformers-LLM
# Transformer-FineTuning-Loop

A PyTorch-based training and evaluation loop for fine-tuning Hugging Face Transformer models on GLUE tasks (e.g., MRPC). This repository provides a modular and extensible script and notebook demonstrating custom training loops, learning rate scheduling, and evaluation metrics integration.

---

## 📁 Repository Structure


├── fullTrainingLoop.ipynb # Jupyter notebook with end-to-end code
├── train.py # Training script
├── eval.py # Evaluation script
├── requirements.txt # Python dependencies
├── data/ # Dataset loading and preprocessing
├── models/ # Model checkpoints and outputs
└── README.md # This file


---

## 🚀 Features

- **Custom Training Loop**  
  - Batch-wise data loading and collation  
  - Device management (`.to(device)` helper)  
  - Optimizer (e.g., AdamW) and linear learning rate decay scheduler  
  - Checkpointing and logging

- **Evaluation Loop**  
  - No-gradient inference (`torch.no_grad()`)  
  - Batch accumulation of predictions and metrics  
  - MRPC metrics: accuracy & F1

- **Interactive Notebook**  
  - `fullTrainingLoop.ipynb` for step-by-step exploration

---
