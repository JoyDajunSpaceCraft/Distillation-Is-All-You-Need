# Distillation-Is-All-You-Need

This project implements a process for distilling large language models (LLMs) using the T5 model. The process involves preparing data, followed by training with `train.py`.

## Project Structure

Below is the project's directory structure:

DIAYN/
+ data/ Prepared dataset for distillation
+ knowledge/
+ wandb/ # Weights & Biases tracking directory
+ config.json # Configuration settings
+ load_selfknowledge.py # Script for loading self-knowledge
+ MedDataLoader.py # Medical data loading utilities
+ metrix.py # Metrics and evaluation utilities
+ model_utils.py # Utilities for model handling
+ prepare_reason_data.py # Script to prepare data for distillation
+ train.py # Main training script for distillation



## How to Run

1. **Prepare the Data**: Start by running the `prepare_reason_data.py` script to generate necessary files for training.

   ```bash
   python prepare_reason_data.py
   ```
   If you have pre-generated files, ensure they are located in the data/csnlp directory.
   
2. **Training**: Once the data is ready, you can start the distillation process by running train.py.
   ```bash
   python train.py
   ```
   Make sure to check config.json for setting up the training configurations according to your requirements.

3. `config.json`: Contains all settings related to the model and training process. Modify according to the specifics of your task.
   
   
