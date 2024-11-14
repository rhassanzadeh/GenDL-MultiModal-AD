# GenDL-MultiModal-AD

## Project Overview
**GenDL-MultiModal-AD** is a deep learning-based pipeline designed for Alzheimer's Disease diagnosis using multimodal data. This repository focuses on leveraging generative models for cross-modality translation, aimed at improving diagnostic accuracy by enhancing data with missing modalities.

## Key Features
- **Multimodal Data Processing**: Supports structural MRI and functional connectivity networks.
- **Generative Models for Data Imputation**: Uses Cycle Generative Adversarial Networks (CGAN) to generate missing modalities and improve robustness.
- **Cross-Modality Translation**: Provides cross-modality translation capabilities to facilitate data analysis with incomplete data.

## Project Structure
- **`models/`**: Contains model architectures and networks.
- **`config.py`**: Configuration settings for training and evaluation.
- **`trainer.py`**: Data loading and preprocessing scripts.
- **`main.py`**: Main entry point for running experiments.
- **`data_loader.py`**: Training loop and logging functions.
- **`utils.py/`**: Utility functions and helper scripts for data handling, visualization, and performance metrics.


## Installation and Setup
1. **Clone the repository**:
   ```bash
   git clone https://github.com/rhassanzadeh/GenDL-MultiModal-AD.git
   cd GenDL-MultiModal-AD

## Example Command
1. **To train a model, use the following command:**:
   ```bash
   python main.py --is_train True
2. **To test a model, use the following command:**:
   ```bash
   python main.py --is_train False
