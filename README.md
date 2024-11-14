# GenDL-MultiModal-AD

## Project Overview
**GenDL-MultiModal-AD** is a deep learning-based pipeline designed for Alzheimer's Disease diagnosis using multimodal data. This repository focuses on leveraging generative models for cross-modality translation, aimed at improving diagnostic accuracy by enhancing data with missing modalities.

## Key Features
- **Multimodal Data Processing**: Supports structural MRI and functional connectivity networks.
- **Generative Models for Data Imputation**: Uses Cycle Generative Adversarial Networks (CGAN) to generate missing modalities and improve robustness.
- **Cross-Modality Translation**: Provides cross-modality translation capabilities to facilitate data analysis with incomplete data.

## Project Structure
- **`models/`**: Defines the core deep learning models used, including VAEs and translation networks.
- **`config.py`**: Includes .
- **`trainer.py`**: Contains .
- **`main.py`**: 
- **`data_loader.py`**:
- **`utils.py/`**: Utility functions and helper scripts for data handling, visualization, and performance metrics.

├── models/                     # Contains model architectures (e.g., cGAN, VAE)
├── config.py                   # Configuration settings for training and evaluation
├── data_loader.py              # Data loading and preprocessing scripts
├── main.py                     # Main entry point for running experiments
├── trainer.py                  # Training loop and logging functions
├── utils.py                    # Utility functions (e.g., for logging, data handling)
└── README.md                   # Project documentation


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
