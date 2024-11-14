# GenDL-MultiModal-AD

## Project Overview
**GenDL-MultiModal-AD** is a deep learning-based pipeline designed for Alzheimer's Disease diagnosis using multimodal data. This repository focuses on leveraging generative models for cross-modality translation, aimed at improving diagnostic accuracy by enhancing data with missing modalities.

## Key Features
- **Multimodal Data Processing**: Supports structural MRI and functional connectivity networks.
- **Generative Models for Data Imputation**: Uses Variational Autoencoders (VAEs) to generate missing modalities and improve robustness.
- **Cross-Modality Translation**: Provides cross-modality translation capabilities to facilitate data analysis with incomplete data.

## Project Structure
- **`data/`**: Contains data processing scripts and sample data.
- **`models/`**: Defines the core deep learning models used, including VAEs and translation networks.
- **`scripts/`**: Includes training, evaluation, and preprocessing scripts.
- **`utils/`**: Utility functions and helper scripts for data handling, visualization, and performance metrics.

## Installation and Setup
1. **Clone the repository**:
   ```bash
   git clone https://github.com/rhassanzadeh/GenDL-MultiModal-AD.git
   cd GenDL-MultiModal-AD
