# 🐋 Whale LTV Transformer

> **Early Customer Lifetime Value Prediction and Whale Identification using Transformer Architecture**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📖 Overview

The Whale LTV Transformer is an advanced machine learning model that predicts customer lifetime value (LTV) and identifies high-value customers ("whales") from early customer behavior patterns. Built on the Brazilian E-Commerce (Olist) dataset, this model uses a Transformer architecture to capture sequential patterns in customer purchasing behavior.

### 🎯 Key Features

- **Early LTV Prediction**: Predict 90-day customer value from first 14 days of behavior
- **Whale Identification**: Identify top 5% high-value customers with 93.6% accuracy
- **Sequential Pattern Recognition**: Transformer architecture captures temporal dependencies
- **Joint Training**: Simultaneous LTV regression and whale classification
- **Production Ready**: PyTorch Lightning framework with comprehensive testing

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/whale-ltv-transformer.git
cd whale-ltv-transformer

# Install dependencies
pip install -r requirements.txt

# Download the Olist dataset
python scripts/download_data.py
```

### Data Preparation

```bash
# Prepare the dataset
python scripts/prepare_data.py
```

### Training

```bash
# Train the model
python scripts/train.py
```

### Evaluation

```bash
# Run evaluation
python scripts/evaluate.py

# Or open the Jupyter notebook
jupyter notebook notebooks/whale_ltv_evaluation.ipynb
```

## 📁 Project Structure

```
whale-ltv-transformer/
├── data/
│   ├── raw/                 # Original Olist dataset
│   └── processed/           # Processed data files
├── src/
│   ├── data/               # Data processing modules
│   ├── models/             # Model architectures
│   ├── training/           # Training scripts
│   └── utils/              # Utility functions
├── notebooks/              # Jupyter notebooks
├── tests/                  # Unit tests
├── docs/                   # Documentation
├── scripts/                # Execution scripts
└── results/                # Model outputs and results
```

## 🏗️ Model Architecture

### Core Components

1. **Event Embedding Layer**: Embeds order value, temporal, and rank features
2. **Transformer Encoder**: Multi-head attention with 4 layers and 8 attention heads
3. **Joint Output Heads**: Simultaneous LTV regression and whale classification
4. **Customer Feature Integration**: Combines sequence and customer-level features

### Key Features

- **Sequential Processing**: Captures temporal patterns in customer behavior
- **Attention Mechanism**: Provides interpretability through attention weights
- **Multi-task Learning**: Joint optimization improves both tasks
- **Scalable Design**: Handles variable-length sequences efficiently

## 📊 Dataset

### Olist Brazilian E-Commerce Dataset

- **Source**: [Kaggle Olist Dataset](https://www.kaggle.com/olistbr/brazilian-ecommerce)
- **Size**: ~100K customers, ~1M orders
- **Features**: Order history, customer demographics, product categories
- **Target**: 90-day customer lifetime value

### Data Processing

1. **Order Aggregation**: Calculate total order values (price + freight)
2. **Sequence Creation**: Create customer order sequences with temporal features
3. **Feature Engineering**: Extract customer-level features from sequences
4. **Target Calculation**: Compute 90-day LTV and whale classification

## 🛠️ Development

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src

# Run specific test file
pytest tests/test_transformer.py
```

### Code Quality

```bash
# Format code
black src/ tests/

# Lint code
flake8 src/ tests/

# Type checking
mypy src/
```

## 📊 Evaluation Notebooks

- **[Comprehensive Evaluation](notebooks/whale_ltv_evaluation.ipynb)** - Complete model analysis and visualization
- **[Baseline Comparison](notebooks/baseline_comparison.ipynb)** - Model vs. traditional ML approaches

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Set up pre-commit hooks
pre-commit install
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Contact

- **Project Link**: [https://github.com/juanfranceschini/whale-ltv-transformer](https://github.com/juanfranceschini/whale-ltv-transformer)

---

**⭐ Star this repository if you find it useful!**

*Open Source ❤️* 