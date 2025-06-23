# Installation and Usage Guide

## Quick Start

### Option 1: Install as a Package (Recommended)

```bash
# Clone the repository
git clone https://github.com/juanfranceschini/whale-ltv-transformer.git
cd whale-ltv-transformer

# Install the package in development mode
pip install -e .

# Install additional dependencies for development
pip install -e ".[dev,notebooks]"
```

### Option 2: Direct Installation

```bash
# Clone the repository
git clone https://github.com/juanfranceschini/whale-ltv-transformer.git
cd whale-ltv-transformer

# Install dependencies
pip install -r requirements.txt

# For development tools
pip install pytest black isort flake8 mypy
```

## Data Setup

### 1. Download the Olist Dataset

The project uses the Brazilian E-Commerce (Olist) dataset. You need to download it from Kaggle:

1. Go to [Olist Brazilian E-Commerce Dataset](https://www.kaggle.com/olistbr/brazilian-ecommerce)
2. Download the dataset
3. Extract the CSV files to `data/raw/` directory

Required files:
- `olist_customers_dataset.csv`
- `olist_orders_dataset.csv`
- `olist_order_items_dataset.csv`
- `olist_order_payments_dataset.csv`
- `olist_products_dataset.csv`
- `olist_sellers_dataset.csv`
- `olist_geolocation_dataset.csv`
- `olist_order_reviews_dataset.csv`
- `product_category_name_translation.csv`

### 2. Prepare the Data

```bash
# Run data preparation
python -m src.data_prep

# Or use the console script (if installed as package)
whale-ltv-prepare-data
```

This will create `data/processed/customer_sequences.parquet` with the processed data.

## Training the Model

### 1. Basic Training

```bash
# Run training with default configuration
python -m src.train

# Or use the console script (if installed as package)
whale-ltv-train
```

### 2. Custom Configuration

```bash
# Train with custom configuration
python -m src.train training.max_epochs=50 model.d_model=256

# Use a different config file
python -m src.train --config-name=custom_config
```

### 3. Configuration Files

Configuration files are in `configs/`:
- `configs/transformer.yaml` - Default transformer configuration
- You can create custom configs for different experiments

## Using the Models

### 1. Import and Use Models

```python
# Import the main classes
from src import WhaleLTVTransformer, WhaleLTVDataModule, OlistDataProcessor

# Load processed data
data = pd.read_parquet('data/processed/customer_sequences.parquet')

# Create data module
datamodule = WhaleLTVDataModule(
    data_path='data/processed/customer_sequences.parquet',
    batch_size=32
)

# Initialize model
model = WhaleLTVTransformer(
    feature_dim=7,
    sequence_dim=3,
    max_sequence_length=10,
    d_model=128,
    nhead=8,
    num_layers=4
)

# Train model
trainer = pl.Trainer(max_epochs=10)
trainer.fit(model, datamodule)
```

### 2. Load Trained Model

```python
# Load from checkpoint
model = WhaleLTVTransformer.load_from_checkpoint(
    'models/checkpoints/whale_ltv_transformer_epoch=39_val_total_loss=0.1234.ckpt'
)

# Make predictions
predictions = model.predict(datamodule.test_dataloader())
```

### 3. Run Baselines

```python
from src.models.baselines import evaluate_baselines

# Evaluate baseline models
results = evaluate_baselines(
    X_train, y_train,
    X_val, y_val, 
    X_test, y_test,
    task="regression"  # or "classification"
)
```

## Evaluation and Analysis

### 1. Run Evaluation Script

```bash
python evaluate_model.py
```

### 2. Use Jupyter Notebooks

```bash
# Start Jupyter
jupyter notebook

# Open notebooks/
# - whale_ltv_transformer_report.ipynb (comprehensive analysis)
# - eval_whale_model.ipynb (evaluation)
```

### 3. Run Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src

# Run specific test
pytest tests/test_transformer.py
```

## Project Structure

```
whale-ltv-transformer/
├── src/                    # Main source code
│   ├── __init__.py        # Package exports
│   ├── data_prep.py       # Data processing
│   ├── train.py           # Training script
│   ├── utils.py           # Utility functions
│   └── models/            # Model implementations
│       ├── __init__.py    # Model exports
│       ├── transformer.py # Transformer model
│       ├── datamodule.py  # PyTorch Lightning data module
│       └── baselines.py   # Baseline models
├── configs/               # Configuration files
├── data/                  # Data directory
│   ├── raw/              # Original Olist dataset
│   └── processed/        # Processed data
├── notebooks/            # Jupyter notebooks
├── tests/               # Unit tests
├── scripts/             # Utility scripts
├── models/              # Trained models (created during training)
├── setup.py             # Package setup
├── requirements.txt     # Dependencies
└── README.md           # Project documentation
```

## Common Issues and Solutions

### Import Errors

If you get import errors like "No module named 'src.models'":

1. **Make sure you're in the project root directory**
2. **Install the package**: `pip install -e .`
3. **Use absolute imports**: `from src.models.transformer import WhaleLTVTransformer`

### Data Not Found

If you get "File not found" errors:

1. **Download the Olist dataset** to `data/raw/`
2. **Run data preparation**: `python -m src.data_prep`
3. **Check file paths** in configuration files

### CUDA/GPU Issues

If you have GPU issues:

1. **Install PyTorch with CUDA**: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`
2. **Set device in config**: `training.accelerator: "gpu"` or `training.accelerator: "cpu"`

### Memory Issues

If you run out of memory:

1. **Reduce batch size**: `data.batch_size: 16`
2. **Reduce model size**: `model.d_model: 64`
3. **Use gradient accumulation**: `training.accumulate_grad_batches: 2`

## Development

### Code Quality

```bash
# Format code
black src/ tests/

# Sort imports
isort src/ tests/

# Lint code
flake8 src/ tests/

# Type checking
mypy src/
```

### Adding New Features

1. **Create new modules** in `src/`
2. **Add tests** in `tests/`
3. **Update exports** in `src/__init__.py`
4. **Update documentation**

## Support

- **Issues**: [GitHub Issues](https://github.com/juanfranceschini/whale-ltv-transformer/issues)
- **Discussions**: [GitHub Discussions](https://github.com/juanfranceschini/whale-ltv-transformer/discussions)
- **Documentation**: Check the notebooks and docstrings in the code 