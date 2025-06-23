#!/usr/bin/env python3
"""
Example usage of the Whale LTV Transformer package.

This script demonstrates how to:
1. Import and use the main classes
2. Load and process data
3. Train a model
4. Make predictions
"""

import sys
import os
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import pandas as pd
import torch
import pytorch_lightning as pl
from sklearn.metrics import mean_squared_error, roc_auc_score

# Import our package modules
try:
    from src import (
        WhaleLTVTransformer, 
        WhaleLTVDataModule, 
        OlistDataProcessor,
        extract_features,
        split_data
    )
    print("✅ Successfully imported all modules!")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're in the project root directory and have installed the package:")
    print("pip install -e .")
    sys.exit(1)

def main():
    """Main example function."""
    print("🐋 Whale LTV Transformer - Example Usage")
    print("=" * 50)
    
    # Check if processed data exists
    data_path = Path("data/processed/customer_sequences.parquet")
    if not data_path.exists():
        print("❌ Processed data not found!")
        print("Please run data preparation first:")
        print("python -m src.data_prep")
        return
    
    # Load data
    print("\n📊 Loading processed data...")
    data = pd.read_parquet(data_path)
    print(f"Loaded {len(data)} customer sequences")
    print(f"Whale percentage: {data['is_whale'].mean():.1%}")
    
    # Create data module
    print("\n🔧 Setting up data module...")
    datamodule = WhaleLTVDataModule(
        data_path=str(data_path),
        batch_size=32,
        max_sequence_length=10,
        test_size=0.2,
        val_size=0.1,
        random_state=42,
        normalize_ltv=True
    )
    
    # Setup data module
    datamodule.setup()
    print(f"Train size: {len(datamodule.train_dataset)}")
    print(f"Val size: {len(datamodule.val_dataset)}")
    print(f"Test size: {len(datamodule.test_dataset)}")
    
    # Initialize model
    print("\n🤖 Initializing model...")
    model = WhaleLTVTransformer(
        feature_dim=datamodule.get_feature_dim(),
        sequence_dim=datamodule.get_sequence_dim(),
        max_sequence_length=datamodule.get_max_sequence_length(),
        d_model=128,
        nhead=8,
        num_layers=4,
        dim_feedforward=512,
        dropout=0.1,
        ltv_weight=1.0,
        learning_rate=0.001,
        weight_decay=0.01
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Quick training (just a few epochs for demonstration)
    print("\n🚀 Training model (demo mode - 3 epochs)...")
    trainer = pl.Trainer(
        max_epochs=3,
        accelerator="auto",
        devices=1,
        enable_progress_bar=True,
        enable_checkpointing=False,
        logger=False
    )
    
    trainer.fit(model, datamodule)
    
    # Test the model
    print("\n🧪 Testing model...")
    test_results = trainer.test(model, datamodule)
    
    print("\n📈 Test Results:")
    for metric, value in test_results[0].items():
        print(f"  {metric}: {value:.4f}")
    
    # Demonstrate prediction
    print("\n🔮 Making predictions...")
    model.eval()
    
    # Get a batch from test set
    test_batch = next(iter(datamodule.test_dataloader()))
    
    with torch.no_grad():
        predictions = model(test_batch)
    
    print(f"Prediction shapes:")
    print(f"  LTV predictions: {predictions['ltv'].shape}")
    print(f"  Whale predictions: {predictions['whale'].shape}")
    
    # Show sample predictions
    print(f"\nSample predictions (first 5):")
    print(f"  LTV: {predictions['ltv'][:5].flatten().tolist()}")
    print(f"  Whale prob: {torch.sigmoid(predictions['whale'][:5]).flatten().tolist()}")
    
    # Demonstrate baseline usage
    print("\n📊 Running baseline comparison...")
    try:
        from src.models.baselines import evaluate_baselines
        
        # Extract features for baselines
        features = extract_features(data)
        train_data, val_data, test_data = split_data(data, test_size=0.2, val_size=0.1, random_state=42)
        
        X_train = extract_features(train_data)
        X_val = extract_features(val_data)
        X_test = extract_features(test_data)
        
        # Run LTV regression baseline
        ltv_results = evaluate_baselines(
            X_train, train_data['ltv_90d'].values,
            X_val, val_data['ltv_90d'].values,
            X_test, test_data['ltv_90d'].values,
            task="regression"
        )
        
        print("LTV Regression Baselines:")
        for model_name, metrics in ltv_results.items():
            print(f"  {model_name}: R² = {metrics['r2_score']:.4f}, RMSE = {metrics['rmse']:.2f}")
            
    except Exception as e:
        print(f"Baseline evaluation failed: {e}")
    
    print("\n✅ Example completed successfully!")
    print("\nNext steps:")
    print("1. Run full training: python -m src.train")
    print("2. Evaluate model: python evaluate_model.py")
    print("3. Open notebooks: jupyter notebook")

if __name__ == "__main__":
    main() 