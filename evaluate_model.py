#!/usr/bin/env python3
"""
Quick evaluation script for the trained Whale LTV Transformer.
"""

import sys
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import yaml

# Add src to path
sys.path.append('src')

from src.models.transformer import WhaleLTVTransformer
from src.models.datamodule import WhaleLTVDataModule
from src.utils import calculate_metrics

def main():
    print("🐋 Evaluating Whale LTV Transformer...")
    
    # Load the best checkpoint
    checkpoint_path = Path("models/checkpoints/whale_ltv_transformer_epoch=29_val_total_loss=0.0137.ckpt")
    
    if not checkpoint_path.exists():
        print("❌ Best checkpoint not found!")
        return
    
    # Load model
    model = WhaleLTVTransformer.load_from_checkpoint(checkpoint_path)
    model.eval()
    print(f"✅ Model loaded from: {checkpoint_path}")
    
    # Setup data
    datamodule = WhaleLTVDataModule(
        data_path="data/processed/customer_sequences.parquet",
        batch_size=32,
        max_sequence_length=10
    )
    datamodule.setup()
    
    # Get test predictions
    print("📊 Running inference on test set...")
    ltv_preds = []
    whale_preds = []
    ltv_targets = []
    whale_targets = []
    
    model.eval()
    with torch.no_grad():
        for batch in datamodule.test_dataloader():
            outputs = model(batch)
            
            ltv_preds.extend(outputs['ltv_pred'].cpu().numpy())
            whale_preds.extend(outputs['whale_pred'].cpu().numpy())
            ltv_targets.extend(batch['ltv'].cpu().numpy())
            whale_targets.extend(batch['whale'].cpu().numpy())
    
    ltv_preds = np.array(ltv_preds)
    whale_preds = np.array(whale_preds)
    ltv_targets = np.array(ltv_targets)
    whale_targets = np.array(whale_targets)
    
    # Denormalize LTVs if scaler is present
    if datamodule.ltv_scaler is not None:
        ltv_preds = datamodule.ltv_scaler.inverse_transform(ltv_preds.reshape(-1, 1)).flatten()
        ltv_targets = datamodule.ltv_scaler.inverse_transform(ltv_targets.reshape(-1, 1)).flatten()
    
    # Calculate metrics
    print("\n📈 Performance Metrics:")
    print("=" * 50)
    
    # LTV Regression metrics
    ltv_metrics = calculate_metrics(ltv_targets, ltv_preds, task="regression")
    print("\n🎯 LTV Prediction:")
    for metric, value in ltv_metrics.items():
        print(f"  {metric.upper()}: {value:.4f}")
    
    # Whale Classification metrics
    whale_metrics = calculate_metrics(whale_targets, whale_preds, task="classification")
    print("\n🐋 Whale Classification:")
    for metric, value in whale_metrics.items():
        print(f"  {metric.upper()}: {value:.4f}")
    
    # Business insights
    print("\n💼 Business Impact:")
    print("=" * 50)
    
    # Whale detection
    whale_threshold = 0.5
    predicted_whales = (whale_preds > whale_threshold).sum()
    true_whales = whale_targets.sum()
    correctly_identified = ((whale_preds > whale_threshold) & (whale_targets == 1)).sum()
    
    print(f"  Total whales in test set: {true_whales}")
    print(f"  Predicted whales: {predicted_whales}")
    print(f"  Correctly identified: {correctly_identified}")
    print(f"  Detection rate: {correctly_identified/true_whales:.1%}")
    print(f"  Precision: {correctly_identified/predicted_whales:.1%}")
    
    # Revenue prediction accuracy
    total_actual_revenue = ltv_targets.sum()
    total_predicted_revenue = ltv_preds.sum()
    revenue_error = abs(total_actual_revenue - total_predicted_revenue) / total_actual_revenue
    
    print(f"\n  Total actual 90-day revenue: ${total_actual_revenue:,.2f}")
    print(f"  Total predicted revenue: ${total_predicted_revenue:,.2f}")
    print(f"  Revenue prediction error: {revenue_error:.1%}")
    
    # Save results
    results = {
        'ltv_regression': ltv_metrics,
        'whale_classification': whale_metrics,
        'business_metrics': {
            'whale_detection_rate': correctly_identified/true_whales,
            'whale_precision': correctly_identified/predicted_whales,
            'revenue_prediction_error': revenue_error
        }
    }
    
    with open('models/transformer_results.yaml', 'w') as f:
        yaml.dump(results, f, default_flow_style=False)
    
    print(f"\n✅ Results saved to models/transformer_results.yaml")
    print("\n🎉 Evaluation complete!")

if __name__ == "__main__":
    main() 