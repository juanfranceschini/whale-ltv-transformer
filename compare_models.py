#!/usr/bin/env python3
"""
Model comparison script for Whale LTV Transformer vs Baselines.
"""

import sys
import yaml
import pandas as pd
import numpy as np
from pathlib import Path

def load_results():
    """Load results from all model files."""
    results = {}
    
    # Load transformer results
    transformer_path = Path("models/transformer_results.yaml")
    if transformer_path.exists():
        with open(transformer_path, 'r') as f:
            results['transformer'] = yaml.safe_load(f)
    
    # Load baseline results
    baseline_path = Path("models/baseline_results.yaml")
    if baseline_path.exists():
        with open(baseline_path, 'r') as f:
            results['baselines'] = yaml.safe_load(f)
    
    return results

def print_comparison_table(results):
    """Print a formatted comparison table."""
    print("🐋 Whale LTV Transformer vs Baselines Comparison")
    print("=" * 80)
    
    if 'transformer' not in results:
        print("❌ Transformer results not found!")
        return
    
    if 'baselines' not in results:
        print("❌ Baseline results not found!")
        return
    
    transformer = results['transformer']
    baselines = results['baselines']
    
    # LTV Regression Comparison
    print("\n📊 LTV Prediction (Regression) Metrics:")
    print("-" * 50)
    print(f"{'Model':<15} {'RMSE':<10} {'MAE':<10} {'R²':<10}")
    print("-" * 50)
    
    # Transformer
    t_ltv = transformer['ltv_regression']
    print(f"{'Transformer':<15} {t_ltv['rmse']:<10.3f} {t_ltv['mae']:<10.3f} {t_ltv['r2']:<10.3f}")
    
    # Baselines
    b_ltv = baselines['ltv_regression']
    for model_name, metrics in b_ltv.items():
        if isinstance(metrics, dict) and 'rmse' in metrics:
            print(f"{model_name.capitalize():<15} {metrics['rmse']:<10.3f} {metrics['mae']:<10.3f} {metrics.get('r2', 0):<10.3f}")
    
    # Whale Classification Comparison
    print("\n🐋 Whale Classification Metrics:")
    print("-" * 50)
    print(f"{'Model':<15} {'AUC':<10} {'F1':<10} {'Precision':<10} {'Recall':<10}")
    print("-" * 50)
    
    # Transformer
    t_whale = transformer['whale_classification']
    print(f"{'Transformer':<15} {t_whale['auc']:<10.3f} {t_whale['f1']:<10.3f} {t_whale['precision']:<10.3f} {t_whale['recall']:<10.3f}")
    
    # Baselines
    b_whale = baselines['whale_classification']
    for model_name, metrics in b_whale.items():
        if isinstance(metrics, dict) and 'auc' in metrics:
            print(f"{model_name.capitalize():<15} {metrics['auc']:<10.3f} {metrics['f1']:<10.3f} {metrics.get('precision', 0):<10.3f} {metrics.get('recall', 0):<10.3f}")
    
    # Business Impact
    print("\n💼 Business Impact Metrics:")
    print("-" * 50)
    if 'business_metrics' in transformer:
        bm = transformer['business_metrics']
        print(f"Whale Detection Rate: {bm['whale_detection_rate']:.1%}")
        print(f"Whale Precision: {bm['whale_precision']:.1%}")
        print(f"Revenue Prediction Error: {bm['revenue_prediction_error']:.1%}")

def find_best_model(results):
    """Find the best performing model for each metric."""
    print("\n🏆 Best Performing Models:")
    print("-" * 30)
    
    if 'transformer' not in results or 'baselines' not in results:
        return
    
    transformer = results['transformer']
    baselines = results['baselines']
    
    # LTV Regression - Best RMSE
    models_rmse = {'Transformer': transformer['ltv_regression']['rmse']}
    for model_name, metrics in baselines['ltv_regression'].items():
        if isinstance(metrics, dict) and 'rmse' in metrics:
            models_rmse[model_name.capitalize()] = metrics['rmse']
    
    best_rmse = min(models_rmse.items(), key=lambda x: x[1])
    print(f"Best RMSE: {best_rmse[0]} ({best_rmse[1]:.3f})")
    
    # Whale Classification - Best AUC
    models_auc = {'Transformer': transformer['whale_classification']['auc']}
    for model_name, metrics in baselines['whale_classification'].items():
        if isinstance(metrics, dict) and 'auc' in metrics:
            models_auc[model_name.capitalize()] = metrics['auc']
    
    best_auc = max(models_auc.items(), key=lambda x: x[1])
    print(f"Best AUC: {best_auc[0]} ({best_auc[1]:.3f})")

def main():
    """Main comparison function."""
    print("🔍 Loading model results...")
    results = load_results()
    
    if not results:
        print("❌ No results found! Please run training and evaluation first.")
        return
    
    print_comparison_table(results)
    find_best_model(results)
    
    print("\n✅ Model comparison complete!")

if __name__ == "__main__":
    main() 