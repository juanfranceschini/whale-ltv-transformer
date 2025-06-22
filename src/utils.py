"""
Utility functions for Whale LTV Transformer.
"""

import pandas as pd
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Any
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import logging

logger = logging.getLogger(__name__)


def normalize_ltv(ltv_values: np.ndarray) -> Tuple[np.ndarray, StandardScaler]:
    """Normalize LTV values using StandardScaler."""
    scaler = StandardScaler()
    normalized = scaler.fit_transform(ltv_values.reshape(-1, 1)).flatten()
    return normalized, scaler


def encode_categories(categories_list: List[List[str]]) -> Tuple[np.ndarray, LabelEncoder]:
    """Encode product categories using LabelEncoder."""
    # Flatten all categories
    all_categories = []
    for cats in categories_list:
        if isinstance(cats, list):
            all_categories.extend(cats)
        else:
            all_categories.append(cats)
    
    # Create encoder
    encoder = LabelEncoder()
    encoder.fit(all_categories)
    
    # Encode each customer's categories
    encoded_categories = []
    for cats in categories_list:
        if isinstance(cats, list) and len(cats) > 0:
            encoded = encoder.transform(cats)
            encoded_categories.append(encoded)
        else:
            encoded_categories.append([])
    
    return encoded_categories, encoder


def create_sequence_tensor(events: List[Dict], max_length: int = 10) -> torch.Tensor:
    """Convert event sequence to tensor representation."""
    # Feature dimensions: [order_value, days_since_signup, order_rank]
    features = []
    
    for event in events[:max_length]:
        features.append([
            float(event.get('order_value', 0) or 0),  # Handle None values
            float(event.get('days_since_signup', 0) or 0),  # Handle None values
            float(event.get('order_rank', 0) or 0)  # Handle None values
        ])
    
    # Pad sequences to max_length
    while len(features) < max_length:
        features.append([0.0, 0.0, 0.0])  # Padding
    
    return torch.tensor(features, dtype=torch.float32)


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, task: str = "regression") -> Dict[str, float]:
    """Calculate evaluation metrics for regression or classification."""
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
    
    if task == "regression":
        return {
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred),
            'spearman': pd.Series(y_true).corr(pd.Series(y_pred), method='spearman')
        }
    elif task == "classification":
        # Convert probabilities to binary predictions
        y_pred_binary = (y_pred > 0.5).astype(int)
        
        return {
            'auc': roc_auc_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred_binary),
            'precision': precision_score(y_true, y_pred_binary),
            'recall': recall_score(y_true, y_pred_binary)
        }
    else:
        raise ValueError(f"Unknown task: {task}")


def split_data(
    data: pd.DataFrame, 
    test_size: float = 0.2, 
    val_size: float = 0.1,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split data into train/validation/test sets."""
    # First split: train+val vs test
    train_val, test = train_test_split(
        data, test_size=test_size, random_state=random_state, stratify=data['is_whale']
    )
    
    # Second split: train vs val
    train, val = train_test_split(
        train_val, test_size=val_size/(1-test_size), random_state=random_state, stratify=train_val['is_whale']
    )
    
    logger.info(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
    return train, val, test


def get_feature_names() -> List[str]:
    """Get list of feature names for the model."""
    return [
        'num_orders',
        'total_spend', 
        'avg_order_value',
        'std_order_value',
        'first_order_day',
        'last_order_day',
        'avg_days_between'
    ]


def extract_features(data: pd.DataFrame) -> np.ndarray:
    """Extract numerical features from processed data."""
    feature_names = get_feature_names()
    features = data[feature_names].values
    
    # Handle missing values
    features = np.nan_to_num(features, nan=0.0)
    
    return features


def create_attention_mask(sequence_lengths: List[int], max_length: int) -> torch.Tensor:
    """Create attention mask for variable length sequences."""
    batch_size = len(sequence_lengths)
    mask = torch.zeros(batch_size, max_length, dtype=torch.bool)
    
    for i, length in enumerate(sequence_lengths):
        mask[i, :length] = True
    
    return mask


def log_model_summary(model: torch.nn.Module) -> None:
    """Log model parameter summary."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # Log layer-wise parameters
    for name, param in model.named_parameters():
        if param.requires_grad:
            logger.info(f"{name}: {param.numel():,} parameters")


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    filepath: str
) -> None:
    """Save model checkpoint."""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, filepath)
    logger.info(f"Checkpoint saved to {filepath}")


def load_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    filepath: str
) -> Tuple[int, float]:
    """Load model checkpoint."""
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    
    logger.info(f"Checkpoint loaded from {filepath}")
    return epoch, loss 