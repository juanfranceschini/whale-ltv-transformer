"""
Unit tests for utility functions.
"""

import pytest
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from src.utils import (
    normalize_ltv, encode_categories, create_sequence_tensor,
    calculate_metrics, split_data, get_feature_names, extract_features,
    create_attention_mask, log_model_summary, save_checkpoint, load_checkpoint
)


class TestNormalizeLTV:
    """Test cases for LTV normalization."""
    
    def test_normalize_ltv_basic(self):
        """Test basic LTV normalization."""
        ltv_values = np.array([100, 200, 300, 400, 500])
        normalized, scaler = normalize_ltv(ltv_values)
        
        assert isinstance(normalized, np.ndarray)
        assert isinstance(scaler, StandardScaler)
        assert normalized.shape == ltv_values.shape
        assert np.isclose(normalized.mean(), 0, atol=1e-10)
        assert np.isclose(normalized.std(), 1, atol=1e-10)
    
    def test_normalize_ltv_roundtrip(self):
        """Test that normalization can be reversed."""
        ltv_values = np.array([100, 200, 300, 400, 500])
        normalized, scaler = normalize_ltv(ltv_values)
        
        # Reverse normalization
        reversed_values = scaler.inverse_transform(normalized.reshape(-1, 1)).flatten()
        
        assert np.allclose(ltv_values, reversed_values, atol=1e-10)
    
    def test_normalize_ltv_single_value(self):
        """Test normalization with single value."""
        ltv_values = np.array([100])
        normalized, scaler = normalize_ltv(ltv_values)
        
        assert normalized.shape == (1,)
        assert np.isclose(normalized[0], 0, atol=1e-10)  # Single value becomes 0


class TestEncodeCategories:
    """Test cases for category encoding."""
    
    def test_encode_categories_basic(self):
        """Test basic category encoding."""
        categories_list = [
            ['electronics', 'books'],
            ['books', 'home'],
            ['electronics'],
            ['sports', 'fashion', 'electronics']
        ]
        
        encoded, encoder = encode_categories(categories_list)
        
        assert isinstance(encoded, list)
        assert isinstance(encoder, type(train_test_split))  # LabelEncoder
        assert len(encoded) == len(categories_list)
        
        # Check that encoded values are integers
        for enc in encoded:
            if len(enc) > 0:
                assert all(isinstance(x, (int, np.integer)) for x in enc)
    
    def test_encode_categories_empty(self):
        """Test encoding with empty categories."""
        categories_list = [
            [],
            ['electronics'],
            [],
            ['books', 'home']
        ]
        
        encoded, encoder = encode_categories(categories_list)
        
        assert len(encoded) == len(categories_list)
        assert encoded[0] == []
        assert encoded[2] == []
        assert len(encoded[1]) == 1
        assert len(encoded[3]) == 2


class TestCreateSequenceTensor:
    """Test cases for sequence tensor creation."""
    
    def test_create_sequence_tensor_basic(self):
        """Test basic sequence tensor creation."""
        events = [
            {'order_value': 100, 'days_since_signup': 0, 'order_rank': 1},
            {'order_value': 200, 'days_since_signup': 2, 'order_rank': 2},
            {'order_value': 150, 'days_since_signup': 5, 'order_rank': 3}
        ]
        
        tensor = create_sequence_tensor(events, max_length=5)
        
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (5, 3)  # max_length, sequence_dim
        assert tensor.dtype == torch.float32
        
        # Check first 3 rows contain event data
        assert tensor[0, 0] == 100  # order_value
        assert tensor[0, 1] == 0    # days_since_signup
        assert tensor[0, 2] == 1    # order_rank
        
        # Check padding rows are zeros
        assert torch.all(tensor[3:] == 0)
    
    def test_create_sequence_tensor_short(self):
        """Test sequence tensor with short sequence."""
        events = [
            {'order_value': 100, 'days_since_signup': 0, 'order_rank': 1}
        ]
        
        tensor = create_sequence_tensor(events, max_length=3)
        
        assert tensor.shape == (3, 3)
        assert tensor[0, 0] == 100
        assert torch.all(tensor[1:] == 0)  # Padding
    
    def test_create_sequence_tensor_long(self):
        """Test sequence tensor with long sequence (truncation)."""
        events = [
            {'order_value': 100, 'days_since_signup': 0, 'order_rank': 1},
            {'order_value': 200, 'days_since_signup': 2, 'order_rank': 2},
            {'order_value': 150, 'days_since_signup': 5, 'order_rank': 3},
            {'order_value': 300, 'days_since_signup': 7, 'order_rank': 4},
            {'order_value': 250, 'days_since_signup': 10, 'order_rank': 5}
        ]
        
        tensor = create_sequence_tensor(events, max_length=3)
        
        assert tensor.shape == (3, 3)
        assert tensor[0, 0] == 100  # First event
        assert tensor[1, 0] == 200  # Second event
        assert tensor[2, 0] == 150  # Third event (truncated)


class TestCalculateMetrics:
    """Test cases for metric calculation."""
    
    def test_calculate_metrics_regression(self):
        """Test regression metrics calculation."""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1.1, 1.9, 3.1, 3.9, 5.1])
        
        metrics = calculate_metrics(y_true, y_pred, task="regression")
        
        assert isinstance(metrics, dict)
        assert 'rmse' in metrics
        assert 'mae' in metrics
        assert 'r2' in metrics
        assert 'spearman' in metrics
        
        assert metrics['rmse'] > 0
        assert metrics['mae'] > 0
        assert metrics['r2'] > 0
        assert abs(metrics['spearman']) <= 1
    
    def test_calculate_metrics_classification(self):
        """Test classification metrics calculation."""
        y_true = np.array([0, 1, 0, 1, 0, 1])
        y_pred = np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.7])
        
        metrics = calculate_metrics(y_true, y_pred, task="classification")
        
        assert isinstance(metrics, dict)
        assert 'auc' in metrics
        assert 'f1' in metrics
        assert 'precision' in metrics
        
        assert 0 <= metrics['auc'] <= 1
        assert 0 <= metrics['f1'] <= 1
        assert 0 <= metrics['precision'] <= 1
    
    def test_calculate_metrics_invalid_task(self):
        """Test metrics calculation with invalid task."""
        y_true = np.array([1, 2, 3])
        y_pred = np.array([1, 2, 3])
        
        with pytest.raises(ValueError):
            calculate_metrics(y_true, y_pred, task="invalid")


class TestSplitData:
    """Test cases for data splitting."""
    
    def test_split_data_basic(self):
        """Test basic data splitting."""
        # Create mock data
        data = pd.DataFrame({
            'customer_id': range(100),
            'ltv_90d': np.random.randn(100),
            'is_whale': np.random.randint(0, 2, 100)
        })
        
        train, val, test = split_data(data, test_size=0.2, val_size=0.1, random_state=42)
        
        assert len(train) + len(val) + len(test) == len(data)
        assert len(test) == int(0.2 * len(data))
        assert len(val) == int(0.1 * len(data) * 0.8)  # 10% of train+val
        
        # Check that all customer_ids are unique across splits
        all_ids = set(train['customer_id']) | set(val['customer_id']) | set(test['customer_id'])
        assert len(all_ids) == len(data)
    
    def test_split_data_stratification(self):
        """Test that stratification works correctly."""
        # Create data with clear whale/non-whale split
        data = pd.DataFrame({
            'customer_id': range(100),
            'ltv_90d': np.random.randn(100),
            'is_whale': [1] * 20 + [0] * 80  # 20% whales
        })
        
        train, val, test = split_data(data, test_size=0.2, val_size=0.1, random_state=42)
        
        # Check that whale proportion is maintained approximately
        train_whale_pct = train['is_whale'].mean()
        val_whale_pct = val['is_whale'].mean()
        test_whale_pct = test['is_whale'].mean()
        
        assert abs(train_whale_pct - 0.2) < 0.1
        assert abs(val_whale_pct - 0.2) < 0.1
        assert abs(test_whale_pct - 0.2) < 0.1


class TestFeatureFunctions:
    """Test cases for feature-related functions."""
    
    def test_get_feature_names(self):
        """Test feature names retrieval."""
        feature_names = get_feature_names()
        
        assert isinstance(feature_names, list)
        assert len(feature_names) == 7
        assert 'num_orders' in feature_names
        assert 'total_spend' in feature_names
        assert 'avg_order_value' in feature_names
        assert 'std_order_value' in feature_names
        assert 'first_order_day' in feature_names
        assert 'last_order_day' in feature_names
        assert 'avg_days_between' in feature_names
    
    def test_extract_features(self):
        """Test feature extraction."""
        # Create mock data with all required features
        data = pd.DataFrame({
            'num_orders': [1, 2, 3],
            'total_spend': [100, 200, 300],
            'avg_order_value': [100, 100, 100],
            'std_order_value': [0, 10, 20],
            'first_order_day': [0, 0, 1],
            'last_order_day': [0, 2, 5],
            'avg_days_between': [0, 2, 2],
            'other_col': ['a', 'b', 'c']  # Should be ignored
        })
        
        features = extract_features(data)
        
        assert isinstance(features, np.ndarray)
        assert features.shape == (3, 7)
        assert features.dtype == np.float64
        
        # Check that features match expected values
        assert features[0, 0] == 1  # num_orders
        assert features[0, 1] == 100  # total_spend
    
    def test_extract_features_missing_values(self):
        """Test feature extraction with missing values."""
        data = pd.DataFrame({
            'num_orders': [1, 2, np.nan],
            'total_spend': [100, 200, 300],
            'avg_order_value': [100, np.nan, 100],
            'std_order_value': [0, 10, 20],
            'first_order_day': [0, 0, 1],
            'last_order_day': [0, 2, 5],
            'avg_days_between': [0, 2, 2]
        })
        
        features = extract_features(data)
        
        assert isinstance(features, np.ndarray)
        assert features.shape == (3, 7)
        assert np.isfinite(features).all()  # No NaN values
        assert (features == 0).any()  # Some values should be 0 (filled NaN)


class TestAttentionMask:
    """Test cases for attention mask creation."""
    
    def test_create_attention_mask_basic(self):
        """Test basic attention mask creation."""
        sequence_lengths = [3, 5, 2, 4]
        max_length = 6
        
        mask = create_attention_mask(sequence_lengths, max_length)
        
        assert isinstance(mask, torch.Tensor)
        assert mask.shape == (4, 6)  # batch_size, max_length
        assert mask.dtype == torch.bool
        
        # Check that mask is True for valid positions
        assert mask[0, :3].all()  # First sequence: positions 0,1,2 should be True
        assert not mask[0, 3:].any()  # First sequence: positions 3,4,5 should be False
        
        assert mask[1, :5].all()  # Second sequence: positions 0,1,2,3,4 should be True
        assert not mask[1, 5:].any()  # Second sequence: position 5 should be False
    
    def test_create_attention_mask_edge_cases(self):
        """Test attention mask with edge cases."""
        # Empty sequences
        sequence_lengths = [0, 1, 0]
        max_length = 3
        
        mask = create_attention_mask(sequence_lengths, max_length)
        
        assert mask.shape == (3, 3)
        assert not mask[0].any()  # First sequence: all False
        assert mask[1, 0]  # Second sequence: only position 0 is True
        assert not mask[1, 1:].any()  # Second sequence: rest are False
        assert not mask[2].any()  # Third sequence: all False
        
        # All sequences at max length
        sequence_lengths = [5, 5, 5]
        max_length = 5
        
        mask = create_attention_mask(sequence_lengths, max_length)
        
        assert mask.shape == (3, 5)
        assert mask.all()  # All positions should be True


class TestModelUtilities:
    """Test cases for model utility functions."""
    
    def test_log_model_summary(self, caplog):
        """Test model summary logging."""
        # Create a simple model
        model = torch.nn.Sequential(
            torch.nn.Linear(10, 5),
            torch.nn.ReLU(),
            torch.nn.Linear(5, 1)
        )
        
        log_model_summary(model)
        
        # Check that logs were created
        assert "Total parameters" in caplog.text
        assert "Trainable parameters" in caplog.text
    
    def test_save_load_checkpoint(self, tmp_path):
        """Test checkpoint saving and loading."""
        # Create a simple model and optimizer
        model = torch.nn.Linear(10, 1)
        optimizer = torch.optim.Adam(model.parameters())
        
        # Save checkpoint
        checkpoint_path = tmp_path / "test_checkpoint.pth"
        save_checkpoint(model, optimizer, epoch=5, loss=0.123, filepath=str(checkpoint_path))
        
        assert checkpoint_path.exists()
        
        # Load checkpoint
        new_model = torch.nn.Linear(10, 1)
        new_optimizer = torch.optim.Adam(new_model.parameters())
        
        epoch, loss = load_checkpoint(new_model, new_optimizer, str(checkpoint_path))
        
        assert epoch == 5
        assert loss == 0.123


if __name__ == "__main__":
    pytest.main([__file__]) 