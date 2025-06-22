"""
Unit tests for transformer model.
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import tempfile
import shutil

from src.models.transformer import WhaleLTVTransformer, TransformerEncoder, PositionalEncoding
from src.models.datamodule import WhaleLTVDataModule, CustomerSequenceDataset


class TestPositionalEncoding:
    """Test cases for PositionalEncoding."""
    
    def test_positional_encoding_shape(self):
        """Test positional encoding output shape."""
        d_model = 64
        max_len = 100
        pe = PositionalEncoding(d_model, max_len)
        
        x = torch.randn(50, 32, d_model)  # (seq_len, batch_size, d_model)
        output = pe(x)
        
        assert output.shape == x.shape
        assert not torch.allclose(output, x)  # Should be different due to positional encoding
    
    def test_positional_encoding_values(self):
        """Test that positional encoding values are reasonable."""
        d_model = 32
        max_len = 50
        pe = PositionalEncoding(d_model, max_len)
        
        x = torch.randn(10, 1, d_model)
        output = pe(x)
        
        # Check that output is finite
        assert torch.isfinite(output).all()
        
        # Check that output is not all zeros
        assert not torch.allclose(output, torch.zeros_like(output))


class TestTransformerEncoder:
    """Test cases for TransformerEncoder."""
    
    def test_transformer_encoder_shape(self):
        """Test transformer encoder output shape."""
        input_dim = 10
        d_model = 64
        max_len = 50
        
        encoder = TransformerEncoder(
            input_dim=input_dim,
            d_model=d_model,
            max_len=max_len
        )
        
        batch_size = 8
        seq_len = 20
        x = torch.randn(batch_size, seq_len, input_dim)
        
        output = encoder(x)
        
        assert output.shape == (batch_size, seq_len, d_model)
    
    def test_transformer_encoder_with_mask(self):
        """Test transformer encoder with attention mask."""
        input_dim = 5
        d_model = 32
        max_len = 30
        
        encoder = TransformerEncoder(
            input_dim=input_dim,
            d_model=d_model,
            max_len=max_len
        )
        
        batch_size = 4
        seq_len = 15
        x = torch.randn(batch_size, seq_len, input_dim)
        mask = torch.randint(0, 2, (batch_size, seq_len), dtype=torch.bool)
        
        output = encoder(x, mask)
        
        assert output.shape == (batch_size, seq_len, d_model)
        assert torch.isfinite(output).all()


class TestWhaleLTVTransformer:
    """Test cases for WhaleLTVTransformer."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.model = WhaleLTVTransformer(
            feature_dim=7,
            sequence_dim=3,
            max_sequence_length=10,
            d_model=64,
            nhead=4,
            num_layers=2,
            dim_feedforward=128,
            dropout=0.1,
            ltv_weight=0.6,
            learning_rate=0.001,
            weight_decay=0.0001
        )
    
    def test_model_initialization(self):
        """Test model initialization."""
        assert self.model is not None
        assert hasattr(self.model, 'feature_projection')
        assert hasattr(self.model, 'sequence_encoder')
        assert hasattr(self.model, 'shared_layers')
        assert hasattr(self.model, 'ltv_head')
        assert hasattr(self.model, 'whale_head')
    
    def test_forward_pass(self):
        """Test model forward pass."""
        batch_size = 4
        feature_dim = 7
        sequence_dim = 3
        max_sequence_length = 10
        
        # Create mock batch
        batch = {
            'features': torch.randn(batch_size, feature_dim),
            'sequence': torch.randn(batch_size, max_sequence_length, sequence_dim),
            'sequence_length': torch.randint(1, max_sequence_length + 1, (batch_size,)),
            'ltv': torch.randn(batch_size),
            'whale': torch.randint(0, 2, (batch_size,)).float()
        }
        
        # Forward pass
        outputs = self.model(batch)
        
        # Check outputs
        assert 'ltv_pred' in outputs
        assert 'whale_pred' in outputs
        assert 'shared_features' in outputs
        
        assert outputs['ltv_pred'].shape == (batch_size,)
        assert outputs['whale_pred'].shape == (batch_size,)
        assert outputs['shared_features'].shape == (batch_size, 32)  # d_model // 2
        
        # Check value ranges
        assert torch.isfinite(outputs['ltv_pred']).all()
        assert torch.isfinite(outputs['whale_pred']).all()
        assert (outputs['whale_pred'] >= 0).all() and (outputs['whale_pred'] <= 1).all()
    
    def test_training_step(self):
        """Test training step."""
        batch_size = 2
        feature_dim = 7
        sequence_dim = 3
        max_sequence_length = 10
        
        # Create mock batch
        batch = {
            'features': torch.randn(batch_size, feature_dim),
            'sequence': torch.randn(batch_size, max_sequence_length, sequence_dim),
            'sequence_length': torch.randint(1, max_sequence_length + 1, (batch_size,)),
            'ltv': torch.randn(batch_size),
            'whale': torch.randint(0, 2, (batch_size,)).float()
        }
        
        # Training step
        loss = self.model.training_step(batch, 0)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.requires_grad
        assert torch.isfinite(loss)
        assert loss > 0
    
    def test_validation_step(self):
        """Test validation step."""
        batch_size = 2
        feature_dim = 7
        sequence_dim = 3
        max_sequence_length = 10
        
        # Create mock batch
        batch = {
            'features': torch.randn(batch_size, feature_dim),
            'sequence': torch.randn(batch_size, max_sequence_length, sequence_dim),
            'sequence_length': torch.randint(1, max_sequence_length + 1, (batch_size,)),
            'ltv': torch.randn(batch_size),
            'whale': torch.randint(0, 2, (batch_size,)).float()
        }
        
        # Validation step
        outputs = self.model.validation_step(batch, 0)
        
        assert isinstance(outputs, dict)
        assert 'val_ltv_loss' in outputs
        assert 'val_whale_loss' in outputs
        assert 'val_total_loss' in outputs
        assert 'ltv_pred' in outputs
        assert 'whale_pred' in outputs
        assert 'ltv_true' in outputs
        assert 'whale_true' in outputs
    
    def test_configure_optimizers(self):
        """Test optimizer configuration."""
        optimizers = self.model.configure_optimizers()
        
        assert 'optimizer' in optimizers
        assert 'lr_scheduler' in optimizers
        
        optimizer = optimizers['optimizer']
        assert isinstance(optimizer, torch.optim.AdamW)
        
        scheduler = optimizers['lr_scheduler']['scheduler']
        assert scheduler is not None


class TestCustomerSequenceDataset:
    """Test cases for CustomerSequenceDataset."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create temporary directory
        self.temp_dir = tempfile.mkdtemp()
        self.processed_dir = Path(self.temp_dir) / "processed"
        self.processed_dir.mkdir()
        
        # Create mock processed data
        self.create_mock_processed_data()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def create_mock_processed_data(self):
        """Create mock processed data for testing."""
        import pandas as pd
        
        # Create mock customer sequences
        data = []
        for i in range(10):
            events = []
            for j in range(min(3, i + 1)):  # Variable number of events
                events.append({
                    'order_value': 100.0 + i * 10 + j * 5,
                    'days_since_signup': j * 2,
                    'category': f'category_{j}',
                    'order_rank': j + 1
                })
            
            data.append({
                'customer_id': f'customer_{i}',
                'signup_date': pd.Timestamp('2020-01-01'),
                'num_orders': len(events),
                'total_spend': sum(e['order_value'] for e in events),
                'avg_order_value': np.mean([e['order_value'] for e in events]),
                'std_order_value': np.std([e['order_value'] for e in events]),
                'categories': [e['category'] for e in events],
                'first_order_day': 0,
                'last_order_day': (len(events) - 1) * 2,
                'avg_days_between': 2.0,
                'events': events,
                'ltv_90d': 500.0 + i * 50,
                'is_whale': 1 if i >= 8 else 0  # Top 20% are whales
            })
        
        df = pd.DataFrame(data)
        df.to_parquet(self.processed_dir / "customer_sequences.parquet", index=False)
    
    def test_dataset_initialization(self):
        """Test dataset initialization."""
        import pandas as pd
        
        data = pd.read_parquet(self.processed_dir / "customer_sequences.parquet")
        dataset = CustomerSequenceDataset(data, max_sequence_length=10, normalize_ltv_values=True)
        
        assert len(dataset) == 10
        assert dataset.max_sequence_length == 10
        assert dataset.ltv_scaler is not None
    
    def test_dataset_getitem(self):
        """Test dataset item retrieval."""
        import pandas as pd
        
        data = pd.read_parquet(self.processed_dir / "customer_sequences.parquet")
        dataset = CustomerSequenceDataset(data, max_sequence_length=10, normalize_ltv_values=True)
        
        item = dataset[0]
        
        assert 'features' in item
        assert 'sequence' in item
        assert 'sequence_length' in item
        assert 'ltv' in item
        assert 'whale' in item
        
        assert item['features'].shape == (7,)  # feature_dim
        assert item['sequence'].shape == (10, 3)  # max_sequence_length, sequence_dim
        assert isinstance(item['sequence_length'], torch.Tensor)
        assert isinstance(item['ltv'], torch.Tensor)
        assert isinstance(item['whale'], torch.Tensor)


class TestWhaleLTVDataModule:
    """Test cases for WhaleLTVDataModule."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create temporary directory
        self.temp_dir = tempfile.mkdtemp()
        self.processed_dir = Path(self.temp_dir) / "processed"
        self.processed_dir.mkdir()
        
        # Create mock processed data
        self.create_mock_processed_data()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def create_mock_processed_data(self):
        """Create mock processed data for testing."""
        import pandas as pd
        
        # Create mock customer sequences
        data = []
        for i in range(100):  # More data for train/val/test split
            events = []
            for j in range(min(3, i % 4 + 1)):  # Variable number of events
                events.append({
                    'order_value': 100.0 + i * 10 + j * 5,
                    'days_since_signup': j * 2,
                    'category': f'category_{j}',
                    'order_rank': j + 1
                })
            
            data.append({
                'customer_id': f'customer_{i}',
                'signup_date': pd.Timestamp('2020-01-01'),
                'num_orders': len(events),
                'total_spend': sum(e['order_value'] for e in events),
                'avg_order_value': np.mean([e['order_value'] for e in events]),
                'std_order_value': np.std([e['order_value'] for e in events]),
                'categories': [e['category'] for e in events],
                'first_order_day': 0,
                'last_order_day': (len(events) - 1) * 2,
                'avg_days_between': 2.0,
                'events': events,
                'ltv_90d': 500.0 + i * 50,
                'is_whale': 1 if i >= 95 else 0  # Top 5% are whales
            })
        
        df = pd.DataFrame(data)
        df.to_parquet(self.processed_dir / "customer_sequences.parquet", index=False)
    
    def test_datamodule_initialization(self):
        """Test datamodule initialization."""
        datamodule = WhaleLTVDataModule(
            data_path=str(self.processed_dir / "customer_sequences.parquet"),
            batch_size=8,
            num_workers=0,  # Use 0 for testing
            max_sequence_length=10,
            test_size=0.2,
            val_size=0.1,
            random_state=42,
            normalize_ltv=True
        )
        
        assert datamodule.data_path == str(self.processed_dir / "customer_sequences.parquet")
        assert datamodule.batch_size == 8
        assert datamodule.max_sequence_length == 10
    
    def test_datamodule_setup(self):
        """Test datamodule setup."""
        datamodule = WhaleLTVDataModule(
            data_path=str(self.processed_dir / "customer_sequences.parquet"),
            batch_size=8,
            num_workers=0,
            max_sequence_length=10,
            test_size=0.2,
            val_size=0.1,
            random_state=42,
            normalize_ltv=True
        )
        
        datamodule.setup()
        
        assert datamodule.train_dataset is not None
        assert datamodule.val_dataset is not None
        assert datamodule.test_dataset is not None
        assert datamodule.ltv_scaler is not None
        
        # Check dataset sizes
        assert len(datamodule.train_dataset) > 0
        assert len(datamodule.val_dataset) > 0
        assert len(datamodule.test_dataset) > 0
    
    def test_datamodule_dataloaders(self):
        """Test datamodule dataloaders."""
        datamodule = WhaleLTVDataModule(
            data_path=str(self.processed_dir / "customer_sequences.parquet"),
            batch_size=8,
            num_workers=0,
            max_sequence_length=10,
            test_size=0.2,
            val_size=0.1,
            random_state=42,
            normalize_ltv=True
        )
        
        datamodule.setup()
        
        # Test train dataloader
        train_loader = datamodule.train_dataloader()
        batch = next(iter(train_loader))
        
        assert 'features' in batch
        assert 'sequence' in batch
        assert 'sequence_length' in batch
        assert 'ltv' in batch
        assert 'whale' in batch
        
        assert batch['features'].shape[0] <= 8  # batch_size
        assert batch['sequence'].shape[1] == 10  # max_sequence_length
        assert batch['sequence'].shape[2] == 3  # sequence_dim
        
        # Test val dataloader
        val_loader = datamodule.val_dataloader()
        batch = next(iter(val_loader))
        
        assert 'features' in batch
        assert 'sequence' in batch
        assert 'sequence_length' in batch
        assert 'ltv' in batch
        assert 'whale' in batch
    
    def test_datamodule_dimensions(self):
        """Test datamodule dimension methods."""
        datamodule = WhaleLTVDataModule(
            data_path=str(self.processed_dir / "customer_sequences.parquet"),
            batch_size=8,
            num_workers=0,
            max_sequence_length=10,
            test_size=0.2,
            val_size=0.1,
            random_state=42,
            normalize_ltv=True
        )
        
        assert datamodule.get_feature_dim() == 7
        assert datamodule.get_sequence_dim() == 3
        assert datamodule.get_max_sequence_length() == 10


if __name__ == "__main__":
    pytest.main([__file__]) 