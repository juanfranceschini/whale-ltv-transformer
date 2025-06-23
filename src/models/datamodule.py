"""
PyTorch Lightning DataModule for Whale LTV Transformer.
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional
import pytorch_lightning as pl
from src.utils import (
    normalize_ltv, extract_features, create_sequence_tensor, 
    split_data, get_feature_names
)
import logging

logger = logging.getLogger(__name__)


class CustomerSequenceDataset(Dataset):
    """Dataset for customer event sequences."""
    
    def __init__(
        self, 
        data: pd.DataFrame, 
        max_sequence_length: int = 10,
        normalize_ltv_values: bool = True
    ):
        self.data = data
        self.max_sequence_length = max_sequence_length
        
        # Extract features
        self.features = extract_features(data)
        
        # Normalize LTV if requested
        if normalize_ltv_values:
            self.ltv_values, self.ltv_scaler = normalize_ltv(data['ltv_90d'].values)
        else:
            self.ltv_values = data['ltv_90d'].values
            self.ltv_scaler = None
        
        # Extract whale labels
        self.whale_labels = data['is_whale'].values
        
        # Create sequence tensors
        self.sequences = []
        self.sequence_lengths = []
        
        for _, row in data.iterrows():
            events = row['events']
            sequence = create_sequence_tensor(events, max_sequence_length)
            self.sequences.append(sequence)
            self.sequence_lengths.append(min(len(events), max_sequence_length))
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            'features': torch.tensor(self.features[idx], dtype=torch.float32),
            'sequence': self.sequences[idx],
            'sequence_length': torch.tensor(self.sequence_lengths[idx], dtype=torch.long),
            'ltv': torch.tensor(self.ltv_values[idx], dtype=torch.float32),
            'whale': torch.tensor(self.whale_labels[idx], dtype=torch.float32)
        }


class WhaleLTVDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for Whale LTV prediction."""
    
    def __init__(
        self,
        data_path: str = "data/processed/customer_sequences.parquet",
        batch_size: int = 32,
        num_workers: int = 4,
        max_sequence_length: int = 10,
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_state: int = 42,
        normalize_ltv: bool = True
    ):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_sequence_length = max_sequence_length
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
        self.normalize_ltv = normalize_ltv
        
        # Will be set during setup
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.ltv_scaler = None
    
    def setup(self, stage: Optional[str] = None):
        """Set up datasets for training, validation, and testing."""
        logger.info("Setting up datasets...")
        
        # Load processed data
        data = pd.read_parquet(self.data_path)
        logger.info(f"Loaded {len(data)} customer sequences")
        
        # Split data
        train_data, val_data, test_data = split_data(
            data, self.test_size, self.val_size, self.random_state
        )
        
        # Create datasets
        self.train_dataset = CustomerSequenceDataset(
            train_data, self.max_sequence_length, self.normalize_ltv
        )
        self.val_dataset = CustomerSequenceDataset(
            val_data, self.max_sequence_length, self.normalize_ltv
        )
        self.test_dataset = CustomerSequenceDataset(
            test_data, self.max_sequence_length, self.normalize_ltv
        )
        
        # Store scaler from training data
        if self.normalize_ltv:
            self.ltv_scaler = self.train_dataset.ltv_scaler
        
        logger.info("Dataset setup complete")
    
    def train_dataloader(self) -> DataLoader:
        """Return training dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def val_dataloader(self) -> DataLoader:
        """Return validation dataloader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def test_dataloader(self) -> DataLoader:
        """Return test dataloader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def get_feature_dim(self) -> int:
        """Get dimension of numerical features."""
        return len(get_feature_names())
    
    def get_sequence_dim(self) -> int:
        """Get dimension of sequence features."""
        return 3  # [order_value, days_since_signup, order_rank]
    
    def get_max_sequence_length(self) -> int:
        """Get maximum sequence length."""
        return self.max_sequence_length 