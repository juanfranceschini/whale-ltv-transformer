"""
Transformer model for Whale LTV prediction.

Implements a Tencent-style architecture with:
- Shared transformer encoder trunk
- Dual heads for LTV regression and whale classification
- Joint loss optimization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import pytorch_lightning as pl
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import logging

logger = logging.getLogger(__name__)


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer sequences."""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:x.size(0), :]


class TransformerEncoder(nn.Module):
    """Transformer encoder for sequence processing."""
    
    def __init__(
        self,
        input_dim: int,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 2,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        max_len: int = 50
    ):
        super().__init__()
        
        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.d_model = d_model
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: (batch_size, seq_len, input_dim)
        x = self.input_projection(x)  # (batch_size, seq_len, d_model)
        x = x.transpose(0, 1)  # (seq_len, batch_size, d_model)
        x = self.pos_encoding(x)
        x = x.transpose(0, 1)  # (batch_size, seq_len, d_model)
        
        if mask is not None:
            # Convert boolean mask to attention mask format
            mask = ~mask  # Invert for transformer attention mask
        
        output = self.transformer(x, src_key_padding_mask=mask)
        return output


class WhaleLTVTransformer(pl.LightningModule):
    """Transformer model for Whale LTV prediction with dual heads."""
    
    def __init__(
        self,
        feature_dim: int = 7,
        sequence_dim: int = 3,
        max_sequence_length: int = 10,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 2,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        ltv_weight: float = 0.6,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Model dimensions
        self.feature_dim = feature_dim
        self.sequence_dim = sequence_dim
        self.max_sequence_length = max_sequence_length
        self.d_model = d_model
        self.ltv_weight = ltv_weight
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        # Feature processing
        self.feature_projection = nn.Linear(feature_dim, d_model)
        
        # Sequence processing
        self.sequence_encoder = TransformerEncoder(
            input_dim=sequence_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            max_len=max_sequence_length
        )
        
        # Shared trunk
        self.shared_layers = nn.Sequential(
            nn.Linear(d_model * 2, d_model),  # Concatenated features + sequence
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # LTV regression head
        self.ltv_head = nn.Sequential(
            nn.Linear(d_model // 2, d_model // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 4, 1)
        )
        
        # Whale classification head
        self.whale_head = nn.Sequential(
            nn.Linear(d_model // 2, d_model // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 4, 1),
            nn.Sigmoid()
        )
        
        # Loss functions
        self.ltv_loss = nn.SmoothL1Loss()
        self.whale_loss = nn.BCELoss()
        
        # Metrics tracking
        self.train_ltv_losses = []
        self.train_whale_losses = []
        self.val_ltv_losses = []
        self.val_whale_losses = []
    
    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Forward pass through the model."""
        features = batch['features']  # (batch_size, feature_dim)
        sequences = batch['sequence']  # (batch_size, seq_len, sequence_dim)
        sequence_lengths = batch['sequence_length']  # (batch_size,)
        
        # Process features
        feature_embeddings = self.feature_projection(features)  # (batch_size, d_model)
        
        # Process sequences
        sequence_embeddings = self.sequence_encoder(sequences)  # (batch_size, seq_len, d_model)
        
        # Global average pooling over sequence dimension
        # Create mask for variable length sequences
        mask = torch.arange(self.max_sequence_length, device=sequences.device).unsqueeze(0) < sequence_lengths.unsqueeze(1)
        mask = mask.unsqueeze(-1).expand(-1, -1, self.d_model)  # (batch_size, seq_len, d_model)
        
        # Apply mask and average
        masked_embeddings = sequence_embeddings * mask.float()
        sequence_pooled = masked_embeddings.sum(dim=1) / sequence_lengths.unsqueeze(1).float()  # (batch_size, d_model)
        
        # Concatenate features and sequence
        combined = torch.cat([feature_embeddings, sequence_pooled], dim=1)  # (batch_size, d_model * 2)
        
        # Shared trunk
        shared_features = self.shared_layers(combined)  # (batch_size, d_model // 2)
        
        # Dual heads
        ltv_pred = self.ltv_head(shared_features).squeeze(-1)  # (batch_size,)
        whale_pred = self.whale_head(shared_features).squeeze(-1)  # (batch_size,)
        
        return {
            'ltv_pred': ltv_pred,
            'whale_pred': whale_pred,
            'shared_features': shared_features
        }
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step with joint loss."""
        outputs = self(batch)
        
        ltv_loss = self.ltv_loss(outputs['ltv_pred'], batch['ltv'])
        whale_loss = self.whale_loss(outputs['whale_pred'], batch['whale'])
        
        # Joint loss
        total_loss = self.ltv_weight * ltv_loss + (1 - self.ltv_weight) * whale_loss
        
        # Log losses
        self.log('train_ltv_loss', ltv_loss, prog_bar=True)
        self.log('train_whale_loss', whale_loss, prog_bar=True)
        self.log('train_total_loss', total_loss, prog_bar=True)
        
        self.train_ltv_losses.append(ltv_loss.item())
        self.train_whale_losses.append(whale_loss.item())
        
        return total_loss
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """Validation step."""
        outputs = self(batch)
        
        ltv_loss = self.ltv_loss(outputs['ltv_pred'], batch['ltv'])
        whale_loss = self.whale_loss(outputs['whale_pred'], batch['whale'])
        total_loss = self.ltv_weight * ltv_loss + (1 - self.ltv_weight) * whale_loss
        
        # Log losses
        self.log('val_ltv_loss', ltv_loss, prog_bar=True)
        self.log('val_whale_loss', whale_loss, prog_bar=True)
        self.log('val_total_loss', total_loss, prog_bar=True)
        
        self.val_ltv_losses.append(ltv_loss.item())
        self.val_whale_losses.append(whale_loss.item())
        
        return {
            'val_ltv_loss': ltv_loss,
            'val_whale_loss': whale_loss,
            'val_total_loss': total_loss,
            'ltv_pred': outputs['ltv_pred'],
            'whale_pred': outputs['whale_pred'],
            'ltv_true': batch['ltv'],
            'whale_true': batch['whale']
        }
    
    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """Test step."""
        outputs = self(batch)
        
        ltv_loss = self.ltv_loss(outputs['ltv_pred'], batch['ltv'])
        whale_loss = self.whale_loss(outputs['whale_pred'], batch['whale'])
        total_loss = self.ltv_weight * ltv_loss + (1 - self.ltv_weight) * whale_loss
        
        return {
            'test_ltv_loss': ltv_loss,
            'test_whale_loss': whale_loss,
            'test_total_loss': total_loss,
            'ltv_pred': outputs['ltv_pred'],
            'whale_pred': outputs['whale_pred'],
            'ltv_true': batch['ltv'],
            'whale_true': batch['whale']
        }
    
    def configure_optimizers(self):
        """Configure optimizer and scheduler."""
        optimizer = AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs,
            eta_min=1e-6
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_total_loss'
            }
        }
    
    def on_train_epoch_end(self):
        """Log epoch-level metrics."""
        if self.train_ltv_losses:
            avg_ltv_loss = sum(self.train_ltv_losses) / len(self.train_ltv_losses)
            avg_whale_loss = sum(self.train_whale_losses) / len(self.train_whale_losses)
            
            self.log('train_epoch_ltv_loss', avg_ltv_loss)
            self.log('train_epoch_whale_loss', avg_whale_loss)
            
            # Clear lists
            self.train_ltv_losses.clear()
            self.train_whale_losses.clear()
    
    def on_validation_epoch_end(self):
        """Log epoch-level validation metrics."""
        if self.val_ltv_losses:
            avg_ltv_loss = sum(self.val_ltv_losses) / len(self.val_ltv_losses)
            avg_whale_loss = sum(self.val_whale_losses) / len(self.val_whale_losses)
            
            self.log('val_epoch_ltv_loss', avg_ltv_loss)
            self.log('val_epoch_whale_loss', avg_whale_loss)
            
            # Clear lists
            self.val_ltv_losses.clear()
            self.val_whale_losses.clear() 