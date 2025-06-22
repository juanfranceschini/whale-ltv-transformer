"""
Training script for Whale LTV Transformer.

Uses Hydra for configuration management and PyTorch Lightning for training.
"""

import os
import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
import torch
import logging
from pathlib import Path

from src.models.datamodule import WhaleLTVDataModule
from src.models.transformer import WhaleLTVTransformer
from src.models.baselines import evaluate_baselines
from src.utils import extract_features, split_data, calculate_metrics
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@hydra.main(config_path="../configs", config_name="transformer", version_base=None)
def main(cfg: DictConfig):
    """Main training function."""
    logger.info("Starting Whale LTV Transformer training...")
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")
    
    # Set random seeds
    pl.seed_everything(cfg.seed)
    
    # Create output directories
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(exist_ok=True)
    (output_dir / "checkpoints").mkdir(exist_ok=True)
    (output_dir / "logs").mkdir(exist_ok=True)
    
    # Initialize data module
    logger.info("Setting up data module...")
    datamodule = WhaleLTVDataModule(
        data_path=cfg.data.data_path,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        max_sequence_length=cfg.data.max_sequence_length,
        test_size=cfg.data.test_size,
        val_size=cfg.data.val_size,
        random_state=cfg.seed,
        normalize_ltv=cfg.data.normalize_ltv
    )
    
    # Setup data module
    datamodule.setup()
    
    # Initialize model
    logger.info("Initializing model...")
    model = WhaleLTVTransformer(
        feature_dim=datamodule.get_feature_dim(),
        sequence_dim=datamodule.get_sequence_dim(),
        max_sequence_length=datamodule.get_max_sequence_length(),
        d_model=cfg.model.d_model,
        nhead=cfg.model.nhead,
        num_layers=cfg.model.num_layers,
        dim_feedforward=cfg.model.dim_feedforward,
        dropout=cfg.model.dropout,
        ltv_weight=cfg.model.ltv_weight,
        learning_rate=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay
    )
    
    # Setup callbacks
    callbacks = []
    
    # Model checkpointing
    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir / "checkpoints",
        filename="whale_ltv_transformer_{epoch:02d}_{val_total_loss:.4f}",
        monitor="val_total_loss",
        mode="min",
        save_top_k=3,
        save_last=True
    )
    callbacks.append(checkpoint_callback)
    
    # Early stopping
    early_stopping = EarlyStopping(
        monitor="val_total_loss",
        mode="min",
        patience=cfg.training.patience,
        verbose=True
    )
    callbacks.append(early_stopping)
    
    # Setup logger
    loggers = []
    if cfg.logging.use_wandb:
        wandb_logger = WandbLogger(
            project=cfg.logging.wandb_project,
            name=cfg.logging.wandb_name,
            log_model=True
        )
        loggers.append(wandb_logger)
    
    # Learning rate monitoring (only if logger is configured)
    if loggers:
        lr_monitor = LearningRateMonitor(logging_interval='step')
        callbacks.append(lr_monitor)
    
    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=cfg.training.max_epochs,
        accelerator=cfg.training.accelerator,
        devices=cfg.training.devices,
        callbacks=callbacks,
        logger=loggers,
        log_every_n_steps=cfg.logging.log_every_n_steps,
        val_check_interval=cfg.training.val_check_interval,
        gradient_clip_val=cfg.training.gradient_clip_val,
        accumulate_grad_batches=cfg.training.accumulate_grad_batches,
        deterministic=cfg.training.deterministic
    )
    
    # Train model
    logger.info("Starting training...")
    trainer.fit(model, datamodule)
    
    # Test model
    logger.info("Evaluating model...")
    test_results = trainer.test(model, datamodule)
    
    # Save test results
    results_path = output_dir / "test_results.yaml"
    OmegaConf.save(test_results, results_path)
    logger.info(f"Test results saved to {results_path}")
    
    # Run baseline comparisons if requested
    if cfg.evaluation.run_baselines:
        logger.info("Running baseline comparisons...")
        run_baseline_comparison(datamodule, output_dir)
    
    logger.info("Training complete!")


def run_baseline_comparison(datamodule: WhaleLTVDataModule, output_dir: Path):
    """Run baseline model comparisons."""
    from src.models.baselines import evaluate_baselines
    
    # Load processed data
    data = pd.read_parquet(datamodule.data_path)
    
    # Extract features
    features = extract_features(data)
    
    # Split data
    train_data, val_data, test_data = split_data(
        data, 
        test_size=0.2, 
        val_size=0.1, 
        random_state=42
    )
    
    # Extract features for each split
    X_train = extract_features(train_data)
    X_val = extract_features(val_data)
    X_test = extract_features(test_data)
    
    # LTV regression baselines
    logger.info("Evaluating LTV regression baselines...")
    ltv_results = evaluate_baselines(
        X_train, train_data['ltv_90d'].values,
        X_val, val_data['ltv_90d'].values,
        X_test, test_data['ltv_90d'].values,
        task="regression"
    )
    
    # Whale classification baselines
    logger.info("Evaluating whale classification baselines...")
    whale_results = evaluate_baselines(
        X_train, train_data['is_whale'].values,
        X_val, val_data['is_whale'].values,
        X_test, test_data['is_whale'].values,
        task="classification"
    )
    
    # Save baseline results
    baseline_results = {
        'ltv_regression': ltv_results,
        'whale_classification': whale_results
    }
    
    results_path = output_dir / "baseline_results.yaml"
    import yaml
    with open(results_path, 'w') as f:
        yaml.dump(baseline_results, f, default_flow_style=False)
    
    logger.info(f"Baseline results saved to {results_path}")


if __name__ == "__main__":
    main() 