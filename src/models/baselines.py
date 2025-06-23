"""
Baseline models for Whale LTV prediction.

Includes:
- XGBoost and CatBoost for both regression and classification
- BG/NBD + Gamma-Gamma for LTV prediction
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, roc_auc_score, f1_score
import xgboost as xgb
import catboost as cb
from lifetimes import BetaGeoFitter, GammaGammaFitter
import logging

logger = logging.getLogger(__name__)


class XGBoostBaseline:
    """XGBoost baseline for LTV regression and whale classification."""
    
    def __init__(self, task: str = "regression", **kwargs):
        self.task = task
        self.model = None
        self.scaler = StandardScaler()
        
        # Default parameters
        default_params = {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'random_state': 42
        }
        
        if task == "regression":
            default_params.update({
                'objective': 'reg:squarederror',
                'eval_metric': 'rmse'
            })
        elif task == "classification":
            default_params.update({
                'objective': 'binary:logistic',
                'eval_metric': 'auc'
            })
        
        # Update with provided kwargs
        default_params.update(kwargs)
        self.params = default_params
    
    def fit(self, X: np.ndarray, y: np.ndarray, X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None):
        """Fit the XGBoost model."""
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Create DMatrix
        dtrain = xgb.DMatrix(X_scaled, label=y)
        
        # Validation data if provided
        evals = [(dtrain, 'train')]
        if X_val is not None and y_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
            dval = xgb.DMatrix(X_val_scaled, label=y_val)
            evals.append((dval, 'val'))
        
        # Train model
        self.model = xgb.train(
            self.params,
            dtrain,
            evals=evals,
            verbose_eval=False
        )
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        X_scaled = self.scaler.transform(X)
        dtest = xgb.DMatrix(X_scaled)
        
        if self.task == "regression":
            return self.model.predict(dtest)
        else:
            return self.model.predict(dtest)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores."""
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        importance = self.model.get_score(importance_type='gain')
        return importance


class CatBoostBaseline:
    """CatBoost baseline for LTV regression and whale classification."""
    
    def __init__(self, task: str = "regression", **kwargs):
        self.task = task
        self.model = None
        self.scaler = StandardScaler()
        
        # Default parameters
        default_params = {
            'iterations': 100,
            'depth': 6,
            'learning_rate': 0.1,
            'random_seed': 42,
            'verbose': False
        }
        
        if task == "regression":
            default_params.update({
                'loss_function': 'RMSE',
                'eval_metric': 'RMSE'
            })
        elif task == "classification":
            default_params.update({
                'loss_function': 'Logloss',
                'eval_metric': 'AUC'
            })
        
        # Update with provided kwargs
        default_params.update(kwargs)
        self.params = default_params
    
    def fit(self, X: np.ndarray, y: np.ndarray, X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None):
        """Fit the CatBoost model."""
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Create pool
        train_pool = cb.Pool(X_scaled, label=y)
        
        # Validation pool if provided
        eval_pool = None
        if X_val is not None and y_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
            eval_pool = cb.Pool(X_val_scaled, label=y_val)
        
        # Train model
        self.model = cb.CatBoost(self.params)
        self.model.fit(train_pool, eval_set=eval_pool, verbose=False)
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores."""
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        importance = self.model.get_feature_importance()
        return dict(enumerate(importance))


class BGNBDGammaGammaBaseline:
    """BG/NBD + Gamma-Gamma baseline for LTV prediction."""
    
    def __init__(self):
        self.bgnbd_model = BetaGeoFitter()
        self.gg_model = GammaGammaFitter()
        self.fitted = False
    
    def fit(self, data: pd.DataFrame):
        """Fit BG/NBD and Gamma-Gamma models."""
        # Prepare data for lifetimes library
        # Need: frequency, recency, T, monetary_value
        
        # Calculate customer-level metrics
        customer_metrics = self._calculate_customer_metrics(data)
        
        # Fit BG/NBD model
        self.bgnbd_model.fit(
            frequency=customer_metrics['frequency'],
            recency=customer_metrics['recency'],
            T=customer_metrics['T']
        )
        
        # Fit Gamma-Gamma model
        self.gg_model.fit(
            frequency=customer_metrics['frequency'],
            monetary_value=customer_metrics['monetary_value']
        )
        
        self.fitted = True
        return self
    
    def predict_ltv(self, data: pd.DataFrame, time_horizon: int = 90) -> np.ndarray:
        """Predict LTV for customers."""
        if not self.fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        customer_metrics = self._calculate_customer_metrics(data)
        
        # Predict future purchases
        future_purchases = self.bgnbd_model.predict(
            time_horizon,
            customer_metrics['frequency'],
            customer_metrics['recency'],
            customer_metrics['T']
        )
        
        # Predict average order value
        avg_order_value = self.gg_model.conditional_expected_average_profit(
            customer_metrics['frequency'],
            customer_metrics['monetary_value']
        )
        
        # Calculate LTV
        ltv = future_purchases * avg_order_value
        
        return ltv.values
    
    def _calculate_customer_metrics(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate customer-level metrics for BG/NBD model."""
        # This is a simplified implementation
        # In practice, you'd need to calculate proper frequency, recency, T, and monetary_value
        
        # For now, create dummy metrics based on available data
        metrics = data.groupby('customer_id').agg({
            'num_orders': 'sum',
            'total_spend': 'sum',
            'ltv_90d': 'first'
        }).reset_index()
        
        # Create dummy metrics (this should be replaced with proper calculation)
        metrics['frequency'] = metrics['num_orders'] - 1  # Frequency = repeat purchases
        metrics['recency'] = 30  # Dummy recency
        metrics['T'] = 90  # Dummy T
        metrics['monetary_value'] = metrics['total_spend'] / metrics['num_orders']
        
        return metrics


class BaselineEnsemble:
    """Ensemble of baseline models."""
    
    def __init__(self, models: List[Any], weights: Optional[List[float]] = None):
        self.models = models
        self.weights = weights or [1.0 / len(models)] * len(models)
        
        if len(self.weights) != len(self.models):
            raise ValueError("Number of weights must match number of models")
    
    def fit(self, X: np.ndarray, y: np.ndarray, X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None):
        """Fit all models in the ensemble."""
        for model in self.models:
            model.fit(X, y, X_val, y_val)
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make weighted ensemble predictions."""
        predictions = []
        for model in self.models:
            pred = model.predict(X)
            predictions.append(pred)
        
        # Weighted average
        weighted_pred = np.zeros_like(predictions[0])
        for pred, weight in zip(predictions, self.weights):
            weighted_pred += weight * pred
        
        return weighted_pred


def evaluate_baselines(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    task: str = "regression"
) -> Dict[str, Dict[str, float]]:
    """Evaluate multiple baseline models."""
    results = {}
    
    # XGBoost
    logger.info("Training XGBoost baseline...")
    xgb_model = XGBoostBaseline(task=task)
    xgb_model.fit(X_train, y_train, X_val, y_val)
    xgb_pred = xgb_model.predict(X_test)
    
    if task == "regression":
        results['xgboost'] = {
            'rmse': np.sqrt(mean_squared_error(y_test, xgb_pred)),
            'mae': np.mean(np.abs(y_test - xgb_pred))
        }
    else:
        results['xgboost'] = {
            'auc': roc_auc_score(y_test, xgb_pred),
            'f1': f1_score(y_test, xgb_pred > 0.5)
        }
    
    # CatBoost
    logger.info("Training CatBoost baseline...")
    cb_model = CatBoostBaseline(task=task)
    cb_model.fit(X_train, y_train, X_val, y_val)
    cb_pred = cb_model.predict(X_test)
    
    if task == "regression":
        results['catboost'] = {
            'rmse': np.sqrt(mean_squared_error(y_test, cb_pred)),
            'mae': np.mean(np.abs(y_test - cb_pred))
        }
    else:
        results['catboost'] = {
            'auc': roc_auc_score(y_test, cb_pred),
            'f1': f1_score(y_test, cb_pred > 0.5)
        }
    
    # Ensemble
    logger.info("Training ensemble baseline...")
    ensemble = BaselineEnsemble([xgb_model, cb_model])
    ensemble_pred = ensemble.predict(X_test)
    
    if task == "regression":
        results['ensemble'] = {
            'rmse': np.sqrt(mean_squared_error(y_test, ensemble_pred)),
            'mae': np.mean(np.abs(y_test - ensemble_pred))
        }
    else:
        results['ensemble'] = {
            'auc': roc_auc_score(y_test, ensemble_pred),
            'f1': f1_score(y_test, ensemble_pred > 0.5)
        }
    
    return results 