"""
Complexity prediction models for scene complexity estimation.
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import pickle
from typing import Dict, Tuple

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import setup_logging, load_config, set_seed


class ComplexityPredictor:
    """Base class for complexity prediction models."""
    
    def __init__(self, model_type='linear', **kwargs):
        """
        Initialize complexity predictor.
        
        Args:
            model_type: Type of model ('linear', 'random_forest', 'xgboost')
            **kwargs: Model-specific hyperparameters
        """
        self.model_type = model_type
        self.model = None
        self.feature_names = None
        self.scaler_mean = None
        self.scaler_std = None
        
        # Initialize model based on type
        if model_type == 'linear':
            self.model = LinearRegression(**kwargs)
        elif model_type == 'random_forest':
            self.model = RandomForestRegressor(**kwargs)
        elif model_type == 'xgboost':
            self.model = xgb.XGBRegressor(**kwargs)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: np.ndarray = None, y_val: np.ndarray = None):
        """
        Train the model.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
        """
        # Normalize features
        self.scaler_mean = np.mean(X_train, axis=0)
        self.scaler_std = np.std(X_train, axis=0)
        self.scaler_std[self.scaler_std == 0] = 1.0  # Avoid division by zero
        
        X_train_norm = (X_train - self.scaler_mean) / self.scaler_std
        
        # Train model
        if self.model_type == 'xgboost' and X_val is not None and y_val is not None:
            X_val_norm = (X_val - self.scaler_mean) / self.scaler_std
            self.model.fit(
                X_train_norm, y_train,
                eval_set=[(X_val_norm, y_val)],
                verbose=False
            )
        else:
            self.model.fit(X_train_norm, y_train)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict complexity scores.
        
        Args:
            X: Input features
            
        Returns:
            Predicted complexity scores
        """
        if self.scaler_mean is None or self.scaler_std is None:
            raise ValueError("Model not trained yet")
        
        X_norm = (X - self.scaler_mean) / self.scaler_std
        predictions = self.model.predict(X_norm)
        
        # Clip to [0, 1]
        return np.clip(predictions, 0.0, 1.0)
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Args:
            X: Input features
            y: True complexity scores
            
        Returns:
            Dictionary of metrics
        """
        y_pred = self.predict(X)
        
        metrics = {
            'mse': mean_squared_error(y, y_pred),
            'rmse': np.sqrt(mean_squared_error(y, y_pred)),
            'mae': mean_absolute_error(y, y_pred),
            'r2': r2_score(y, y_pred)
        }
        
        # Classification accuracy (for model selection)
        y_class_true = self._complexity_to_class(y)
        y_class_pred = self._complexity_to_class(y_pred)
        metrics['accuracy'] = np.mean(y_class_true == y_class_pred)
        
        return metrics
    
    def _complexity_to_class(self, complexity: np.ndarray) -> np.ndarray:
        """
        Convert complexity scores to class labels.
        
        Args:
            complexity: Complexity scores
            
        Returns:
            Class labels (0: skip, 1: light, 2: medium, 3: heavy)
        """
        classes = np.zeros(len(complexity), dtype=int)
        classes[complexity >= 0.1] = 1  # light
        classes[complexity >= 0.3] = 2  # medium
        classes[complexity >= 0.6] = 3  # heavy
        return classes
    
    def save(self, filepath: str):
        """Save model to file."""
        model_data = {
            'model_type': self.model_type,
            'model': self.model,
            'scaler_mean': self.scaler_mean,
            'scaler_std': self.scaler_std,
            'feature_names': self.feature_names
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    @classmethod
    def load(cls, filepath: str):
        """Load model from file."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        predictor = cls(model_type=model_data['model_type'])
        predictor.model = model_data['model']
        predictor.scaler_mean = model_data['scaler_mean']
        predictor.scaler_std = model_data['scaler_std']
        predictor.feature_names = model_data.get('feature_names')
        
        return predictor


def train_model(config, logger, model_type='linear', dataset='kitti'):
    """
    Train a complexity prediction model.
    
    Args:
        config: Configuration dictionary
        logger: Logger instance
        model_type: Type of model to train
        dataset: Dataset name
    """
    logger.info(f"Training {model_type} model on {dataset} dataset...")
    
    # Set random seed
    set_seed(config['training']['random_seed'])
    
    # Load processed data
    data_path = Path(config['output']['processed_data']) / f'{dataset}_processed.npz'
    
    if not data_path.exists():
        raise FileNotFoundError(f"Processed data not found: {data_path}")
    
    data = np.load(data_path, allow_pickle=True)
    X = data['X']
    y = data['y_complexity']
    train_idx = data['train_idx']
    val_idx = data['val_idx']
    test_idx = data['test_idx']
    
    # Split data
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    
    logger.info(f"Train size: {len(X_train)}, Val size: {len(X_val)}, Test size: {len(X_test)}")
    
    # Get hyperparameters from config
    model_config = config['models']['complexity'].get(model_type, {})
    hyperparams = model_config.get('hyperparameters', {})
    
    # Create and train model
    predictor = ComplexityPredictor(model_type=model_type, **hyperparams)
    predictor.feature_names = data['feature_names'].tolist()
    
    logger.info("Training model...")
    predictor.fit(X_train, y_train, X_val, y_val)
    
    # Evaluate
    logger.info("Evaluating model...")
    train_metrics = predictor.evaluate(X_train, y_train)
    val_metrics = predictor.evaluate(X_val, y_val)
    test_metrics = predictor.evaluate(X_test, y_test)
    
    # Log metrics
    logger.info("\n" + "="*50)
    logger.info(f"{model_type.upper()} MODEL RESULTS")
    logger.info("="*50)
    logger.info(f"Train - RMSE: {train_metrics['rmse']:.4f}, R²: {train_metrics['r2']:.4f}, Acc: {train_metrics['accuracy']:.4f}")
    logger.info(f"Val   - RMSE: {val_metrics['rmse']:.4f}, R²: {val_metrics['r2']:.4f}, Acc: {val_metrics['accuracy']:.4f}")
    logger.info(f"Test  - RMSE: {test_metrics['rmse']:.4f}, R²: {test_metrics['r2']:.4f}, Acc: {test_metrics['accuracy']:.4f}")
    logger.info("="*50)
    
    # Save model
    model_dir = Path(config['output']['models'])
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f'{model_type}_complexity_predictor.pkl'
    predictor.save(model_path)
    logger.info(f"Model saved to {model_path}")
    
    # Save metrics
    metrics_path = model_dir / f'{model_type}_metrics.pkl'
    with open(metrics_path, 'wb') as f:
        pickle.dump({
            'train': train_metrics,
            'val': val_metrics,
            'test': test_metrics
        }, f)
    logger.info(f"Metrics saved to {metrics_path}")
    
    return predictor, test_metrics


def main():
    """Main training function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train complexity prediction model')
    parser.add_argument('--model', type=str, default='random_forest',
                       choices=['linear', 'random_forest', 'xgboost'],
                       help='Model type to train')
    parser.add_argument('--dataset', type=str, default='kitti',
                       help='Dataset name')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Path to config file')
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Setup logging
    logger = setup_logging(config['output']['logs'])
    
    try:
        # Train model
        predictor, metrics = train_model(config, logger, args.model, args.dataset)
        
        logger.info(f"\n{args.model} model training completed successfully!")
        logger.info(f"Final test accuracy: {metrics['accuracy']:.2%}")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
