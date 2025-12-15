"""
Utility functions for the energy-aware perception scheduling system.
"""

import os
import yaml
import logging
import torch
import numpy as np
import random
from pathlib import Path
from typing import Dict, Any, Optional


def load_config(config_path: str = "configs/config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_logging(log_dir: str = "logs", log_level: int = logging.INFO) -> logging.Logger:
    """
    Setup logging configuration.
    
    Args:
        log_dir: Directory to save logs
        log_level: Logging level
        
    Returns:
        Configured logger
    """
    os.makedirs(log_dir, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger("energy_aware_av")
    logger.setLevel(log_level)
    
    # Create handlers
    console_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(os.path.join(log_dir, "training.log"))
    
    # Create formatters
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    # Add handlers
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger


def set_seed(seed: int = 42):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(device: Optional[str] = None) -> torch.device:
    """
    Get computing device (CUDA, MPS, or CPU).
    
    Args:
        device: Preferred device ('cuda', 'mps', 'cpu', or None for auto)
        
    Returns:
        torch.device object
    """
    if device is None:
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif torch.backends.mps.is_available():
            return torch.device('mps')
        else:
            return torch.device('cpu')
    return torch.device(device)


def create_directories(config: Dict[str, Any]):
    """
    Create necessary directories for the project.
    
    Args:
        config: Configuration dictionary
    """
    directories = [
        config['output']['processed_data'],
        config['output']['models'],
        config['output']['results'],
        config['output']['logs'],
        config['output']['checkpoints'],
        "data/raw/kitti/training/image_2",
        "data/raw/kitti/training/label_2",
        "data/raw/kitti/testing/image_2",
        "data/raw/bdd100k/images",
        "data/raw/bdd100k/labels",
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)


def save_checkpoint(model, optimizer, epoch, loss, filepath):
    """
    Save model checkpoint.
    
    Args:
        model: Model to save
        optimizer: Optimizer state
        epoch: Current epoch
        loss: Current loss
        filepath: Path to save checkpoint
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, filepath)


def load_checkpoint(filepath, model, optimizer=None):
    """
    Load model checkpoint.
    
    Args:
        filepath: Path to checkpoint file
        model: Model to load weights into
        optimizer: Optional optimizer to load state into
        
    Returns:
        Epoch and loss from checkpoint
    """
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint['epoch'], checkpoint['loss']


class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve."""
    
    def __init__(self, patience=10, min_delta=0.001, mode='min'):
        """
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'min' for loss, 'max' for accuracy
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score):
        """
        Check if training should stop.
        
        Args:
            score: Current metric value
            
        Returns:
            True if training should stop
        """
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == 'min':
            improved = score < (self.best_score - self.min_delta)
        else:
            improved = score > (self.best_score + self.min_delta)
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop


def normalize_features(features: np.ndarray, mean: Optional[np.ndarray] = None,
                      std: Optional[np.ndarray] = None):
    """
    Normalize features using z-score normalization.
    
    Args:
        features: Feature array
        mean: Precomputed mean (if None, compute from features)
        std: Precomputed std (if None, compute from features)
        
    Returns:
        Normalized features, mean, std
    """
    if mean is None:
        mean = np.mean(features, axis=0)
    if std is None:
        std = np.std(features, axis=0)
    
    # Avoid division by zero
    std[std == 0] = 1.0
    
    normalized = (features - mean) / std
    return normalized, mean, std


def calculate_energy_savings(baseline_energy: float, adaptive_energy: float) -> float:
    """
    Calculate energy savings percentage.
    
    Args:
        baseline_energy: Energy consumption with constant heavy processing
        adaptive_energy: Energy consumption with adaptive scheduling
        
    Returns:
        Percentage energy savings
    """
    if baseline_energy == 0:
        return 0.0
    return ((baseline_energy - adaptive_energy) / baseline_energy) * 100


def format_time(seconds: float) -> str:
    """
    Format seconds into readable string.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


if __name__ == "__main__":
    # Test utilities
    config = load_config()
    logger = setup_logging()
    logger.info("Utilities module loaded successfully")
    
    device = get_device()
    logger.info(f"Using device: {device}")
    
    create_directories(config)
    logger.info("Directories created successfully")
