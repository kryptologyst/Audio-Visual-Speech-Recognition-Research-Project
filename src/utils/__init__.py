"""Utility functions for the AVSR project."""

import random
import numpy as np
import torch
import logging
from typing import Any, Dict, Optional, Union
from pathlib import Path

logger = logging.getLogger(__name__)


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Enable deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    logger.info(f"Random seed set to {seed}")


def get_device() -> torch.device:
    """Get the best available device (CUDA -> MPS -> CPU).
    
    Returns:
        PyTorch device object.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using CUDA device: {torch.cuda.get_device_name()}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using MPS device (Apple Silicon)")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU device")
    
    return device


def count_parameters(model: torch.nn.Module) -> int:
    """Count the number of trainable parameters in a model.
    
    Args:
        model: PyTorch model.
        
    Returns:
        Number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def format_time(seconds: float) -> str:
    """Format time in seconds to human-readable string.
    
    Args:
        seconds: Time in seconds.
        
    Returns:
        Formatted time string.
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def ensure_dir(path: Union[str, Path]) -> Path:
    """Ensure directory exists, create if it doesn't.
    
    Args:
        path: Directory path.
        
    Returns:
        Path object.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file.
        
    Returns:
        Configuration dictionary.
    """
    try:
        from omegaconf import OmegaConf
        config = OmegaConf.load(config_path)
        return OmegaConf.to_container(config, resolve=True)
    except ImportError:
        import yaml
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)


def save_config(config: Dict[str, Any], config_path: Union[str, Path]) -> None:
    """Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary.
        config_path: Path to save configuration.
    """
    try:
        from omegaconf import OmegaConf
        OmegaConf.save(config, config_path)
    except ImportError:
        import yaml
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)


class EarlyStopping:
    """Early stopping utility to prevent overfitting."""
    
    def __init__(self, patience: int = 7, min_delta: float = 0.0, restore_best_weights: bool = True):
        """Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait before stopping.
            min_delta: Minimum change to qualify as improvement.
            restore_best_weights: Whether to restore best weights when stopping.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_score = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_score: float, model: torch.nn.Module) -> bool:
        """Check if training should stop.
        
        Args:
            val_score: Current validation score.
            model: Model to potentially save weights from.
            
        Returns:
            True if training should stop, False otherwise.
        """
        if self.best_score is None:
            self.best_score = val_score
            self.save_checkpoint(model)
        elif val_score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                if self.restore_best_weights:
                    model.load_state_dict(self.best_weights)
                return True
        else:
            self.best_score = val_score
            self.counter = 0
            self.save_checkpoint(model)
        
        return False
    
    def save_checkpoint(self, model: torch.nn.Module) -> None:
        """Save model checkpoint.
        
        Args:
            model: Model to save.
        """
        self.best_weights = model.state_dict().copy()


def compute_rtf(inference_time: float, audio_duration: float) -> float:
    """Compute Real-Time Factor (RTF).
    
    Args:
        inference_time: Time taken for inference in seconds.
        audio_duration: Duration of audio in seconds.
        
    Returns:
        Real-time factor (RTF < 1.0 means faster than real-time).
    """
    return inference_time / audio_duration


def sanitize_filename(filename: str) -> str:
    """Sanitize filename for privacy-preserving logging.
    
    Args:
        filename: Original filename.
        
    Returns:
        Sanitized filename.
    """
    # Remove or hash potentially identifying information
    import hashlib
    path = Path(filename)
    stem_hash = hashlib.md5(path.stem.encode()).hexdigest()[:8]
    return f"{stem_hash}_{path.suffix}"


def log_model_info(model: torch.nn.Module, logger: logging.Logger) -> None:
    """Log model information.
    
    Args:
        model: PyTorch model.
        logger: Logger instance.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = count_parameters(model)
    
    logger.info(f"Model: {model.__class__.__name__}")
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    logger.info(f"Model size: {total_params * 4 / 1024 / 1024:.2f} MB")
