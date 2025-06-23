# Whale LTV Transformer package

# Core modules
from . import data_prep
from . import utils
from . import train

# Models
from .models import transformer
from .models import datamodule
from .models import baselines

# Main classes for easy importing
from .data_prep import OlistDataProcessor
from .models.transformer import WhaleLTVTransformer
from .models.datamodule import WhaleLTVDataModule
from .models.baselines import evaluate_baselines
from .utils import extract_features, split_data, calculate_metrics, create_sequence_tensor

__version__ = "1.0.0"
__author__ = "Juan Franceschini"

__all__ = [
    # Core modules
    "data_prep",
    "utils", 
    "train",
    
    # Model modules
    "transformer",
    "datamodule", 
    "baselines",
    
    # Main classes
    "OlistDataProcessor",
    "WhaleLTVTransformer",
    "WhaleLTVDataModule", 
    "evaluate_baselines",
    
    # Utility functions
    "extract_features",
    "split_data",
    "calculate_metrics",
    "create_sequence_tensor"
] 