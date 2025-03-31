"""
Model manager implementation for efficient AI/ML model loading and inference.
"""

import os
import logging
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
import time
import threading
import numpy as np
import joblib
from functools import lru_cache

import torch
from app.config import MODELS_DIR

logger = logging.getLogger(__name__)

class ModelManager:
    """
    Manager for AI/ML models with efficient loading, caching, and inference.
    
    This class handles:
    1. Loading models on demand and caching them in memory
    2. Periodic reloading of models for updates
    3. GPU acceleration where available
    4. Batch inference for efficient prediction
    """

    def __init__(self, models_dir: Union[str, Path] = None):
        """
        Initialize the model manager.
        
        Args:
            models_dir: Directory where models are stored
        """
        self.models_dir = Path(models_dir or MODELS_DIR)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Model cache
        self._models: Dict[str, Dict[str, Any]] = {}
        self._model_lock = threading.RLock()
        
        # Available hardware acceleration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Model manager initialized with device: {self.device}")

    def get_model(self, model_name: str, model_type: str, version: str = "latest") -> Any:
        """
        Get a model from cache or load it from disk.
        
        Args:
            model_name: Name of the model
            model_type: Type of model (lstm, prophet, sentiment, etc.)
            version: Model version
            
        Returns:
            The loaded model
        """
        model_key = f"{model_type}/{model_name}/{version}"
        
        # Check if model is already loaded
        with self._model_lock:
            if model_key in self._models:
                logger.debug(f"Using cached model: {model_key}")
                # Update last access time
                self._models[model_key]["last_access"] = time.time()
                return self._models[model_key]["model"]
        
        # Model not loaded, load it
        logger.info(f"Loading model: {model_key}")
        model = self._load_model(model_name, model_type, version)
        
        # Cache the model
        with self._model_lock:
            self._models[model_key] = {
                "model": model,
                "load_time": time.time(),
                "last_access": time.time()
            }
        
        return model

    def _load_model(self, model_name: str, model_type: str, version: str) -> Any:
        """
        Load a model from disk.
        
        Args:
            model_name: Name of the model
            model_type: Type of model (lstm, prophet, sentiment, etc.)
            version: Model version
            
        Returns:
            The loaded model
        """
        if version == "latest":
            # Find the latest version
            model_dir = self.models_dir / model_type
            if not model_dir.exists():
                raise FileNotFoundError(f"Model type directory not found: {model_dir}")
            
            # Look for model files with version numbers
            model_files = list(model_dir.glob(f"{model_name}_v*.joblib"))
            if not model_files:
                raise FileNotFoundError(f"No models found for {model_name} in {model_dir}")
            
            # Sort by version number and get the latest
            model_files.sort(key=lambda x: x.stem.split('_v')[-1])
            model_path = model_files[-1]
        else:
            # Use the specified version
            model_path = self.models_dir / model_type / f"{model_name}_v{version}.joblib"
            if not model_path.exists():
                raise FileNotFoundError(f"Model not found: {model_path}")
        
        # Load the model using joblib
        model = joblib.load(model_path)
        
        # Move model to correct device if it's a PyTorch model
        if hasattr(model, "to") and callable(getattr(model, "to")):
            model = model.to(self.device)
        
        return model

    def predict(self, model_name: str, model_type: str, data: Any, version: str = "latest") -> Any:
        """
        Make predictions using a model.
        
        Args:
            model_name: Name of the model
            model_type: Type of model (lstm, prophet, sentiment, etc.)
            data: Input data for prediction
            version: Model version
            
        Returns:
            Model predictions
        """
        model = self.get_model(model_name, model_type, version)
        
        # Prepare data
        if hasattr(model, "predict") and callable(getattr(model, "predict")):
            # scikit-learn style interface
            return model.predict(data)
        elif hasattr(model, "forward") and callable(getattr(model, "forward")):
            # PyTorch model
            with torch.no_grad():
                # Convert data to tensor if it's not already
                if not isinstance(data, torch.Tensor):
                    data = torch.tensor(data, device=self.device)
                return model(data)
        else:
            raise ValueError(f"Model {model_name} doesn't have a standard prediction interface")

    def clean_cache(self, max_age_minutes: int = 30) -> None:
        """
        Clean the model cache by removing models that haven't been used recently.
        
        Args:
            max_age_minutes: Maximum time in minutes since last access
        """
        current_time = time.time()
        with self._model_lock:
            models_to_remove = []
            for model_key, model_info in self._models.items():
                last_access = model_info.get("last_access", 0)
                if current_time - last_access > max_age_minutes * 60:
                    models_to_remove.append(model_key)
            
            for model_key in models_to_remove:
                logger.info(f"Removing unused model from cache: {model_key}")
                del self._models[model_key]

# Singleton instance
_model_manager: Optional[ModelManager] = None

def get_model_manager() -> ModelManager:
    """
    Get the singleton model manager instance.
    
    Returns:
        ModelManager: The singleton instance
    """
    global _model_manager
    if _model_manager is None:
        _model_manager = ModelManager()
    return _model_manager 