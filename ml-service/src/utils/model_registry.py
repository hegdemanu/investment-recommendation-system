#!/usr/bin/env python3
"""
Model Registry for Investment Recommendation System
Handles model storage, retrieval, and metadata management
"""

import os
from pathlib import Path
import json
import logging
import uuid
import datetime
from typing import Dict, List, Optional, Any, Union
import pickle
import tensorflow as tf

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelRegistry:
    def __init__(self, registry_path: str = None):
        """Initialize model registry with path to storage directory"""
        self.registry_path = registry_path or os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "models"
        )
        self._ensure_registry_exists()
        self.metadata_file = os.path.join(self.registry_path, "metadata.json")
        self.metadata = self._load_metadata()
    
    def _ensure_registry_exists(self):
        """Ensure the registry directory exists"""
        os.makedirs(self.registry_path, exist_ok=True)
    
    def _load_metadata(self) -> Dict:
        """Load metadata from file or create if it doesn't exist"""
        if os.path.exists(self.metadata_file):
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading metadata: {e}")
                return {"models": []}
        else:
            return {"models": []}
    
    def _save_metadata(self):
        """Save metadata to file"""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving metadata: {e}")
    
    def register_model(
        self,
        model_type: str,
        model_object: Any,
        symbol: str,
        metrics: Dict = None,
        hyperparameters: Dict = None,
        features: List[str] = None
    ) -> str:
        """Register a model in the registry"""
        model_id = str(uuid.uuid4())
        timestamp = datetime.datetime.now().isoformat()
        
        # Create model directory
        model_dir = os.path.join(self.registry_path, model_id)
        os.makedirs(model_dir, exist_ok=True)
        
        # Save model
        model_path = os.path.join(model_dir, "model")
        self._save_model(model_object, model_path, model_type)
        
        # Create metadata entry
        model_metadata = {
            "id": model_id,
            "timestamp": timestamp,
            "symbol": symbol,
            "modelType": model_type,
            "metrics": metrics or {},
            "hyperparameters": hyperparameters or {},
            "features": features or [],
            "path": model_path
        }
        
        # Add to metadata
        self.metadata["models"].append(model_metadata)
        self._save_metadata()
        
        return model_id
    
    def _save_model(self, model_object: Any, model_path: str, model_type: str):
        """Save model to disk based on its type"""
        try:
            if model_type == "LSTM":
                # Save TensorFlow model
                model_object.save(model_path)
            elif model_type == "ARIMA":
                # Save ARIMA+GARCH model using pickle
                with open(f"{model_path}.pkl", 'wb') as f:
                    pickle.dump(model_object, f)
            elif model_type == "PROPHET":
                # Save Prophet model using pickle
                with open(f"{model_path}.pkl", 'wb') as f:
                    pickle.dump(model_object, f)
            elif model_type == "SENTIMENT":
                # Save sentiment model and tokenizer
                model_object["model"].save_pretrained(model_path)
                model_object["tokenizer"].save_pretrained(model_path)
            else:
                # Default to pickle for unknown model types
                with open(f"{model_path}.pkl", 'wb') as f:
                    pickle.dump(model_object, f)
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise
    
    def _load_model(self, model_path: str, model_type: str) -> Any:
        """Load model from disk based on its type"""
        try:
            if model_type == "LSTM":
                # Load TensorFlow model
                return tf.keras.models.load_model(model_path)
            elif model_type == "ARIMA" or model_type == "PROPHET":
                # Load pickle model
                with open(f"{model_path}.pkl", 'rb') as f:
                    return pickle.load(f)
            elif model_type == "SENTIMENT":
                # This is a placeholder - in actual implementation,
                # you would load the model and tokenizer using appropriate library
                return {
                    "model": None,
                    "tokenizer": None
                }
            else:
                # Default to pickle for unknown model types
                with open(f"{model_path}.pkl", 'rb') as f:
                    return pickle.load(f)
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return None
    
    def get_model(self, model_id: str) -> Dict:
        """Get a model by ID"""
        for model in self.metadata["models"]:
            if model["id"] == model_id:
                # Load the actual model
                model_object = self._load_model(model["path"], model["modelType"])
                if model_object is not None:
                    result = model.copy()
                    result["model"] = model_object
                    return result
                else:
                    return None
        return None
    
    def get_latest_model(self, symbol: str, model_type: str) -> Optional[Dict]:
        """Get the latest model for a symbol and type"""
        models = [
            model for model in self.metadata["models"]
            if model["symbol"] == symbol and model["modelType"] == model_type
        ]
        
        if not models:
            return None
            
        # Sort by timestamp and get the latest
        latest_model = sorted(
            models,
            key=lambda x: x["timestamp"],
            reverse=True
        )[0]
        
        # Load the actual model
        model_object = self._load_model(latest_model["path"], latest_model["modelType"])
        if model_object is not None:
            result = latest_model.copy()
            result["model"] = model_object
            return result
        
        return None
    
    def get_models_by_symbol(self, symbol: str) -> List[Dict]:
        """Get all models for a symbol"""
        return [
            {k: v for k, v in model.items() if k != "path"}
            for model in self.metadata["models"]
            if model["symbol"] == symbol
        ]
    
    def delete_model(self, model_id: str) -> bool:
        """Delete a model by ID"""
        for i, model in enumerate(self.metadata["models"]):
            if model["id"] == model_id:
                # Delete model files
                model_path = model["path"]
                self._delete_model_files(model_path, model["modelType"])
                
                # Remove from metadata
                self.metadata["models"].pop(i)
                self._save_metadata()
                return True
        return False
    
    def _delete_model_files(self, model_path: str, model_type: str):
        """Delete model files based on type"""
        try:
            if model_type == "LSTM":
                # Delete TensorFlow model directory
                import shutil
                if os.path.exists(model_path):
                    shutil.rmtree(model_path)
            else:
                # Delete pickle file
                pickle_path = f"{model_path}.pkl"
                if os.path.exists(pickle_path):
                    os.remove(pickle_path)
        except Exception as e:
            logger.error(f"Error deleting model files: {e}")
    
    def compare_models(self, symbol: str) -> Dict:
        """Compare models for a symbol and determine the best one"""
        models = self.get_models_by_symbol(symbol)
        if not models:
            return {
                "symbol": symbol,
                "models": [],
                "recommendedModel": None,
                "comparisonMetrics": {}
            }
        
        # Group models by type
        model_types = {}
        for model in models:
            model_type = model["modelType"]
            if model_type not in model_types:
                model_types[model_type] = []
            model_types[model_type].append(model)
        
        # Get the latest model for each type
        latest_models = {}
        for model_type, models_list in model_types.items():
            latest = sorted(
                models_list,
                key=lambda x: x["timestamp"],
                reverse=True
            )[0]
            latest_models[model_type] = latest
        
        # Compare models based on accuracy
        comparison = {}
        recommended = None
        max_accuracy = -1
        
        for model_type, model in latest_models.items():
            accuracy = model.get("metrics", {}).get("accuracy", 0)
            comparison[model_type] = {
                "accuracy": accuracy,
                "timestamp": model["timestamp"],
                "id": model["id"]
            }
            
            if accuracy > max_accuracy:
                max_accuracy = accuracy
                recommended = model_type
        
        return {
            "symbol": symbol,
            "models": list(latest_models.values()),
            "recommendedModel": recommended,
            "comparisonMetrics": comparison
        }
    
    def get_model_types(self) -> List[str]:
        """Get all available model types"""
        return list(set(model["modelType"] for model in self.metadata["models"]))
    
    def get_symbols(self) -> List[str]:
        """Get all symbols with models"""
        return list(set(model["symbol"] for model in self.metadata["models"]))
    
    def get_stats(self) -> Dict:
        """Get registry statistics"""
        models = self.metadata["models"]
        symbols = self.get_symbols()
        model_types = self.get_model_types()
        
        return {
            "totalModels": len(models),
            "symbols": len(symbols),
            "modelTypes": model_types,
            "symbolsList": symbols,
            "latestUpdate": max([model["timestamp"] for model in models]) if models else None
        } 