import os
import json
from typing import Dict, List, Optional
from datetime import datetime

class ModelRegistry:
    def __init__(self, base_dir: str = None):
        if base_dir is None:
            base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models")
        self.base_dir = base_dir
        self.models_info = self._load_models_info()
    
    def _load_models_info(self) -> Dict:
        """Load information about all available models"""
        models = {
            "lstm": {},
            "prophet": {},
            "sentiment": {}
        }
        
        # Load LSTM models
        lstm_dir = os.path.join(self.base_dir, "lstm")
        if os.path.exists(lstm_dir):
            for file in os.listdir(lstm_dir):
                if file.endswith("_metadata.json"):
                    symbol = file.replace("_metadata.json", "")
                    with open(os.path.join(lstm_dir, file), 'r') as f:
                        metadata = json.load(f)
                    models["lstm"][symbol] = {
                        "files": {
                            "model": f"{symbol}_lstm.h5",
                            "scaler": f"{symbol}_scaler.pkl",
                            "features": f"{symbol}_features.pkl",
                            "horizons": f"{symbol}_horizons.pkl",
                            "metadata": file
                        },
                        "metadata": metadata
                    }
        
        # Load Prophet models
        prophet_dir = os.path.join(self.base_dir, "prophet")
        if os.path.exists(prophet_dir):
            for file in os.listdir(prophet_dir):
                if file.endswith(".json"):
                    with open(os.path.join(prophet_dir, file), 'r') as f:
                        metadata = json.load(f)
                    symbol = metadata.get("symbol", file.replace(".json", ""))
                    models["prophet"][symbol] = {
                        "files": {"metadata": file},
                        "metadata": metadata
                    }
        
        # Load Sentiment models
        sentiment_dir = os.path.join(self.base_dir, "sentiment")
        if os.path.exists(sentiment_dir):
            for file in os.listdir(sentiment_dir):
                if file.endswith(".json"):
                    with open(os.path.join(sentiment_dir, file), 'r') as f:
                        metadata = json.load(f)
                    model_name = metadata.get("model_name", file.replace(".json", ""))
                    models["sentiment"][model_name] = {
                        "files": {"metadata": file},
                        "metadata": metadata
                    }
        
        return models
    
    def get_model_path(self, model_type: str, symbol: str, file_type: str) -> Optional[str]:
        """Get the full path for a model file"""
        if model_type not in self.models_info:
            return None
            
        model_info = self.models_info[model_type].get(symbol)
        if not model_info:
            return None
            
        file_name = model_info["files"].get(file_type)
        if not file_name:
            return None
            
        return os.path.join(self.base_dir, model_type, file_name)
    
    def get_model_metadata(self, model_type: str, symbol: str) -> Optional[Dict]:
        """Get metadata for a specific model"""
        if model_type not in self.models_info:
            return None
            
        model_info = self.models_info[model_type].get(symbol)
        return model_info["metadata"] if model_info else None
    
    def list_available_models(self, model_type: Optional[str] = None) -> Dict:
        """List all available models or models of a specific type"""
        if model_type:
            return {model_type: self.models_info.get(model_type, {})}
        return self.models_info
    
    def register_model(self, model_type: str, symbol: str, metadata: Dict, files: Dict[str, str]) -> bool:
        """Register a new model in the registry"""
        try:
            if model_type not in self.models_info:
                return False
                
            # Update metadata with registration time
            metadata["registered_at"] = datetime.now().isoformat()
            
            # Save metadata file
            metadata_path = os.path.join(self.base_dir, model_type, f"{symbol}_metadata.json")
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Update registry
            self.models_info[model_type][symbol] = {
                "files": files,
                "metadata": metadata
            }
            
            return True
            
        except Exception as e:
            print(f"Error registering model: {str(e)}")
            return False
    
    def unregister_model(self, model_type: str, symbol: str) -> bool:
        """Remove a model from the registry"""
        try:
            if model_type not in self.models_info:
                return False
                
            if symbol not in self.models_info[model_type]:
                return False
                
            # Remove model files
            model_info = self.models_info[model_type][symbol]
            for file_name in model_info["files"].values():
                file_path = os.path.join(self.base_dir, model_type, file_name)
                if os.path.exists(file_path):
                    os.remove(file_path)
            
            # Remove from registry
            del self.models_info[model_type][symbol]
            
            return True
            
        except Exception as e:
            print(f"Error unregistering model: {str(e)}")
            return False 