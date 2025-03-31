"""
Configuration utility functions for the investment recommendation system.

This module provides utility functions for handling configuration settings
and environment variables.
"""
import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Union, Optional, TypeVar
import yaml

# Set up logging
logger = logging.getLogger(__name__)

# Define a generic type variable for the default return type
T = TypeVar('T')

class Config:
    """Configuration manager for the application."""
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Path to configuration file (JSON or YAML)
        """
        self._config = {}
        self._config_path = None
        
        # Load default configuration
        self._load_defaults()
        
        # Load configuration from file if provided
        if config_path:
            self.load_config(config_path)
    
    def _load_defaults(self):
        """Load default configuration settings."""
        self._config = {
            # General settings
            "app_name": "Investment Recommendation System",
            "debug_mode": False,
            "log_level": "INFO",
            
            # API settings
            "api": {
                "host": "0.0.0.0",
                "port": 8000,
                "workers": 4,
                "timeout": 60,
                "cors_origins": ["*"],
                "api_prefix": "/api/v1"
            },
            
            # Database settings
            "database": {
                "type": "sqlite",
                "sqlite_path": "data/investment.db",
                "postgres_dsn": "postgresql://user:password@localhost:5432/investment",
                "pool_size": 5,
                "max_overflow": 10,
                "use_orm": True
            },
            
            # Storage settings
            "storage": {
                "data_dir": "data",
                "results_dir": "results",
                "models_dir": "models",
                "cache_dir": "cache"
            },
            
            # Market data settings
            "market_data": {
                "default_source": "yfinance",
                "cache_expiry": 24,  # hours
                "default_period": "1y",
                "default_interval": "1d",
                "rate_limit": 0.5  # seconds between requests
            },
            
            # Model settings
            "models": {
                "sentiment": {
                    "model_type": "finbert",
                    "model_path": "models/finbert-sentiment",
                    "batch_size": 16,
                    "threshold": 0.5
                },
                "forecast": {
                    "model_type": "prophet",
                    "changepoint_prior_scale": 0.05,
                    "seasonality_prior_scale": 10.0,
                    "seasonality_mode": "multiplicative",
                    "mcmc_samples": 0,
                    "interval_width": 0.95
                },
                "technical": {
                    "indicators": ["RSI", "MACD", "Bollinger"],
                    "rsi_period": 14,
                    "macd_fast": 12,
                    "macd_slow": 26,
                    "macd_signal": 9,
                    "bb_period": 20,
                    "bb_std": 2
                }
            },
            
            # API keys
            "api_keys": {
                "newsapi": "",
                "alpha_vantage": "",
                "finnhub": ""
            },
            
            # Visualization settings
            "visualization": {
                "theme": "dark",
                "default_figsize": [12, 8],
                "dpi": 100,
                "save_format": ["png", "pdf"]
            }
        }
    
    def load_config(self, config_path: Union[str, Path]) -> bool:
        """
        Load configuration from file.
        
        Args:
            config_path: Path to configuration file (JSON or YAML)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            config_path = Path(config_path)
            self._config_path = config_path
            
            if not config_path.exists():
                logger.warning(f"Configuration file not found: {config_path}")
                return False
            
            # Load based on file extension
            suffix = config_path.suffix.lower()
            
            if suffix == '.json':
                with open(config_path, 'r') as f:
                    file_config = json.load(f)
            elif suffix in ['.yaml', '.yml']:
                with open(config_path, 'r') as f:
                    file_config = yaml.safe_load(f)
            else:
                logger.error(f"Unsupported configuration file format: {suffix}")
                return False
            
            # Update configuration with loaded values
            self._update_recursive(self._config, file_config)
            
            logger.info(f"Loaded configuration from {config_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading configuration from {config_path}: {str(e)}")
            return False
    
    def _update_recursive(self, target: Dict, source: Dict):
        """
        Recursively update nested dictionary.
        
        Args:
            target: Dictionary to update
            source: Dictionary with new values
        """
        for key, value in source.items():
            if isinstance(value, dict) and key in target and isinstance(target[key], dict):
                # Recursive update for nested dictionaries
                self._update_recursive(target[key], value)
            else:
                # Direct update for non-dictionary values
                target[key] = value
    
    def save_config(self, config_path: Optional[Union[str, Path]] = None, 
                   format: str = 'json') -> bool:
        """
        Save configuration to file.
        
        Args:
            config_path: Path to save configuration file
            format: Format to save ('json' or 'yaml')
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Use provided path or existing path
            save_path = Path(config_path) if config_path else self._config_path
            
            if not save_path:
                logger.error("No configuration path provided or set")
                return False
            
            # Create directory if it doesn't exist
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save based on format
            format = format.lower()
            
            if format == 'json':
                with open(save_path, 'w') as f:
                    json.dump(self._config, f, indent=2)
            elif format in ['yaml', 'yml']:
                with open(save_path, 'w') as f:
                    yaml.dump(self._config, f, default_flow_style=False, sort_keys=False)
            else:
                logger.error(f"Unsupported configuration format: {format}")
                return False
            
            logger.info(f"Saved configuration to {save_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving configuration: {str(e)}")
            return False
    
    def get(self, key: str, default: T = None) -> Union[Any, T]:
        """
        Get configuration value.
        
        Args:
            key: Dot-separated path to configuration value
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        try:
            # Split key into parts
            parts = key.split('.')
            value = self._config
            
            # Navigate through nested dictionaries
            for part in parts:
                if part not in value:
                    return default
                value = value[part]
            
            return value
            
        except Exception:
            return default
    
    def set(self, key: str, value: Any) -> bool:
        """
        Set configuration value.
        
        Args:
            key: Dot-separated path to configuration value
            value: Value to set
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Split key into parts
            parts = key.split('.')
            
            # Navigate to the parent dictionary
            current = self._config
            for part in parts[:-1]:
                if part not in current or not isinstance(current[part], dict):
                    current[part] = {}
                current = current[part]
            
            # Set the value
            current[parts[-1]] = value
            return True
            
        except Exception as e:
            logger.error(f"Error setting configuration value for {key}: {str(e)}")
            return False
    
    def get_all(self) -> Dict:
        """
        Get full configuration.
        
        Returns:
            Copy of the full configuration dictionary
        """
        return self._config.copy()
    
    def from_env(self, prefix: str = "APP_", include_unprefixed: bool = False) -> None:
        """
        Load configuration from environment variables.
        
        Args:
            prefix: Environment variable prefix to match
            include_unprefixed: Whether to include variables without prefix
        """
        for key, value in os.environ.items():
            if key.startswith(prefix) or (include_unprefixed and '_' in key):
                # Remove prefix if present
                if key.startswith(prefix):
                    config_key = key[len(prefix):].lower()
                else:
                    config_key = key.lower()
                
                # Replace underscores with dots for nested paths
                config_key = config_key.replace('__', '.').replace('_', '.')
                
                # Convert value types
                if value.lower() in ['true', 'yes', '1']:
                    typed_value = True
                elif value.lower() in ['false', 'no', '0']:
                    typed_value = False
                elif value.isdigit():
                    typed_value = int(value)
                elif value.replace('.', '', 1).isdigit() and value.count('.') == 1:
                    typed_value = float(value)
                else:
                    typed_value = value
                
                # Set the value
                self.set(config_key, typed_value)
        
        logger.info(f"Loaded configuration from environment variables with prefix '{prefix}'")

def load_settings(config_path: Optional[Union[str, Path]] = None,
                 env_prefix: str = "APP_") -> Config:
    """
    Load application settings.
    
    Args:
        config_path: Path to configuration file
        env_prefix: Environment variable prefix
        
    Returns:
        Config object with loaded settings
    """
    # Create config with defaults
    config = Config()
    
    # Load from file if provided
    if config_path:
        config.load_config(config_path)
    
    # Override with environment variables
    config.from_env(prefix=env_prefix)
    
    return config 