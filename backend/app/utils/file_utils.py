"""
File utilities for the investment recommendation system.

This module provides utility functions for file operations.
"""
import os
import json
import csv
import logging
from pathlib import Path
from typing import Dict, List, Any, Union, Optional

# Set up logging
logger = logging.getLogger(__name__)

def ensure_directory(directory_path: Union[str, Path]) -> Path:
    """
    Ensure directory exists, create if it doesn't
    
    Args:
        directory_path: Path to the directory
        
    Returns:
        Path object to the directory
    """
    dir_path = Path(directory_path)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path

def save_json(data: Dict[str, Any], filepath: Union[str, Path]) -> bool:
    """
    Save data as JSON file
    
    Args:
        data: Dictionary to save as JSON
        filepath: Path to save the file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        dir_path = ensure_directory(os.path.dirname(filepath))
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        logger.info(f"Saved JSON data to {filepath}")
        return True
    except Exception as e:
        logger.error(f"Error saving JSON data to {filepath}: {e}")
        return False

def load_json(filepath: Union[str, Path], default=None) -> Optional[Dict[str, Any]]:
    """
    Load data from JSON file
    
    Args:
        filepath: Path to the JSON file
        default: Default value to return if loading fails
        
    Returns:
        Dictionary with loaded data or default if failed
    """
    try:
        if not os.path.exists(filepath):
            logger.warning(f"JSON file not found: {filepath}")
            return default
            
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"Loaded JSON data from {filepath}")
        return data
    except Exception as e:
        logger.error(f"Error loading JSON from {filepath}: {e}")
        return default

def save_dataframe(df, filepath: Union[str, Path], format='csv') -> bool:
    """
    Save pandas DataFrame to file
    
    Args:
        df: pandas DataFrame to save
        filepath: Path to save the file
        format: Format to save (csv, json, excel)
        
    Returns:
        True if successful, False otherwise
    """
    try:
        dir_path = ensure_directory(os.path.dirname(filepath))
        
        if format.lower() == 'csv':
            df.to_csv(filepath, index=True)
        elif format.lower() == 'json':
            df.to_json(filepath, orient='records', date_format='iso')
        elif format.lower() in ['excel', 'xlsx', 'xls']:
            df.to_excel(filepath, index=True)
        else:
            logger.error(f"Unsupported format: {format}")
            return False
            
        logger.info(f"Saved DataFrame to {filepath} in {format} format")
        return True
    except Exception as e:
        logger.error(f"Error saving DataFrame to {filepath}: {e}")
        return False

def load_dataframe(filepath: Union[str, Path], format=None):
    """
    Load pandas DataFrame from file
    
    Args:
        filepath: Path to the file
        format: Format to load (csv, json, excel), inferred from extension if None
        
    Returns:
        pandas DataFrame or None if loading fails
    """
    try:
        import pandas as pd
        
        if not os.path.exists(filepath):
            logger.warning(f"File not found: {filepath}")
            return None
            
        # Infer format from file extension if not specified
        if format is None:
            ext = os.path.splitext(filepath)[1].lower()
            if ext in ['.csv']:
                format = 'csv'
            elif ext in ['.json']:
                format = 'json'
            elif ext in ['.xlsx', '.xls']:
                format = 'excel'
            else:
                logger.error(f"Could not infer format from extension: {ext}")
                return None
                
        # Load based on format
        if format.lower() == 'csv':
            df = pd.read_csv(filepath)
        elif format.lower() == 'json':
            df = pd.read_json(filepath)
        elif format.lower() in ['excel', 'xlsx', 'xls']:
            df = pd.read_excel(filepath)
        else:
            logger.error(f"Unsupported format: {format}")
            return None
            
        logger.info(f"Loaded DataFrame from {filepath}")
        return df
    except Exception as e:
        logger.error(f"Error loading DataFrame from {filepath}: {e}")
        return None

def save_text(content: str, filepath: Union[str, Path], mode='w') -> bool:
    """
    Save text content to a file
    
    Args:
        content: Text content to save
        filepath: Path to save the file
        mode: File open mode ('w' for write, 'a' for append)
        
    Returns:
        True if successful, False otherwise
    """
    try:
        dir_path = ensure_directory(os.path.dirname(filepath))
        with open(filepath, mode, encoding='utf-8') as f:
            f.write(content)
        logger.info(f"Saved text to {filepath}")
        return True
    except Exception as e:
        logger.error(f"Error saving text to {filepath}: {e}")
        return False

def load_text(filepath: Union[str, Path], default='') -> str:
    """
    Load text content from a file
    
    Args:
        filepath: Path to the text file
        default: Default value to return if loading fails
        
    Returns:
        Text content or default if failed
    """
    try:
        if not os.path.exists(filepath):
            logger.warning(f"Text file not found: {filepath}")
            return default
            
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        logger.info(f"Loaded text from {filepath}")
        return content
    except Exception as e:
        logger.error(f"Error loading text from {filepath}: {e}")
        return default

def file_exists(filepath: Union[str, Path]) -> bool:
    """
    Check if file exists
    
    Args:
        filepath: Path to the file
        
    Returns:
        True if file exists, False otherwise
    """
    return os.path.isfile(filepath) 