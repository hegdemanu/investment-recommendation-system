"""
File Utilities Module

This module provides utility functions for file operations such as 
reading, writing, and managing files within the system.
"""
import os
import json
import csv
import logging
import shutil
from datetime import datetime
from pathlib import Path

# Set up logging
logger = logging.getLogger(__name__)

def ensure_directory(directory_path):
    """
    Ensure the specified directory exists, creating it if necessary.
    
    Args:
        directory_path (str): Path to the directory
        
    Returns:
        bool: True if directory exists or was created successfully, False otherwise
    """
    try:
        os.makedirs(directory_path, exist_ok=True)
        return True
    except Exception as e:
        logger.error(f"Failed to create directory {directory_path}: {str(e)}")
        return False

def save_json(file_path, data, indent=2):
    """
    Save data as JSON to the specified file path.
    
    Args:
        file_path (str): Path to save the JSON file
        data (dict): Data to save as JSON
        indent (int): JSON indentation level
        
    Returns:
        bool: True if save was successful, False otherwise
    """
    try:
        # Ensure directory exists
        directory = os.path.dirname(file_path)
        ensure_directory(directory)
        
        # Write JSON data
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent)
        
        logger.info(f"JSON data saved to {file_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to save JSON to {file_path}: {str(e)}")
        return False

def load_json(file_path, default=None):
    """
    Load JSON data from the specified file path.
    
    Args:
        file_path (str): Path to the JSON file
        default: Default value to return if loading fails
        
    Returns:
        dict: Loaded JSON data or default value if loading fails
    """
    try:
        if not os.path.exists(file_path):
            logger.warning(f"JSON file not found: {file_path}")
            return default
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        logger.info(f"JSON data loaded from {file_path}")
        return data
    except Exception as e:
        logger.error(f"Failed to load JSON from {file_path}: {str(e)}")
        return default

def save_csv(file_path, data, header=None, mode='w'):
    """
    Save data as CSV to the specified file path.
    
    Args:
        file_path (str): Path to save the CSV file
        data (list): List of rows to save
        header (list, optional): List of column headers
        mode (str): File open mode ('w' for write, 'a' for append)
        
    Returns:
        bool: True if save was successful, False otherwise
    """
    try:
        # Ensure directory exists
        directory = os.path.dirname(file_path)
        ensure_directory(directory)
        
        # Write CSV data
        with open(file_path, mode, newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            if header and mode == 'w':
                writer.writerow(header)
            
            writer.writerows(data)
        
        logger.info(f"CSV data saved to {file_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to save CSV to {file_path}: {str(e)}")
        return False

def load_csv(file_path, has_header=True, default=None):
    """
    Load CSV data from the specified file path.
    
    Args:
        file_path (str): Path to the CSV file
        has_header (bool): Whether the CSV has a header row
        default: Default value to return if loading fails
        
    Returns:
        tuple: (header, rows) if has_header is True, else (None, rows)
              Returns default if loading fails
    """
    try:
        if not os.path.exists(file_path):
            logger.warning(f"CSV file not found: {file_path}")
            return default
        
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            
            if has_header:
                header = next(reader)
                rows = list(reader)
                return header, rows
            else:
                rows = list(reader)
                return None, rows
    except Exception as e:
        logger.error(f"Failed to load CSV from {file_path}: {str(e)}")
        return default

def save_text(file_path, content, mode='w'):
    """
    Save text content to the specified file path.
    
    Args:
        file_path (str): Path to save the text file
        content (str): Text content to save
        mode (str): File open mode ('w' for write, 'a' for append)
        
    Returns:
        bool: True if save was successful, False otherwise
    """
    try:
        # Ensure directory exists
        directory = os.path.dirname(file_path)
        ensure_directory(directory)
        
        # Write text content
        with open(file_path, mode, encoding='utf-8') as f:
            f.write(content)
        
        logger.info(f"Text content saved to {file_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to save text to {file_path}: {str(e)}")
        return False

def load_text(file_path, default=''):
    """
    Load text content from the specified file path.
    
    Args:
        file_path (str): Path to the text file
        default (str): Default value to return if loading fails
        
    Returns:
        str: Loaded text content or default value if loading fails
    """
    try:
        if not os.path.exists(file_path):
            logger.warning(f"Text file not found: {file_path}")
            return default
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        logger.info(f"Text content loaded from {file_path}")
        return content
    except Exception as e:
        logger.error(f"Failed to load text from {file_path}: {str(e)}")
        return default

def file_exists(file_path):
    """
    Check if a file exists.
    
    Args:
        file_path (str): Path to the file
        
    Returns:
        bool: True if file exists, False otherwise
    """
    return os.path.isfile(file_path)

def get_file_info(file_path):
    """
    Get information about a file.
    
    Args:
        file_path (str): Path to the file
        
    Returns:
        dict: File information including name, size, creation time, etc.
             Returns None if file doesn't exist
    """
    try:
        if not os.path.isfile(file_path):
            logger.warning(f"File not found: {file_path}")
            return None
        
        file_stats = os.stat(file_path)
        
        file_info = {
            'name': os.path.basename(file_path),
            'path': os.path.abspath(file_path),
            'size': file_stats.st_size,
            'size_readable': get_readable_file_size(file_stats.st_size),
            'created': datetime.fromtimestamp(file_stats.st_ctime).strftime('%Y-%m-%d %H:%M:%S'),
            'modified': datetime.fromtimestamp(file_stats.st_mtime).strftime('%Y-%m-%d %H:%M:%S'),
            'extension': os.path.splitext(file_path)[1].lower()
        }
        
        return file_info
    except Exception as e:
        logger.error(f"Failed to get file info for {file_path}: {str(e)}")
        return None

def get_readable_file_size(size_in_bytes):
    """
    Convert file size in bytes to a human-readable format.
    
    Args:
        size_in_bytes (int): File size in bytes
        
    Returns:
        str: Human-readable file size
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_in_bytes < 1024.0:
            return f"{size_in_bytes:.2f} {unit}"
        size_in_bytes /= 1024.0
    
    return f"{size_in_bytes:.2f} PB"

def file_to_base64(file_path):
    """
    Convert a file to base64 encoded string.
    
    Args:
        file_path (str): Path to the file
        
    Returns:
        str: Base64 encoded string or empty string if conversion fails
    """
    try:
        import base64
        
        if not os.path.isfile(file_path):
            logger.warning(f"File not found for base64 encoding: {file_path}")
            return ""
        
        with open(file_path, 'rb') as f:
            encoded = base64.b64encode(f.read()).decode('utf-8')
        
        return encoded
    except Exception as e:
        logger.error(f"Failed to convert file to base64 {file_path}: {str(e)}")
        return ""

def copy_file(source_path, destination_path, overwrite=False):
    """
    Copy a file from source to destination.
    
    Args:
        source_path (str): Path to the source file
        destination_path (str): Path to the destination file
        overwrite (bool): Whether to overwrite existing destination file
        
    Returns:
        bool: True if copy was successful, False otherwise
    """
    try:
        if not os.path.isfile(source_path):
            logger.warning(f"Source file not found: {source_path}")
            return False
        
        if os.path.exists(destination_path) and not overwrite:
            logger.warning(f"Destination file already exists: {destination_path}")
            return False
        
        # Ensure destination directory exists
        destination_dir = os.path.dirname(destination_path)
        ensure_directory(destination_dir)
        
        # Copy file
        shutil.copy2(source_path, destination_path)
        
        logger.info(f"File copied from {source_path} to {destination_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to copy file from {source_path} to {destination_path}: {str(e)}")
        return False 