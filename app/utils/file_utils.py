"""
File Utility Functions

This module provides utility functions for working with files,
including file conversion, loading, and data processing.
"""
import os
import json
import base64
import re
from config.settings import RESULTS_DIR

def file_to_base64(file_path):
    """Convert a file to base64 for embedding in HTML."""
    try:
        with open(file_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode('utf-8')
            
        # Get the MIME type based on extension
        extension = os.path.splitext(file_path)[1].lower()
        mime_types = {
            '.png': 'image/png',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.svg': 'image/svg+xml',
            '.gif': 'image/gif',
            '.pdf': 'application/pdf',
        }
        
        mime_type = mime_types.get(extension, 'application/octet-stream')
        return f"data:{mime_type};base64,{encoded}"
    except Exception as e:
        print(f"Error encoding {file_path}: {e}")
        return None

def load_json_file(file_path):
    """Load a JSON file or return None if invalid."""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
            # Add better error handling for empty files
            if not content or content.isspace():
                print(f"Warning: Empty JSON file {file_path}")
                return {}
                
            try:
                json_data = json.loads(content)
                # Add debug info about loaded JSON
                if isinstance(json_data, dict):
                    print(f"Loaded JSON from {os.path.basename(file_path)}: {len(str(json_data))} chars, keys: {list(json_data.keys())}")
                else:
                    print(f"Loaded JSON from {os.path.basename(file_path)}: {len(str(json_data))} chars, not a dict")
                return json_data
            except json.JSONDecodeError as je:
                print(f"JSON decode error in {file_path}: {je}")
                
                # Log the problematic content for debugging
                print(f"Content preview: {content[:100]}{'...' if len(content) > 100 else ''}")
                
                # Try to identify the exact position of the error
                error_line = je.lineno
                error_col = je.colno
                content_lines = content.split('\n')
                if error_line <= len(content_lines):
                    error_line_content = content_lines[error_line - 1]
                    print(f"Error at line {error_line}, column {error_col}: {error_line_content}")
                    if error_col < len(error_line_content):
                        print(" " * (error_col + 24) + "^")
                
                # Try to salvage malformed JSON by first validating structure
                if content.strip().startswith('{') and not content.strip().endswith('}'):
                    print("JSON appears to be truncated (missing closing brace)")
                    # Try to add a closing brace if it seems truncated
                    fixed_content = content.strip() + '}'
                    try:
                        return json.loads(fixed_content)
                    except:
                        pass
                
                # Try to remove comments (which are not valid in JSON)
                cleaned_content = re.sub(r'//.*?\n', '\n', content)  # Remove single-line comments
                cleaned_content = re.sub(r'/\*.*?\*/', '', cleaned_content, flags=re.DOTALL)  # Remove multi-line comments
                
                # Try to salvage malformed JSON by replacing problematic characters
                cleaned_content = cleaned_content.replace('\n', ' ').replace('\r', '').replace('\t', ' ')
                try:
                    return json.loads(cleaned_content)
                except:
                    # If all else fails, return an empty dict with an error indicator
                    print(f"Could not parse JSON in {file_path} even after cleaning")
                    return {"error": "Failed to parse JSON file", "file": os.path.basename(file_path)}
    except Exception as e:
        print(f"Error loading JSON file {file_path}: {e}")
        return {"error": "Failed to load file", "file": os.path.basename(file_path), "message": str(e)}

def ensure_file_exists(file_path, default_content=""):
    """Ensure a file exists, creating it with default content if it doesn't."""
    if not os.path.exists(file_path):
        directory = os.path.dirname(file_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
        
        with open(file_path, 'w') as f:
            f.write(default_content)
        return False
    return True

def get_file_extension(file_path):
    """Get the extension of a file."""
    return os.path.splitext(file_path)[1].lower()

def is_image_file(file_path):
    """Check if a file is an image based on its extension."""
    image_extensions = ['.png', '.jpg', '.jpeg', '.gif', '.svg']
    return get_file_extension(file_path) in image_extensions

def is_data_file(file_path):
    """Check if a file is a data file based on its extension."""
    data_extensions = ['.csv', '.xlsx', '.xls', '.json']
    return get_file_extension(file_path) in data_extensions 