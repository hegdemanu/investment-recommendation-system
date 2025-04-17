#!/usr/bin/env python3
"""
Generate a comprehensive HTML dashboard report that embeds all reports.
This script creates a single HTML file that shows all reports in an organized way.
"""
import os
import json
import glob
import base64
from datetime import datetime
import webbrowser
import re

# Constants and directories setup
RESULTS_DIR = os.path.join(".", "results")
REPORTS_DIR = os.path.join(RESULTS_DIR, "reports")
MODELS_DIR = os.path.join(".", "models")  # Add models directory definition
OUTPUT_DIR = os.path.join(RESULTS_DIR, "dashboard")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "investment_dashboard.html")

# Dictionary of report directories to check
REPORTS_DIRS = {
    "training": os.path.join(RESULTS_DIR, "training"),
    "validation": os.path.join(RESULTS_DIR, "validation"),
    "reports": REPORTS_DIR,
    "models": MODELS_DIR  # Add models directory to the dictionary
}

def ensure_directories():
    """Ensure all directories exist."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for directory in REPORTS_DIRS.values():
        os.makedirs(directory, exist_ok=True)

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

def create_sample_json_file():
    """Create a sample JSON file for demonstration purposes."""
    sample_data = {
        "stocks": {
            "total": 5,
            "success": 4,
            "failed": 1,
            "details": {
                "AAPL": {
                    "status": "success",
                    "training_time": 45.2,
                    "model_path": "./models/AAPL_lstm.h5"
                },
                "MSFT": {
                    "status": "success",
                    "training_time": 38.7,
                    "model_path": "./models/MSFT_lstm.h5"
                },
                "GOOGL": {
                    "status": "success",
                    "training_time": 42.1,
                    "model_path": "./models/GOOGL_lstm.h5"
                },
                "AMZN": {
                    "status": "success",
                    "training_time": 40.5,
                    "model_path": "./models/AMZN_lstm.h5"
                },
                "TSLA": {
                    "status": "failed",
                    "error": "Insufficient data"
                }
            }
        }
    }
    
    # Create training directory if it doesn't exist
    training_dir = os.path.join(RESULTS_DIR, "training")
    os.makedirs(training_dir, exist_ok=True)
    
    # Save the sample JSON file
    sample_file_path = os.path.join(training_dir, "sample_training_summary.json")
    with open(sample_file_path, 'w') as f:
        json.dump(sample_data, f, indent=4)
    
    print(f"Created sample JSON file at {sample_file_path}")
    
    # Create a sample validation summary
    validation_data = {
        "AAPL": {
            "metrics": {
                "rmse": 2.34,
                "mae": 1.98,
                "r2": 0.87,
                "mape": 1.25
            },
            "best_horizon": 5
        },
        "MSFT": {
            "metrics": {
                "rmse": 1.87,
                "mae": 1.42,
                "r2": 0.91,
                "mape": 0.98
            },
            "best_horizon": 3
        }
    }
    
    # Save the sample validation file
    validation_file_path = os.path.join(RESULTS_DIR, "validation_summary.json")
    with open(validation_file_path, 'w') as f:
        json.dump(validation_data, f, indent=4)
    
    print(f"Created sample validation file at {validation_file_path}")
    
    return [sample_file_path, validation_file_path]

def create_sample_csv_file():
    """Create a sample CSV file for demonstration purposes."""
    sample_data = """date,open,high,low,close,volume,ticker
2023-01-01,150.23,153.45,149.87,152.54,12345678,AAPL
2023-01-02,152.76,155.21,151.89,154.32,23456789,AAPL
2023-01-03,154.55,156.78,153.21,155.98,34567890,AAPL
2023-01-04,156.01,158.43,155.67,157.76,45678901,AAPL
2023-01-05,157.89,159.34,156.54,158.92,56789012,AAPL
2023-01-06,159.01,160.75,158.32,160.45,67890123,AAPL
2023-01-07,160.32,162.43,159.87,161.78,78901234,AAPL
"""
    
    # Create data directory if it doesn't exist
    data_dir = os.path.join(".", "data")
    os.makedirs(data_dir, exist_ok=True)
    
    # Save the sample CSV file
    sample_file_path = os.path.join(data_dir, "sample_stock_data.csv")
    with open(sample_file_path, 'w') as f:
        f.write(sample_data)
    
    print(f"Created sample CSV file at {sample_file_path}")
    
    return sample_file_path

def generate_dashboard():
    """Generate the HTML dashboard."""
    # Check if we need to create sample files
    create_samples = False
    if not os.path.exists(os.path.join(RESULTS_DIR, "validation_summary.json")) and not os.path.exists(os.path.join(RESULTS_DIR, "training", "training_summary.json")):
        create_samples = True
    
    # Collect all image files (plots)
    image_files = []
    for ext in ['.png', '.jpg', '.jpeg', '.svg', '.gif']:
        for directory in REPORTS_DIRS.values():
            if os.path.exists(directory):
                found_images = glob.glob(os.path.join(directory, f"*{ext}"))
                image_files.extend(found_images)
                print(f"Found {len(found_images)} {ext} files in {directory}")
    
    # Collect all JSON report files
    json_files = []
    for directory in REPORTS_DIRS.values():
        if os.path.exists(directory):
            found_jsons = glob.glob(os.path.join(directory, "*.json"))
            json_files.extend(found_jsons)
            print(f"Found {len(found_jsons)} JSON files in {directory}")
    
    # Also look in the root results directory
    if os.path.exists(RESULTS_DIR):
        found_jsons = glob.glob(os.path.join(RESULTS_DIR, "*.json"))
        json_files.extend(found_jsons)
        print(f"Found {len(found_jsons)} JSON files in {RESULTS_DIR}")
    
    # Create sample files if needed
    if len(json_files) == 0:
        print("No JSON files found. Creating sample data files...")
        sample_json_files = create_sample_json_file()
        json_files.extend(sample_json_files)
        print(f"Created {len(sample_json_files)} sample JSON files")
    
    # Collect Excel and CSV files for raw data section
    raw_data_files = []
    
    # Use both recursive and non-recursive search to ensure we find all files
    data_locations = [
        os.path.join(".", "data"), 
        RESULTS_DIR,
        os.path.join(".", "data", "uploads"),
        os.path.join(".", "data", "stocks"),
        os.path.join(".", "data", "mutual_funds"),
        os.path.join(".", "data", "processed"),
        os.path.join(".", "data", "raw")
    ]

    for location in data_locations:
        if os.path.exists(location):
            # Non-recursive search first (for immediate files)
            for ext in ['.xlsx', '.xls', '.csv']:
                found_files = glob.glob(os.path.join(location, f"*{ext}"))
                raw_data_files.extend(found_files)
                print(f"Found {len(found_files)} {ext} files in {location}")
            
            # Then recursive search (for nested files)
            try:
                for root, _, files in os.walk(location):
                    for file in files:
                        if file.endswith(('.xlsx', '.xls', '.csv')):
                            full_path = os.path.join(root, file)
                            if full_path not in raw_data_files:
                                raw_data_files.append(full_path)
            except Exception as e:
                print(f"Error walking directory {location}: {e}")
    
    print(f"After recursive search: Found {len(raw_data_files)} total raw data files")
    
    # Print first 5 raw data files for debugging
    if raw_data_files:
        print("First 5 raw data files:")
        for i, file in enumerate(raw_data_files[:5]):
            print(f"  {i+1}. {file}")
    
    # Create sample CSV file if needed
    if len(raw_data_files) == 0:
        print("No raw data files found. Creating sample CSV file...")
        sample_csv_file = create_sample_csv_file()
        raw_data_files.append(sample_csv_file)
        print(f"Created sample CSV file: {sample_csv_file}")
    
    # Add validation summary if it exists
    validation_file = os.path.join(RESULTS_DIR, "validation_summary.json")
    if os.path.exists(validation_file):
        if validation_file not in json_files:
            json_files.append(validation_file)
            print(f"Added validation summary file: {validation_file}")
    else:
        print(f"Warning: Validation summary file not found at {validation_file}")
    
    # Print summary of findings
    print(f"Total files found: {len(image_files)} images, {len(json_files)} JSONs, {len(raw_data_files)} data files")
    
    # Add processed data content
    processed_data_items = ""
    # Count processed data files by getting JSON files that include 'processed' in path
    processed_data_files = [f for f in json_files if "processed" in f.lower() or "validation" in f.lower() or "training" in f.lower()]
    
    if processed_data_files:
        for data_file in processed_data_files:
            file_name = os.path.basename(data_file)
            file_path = data_file
            file_size = os.path.getsize(data_file) if os.path.exists(data_file) else 0
            file_size_str = f"{file_size / 1024:.1f} KB" if file_size > 0 else "N/A"
            
            # Get file type from path
            file_type = "other"
            if "validation" in data_file.lower():
                file_type = "validation"
                icon = "fas fa-check-circle"
            elif "training" in data_file.lower():
                file_type = "training"
                icon = "fas fa-cogs"
            elif "processed" in data_file.lower():
                file_type = "processed"
                icon = "fas fa-database"
            else:
                icon = "fas fa-file-alt"
            
            processed_data_items += f"""
                <div class="file-item" data-name="{file_name}" data-type="{file_type}" data-size="{file_size}">
                    <div class="file-icon"><i class="{icon}"></i></div>
                    <div class="file-name">{file_name}</div>
                    <div class="file-info">{file_type.capitalize()} • {file_size_str}</div>
                    <div class="file-actions">
                        <button class="button view-json" onclick="viewJsonFile('{file_path}')">
                            <i class="fas fa-eye"></i> View
                        </button>
                    </div>
                </div>
            """
        
        processed_data_warning = ""
    else:
        processed_data_warning = """
            <div class="card" style="margin-top: 20px;">
                <h3><i class="fas fa-exclamation-triangle"></i> No Processed Data Files Found</h3>
                <p>No processed data files (.json) were found in the following locations:</p>
                <ul>
                    <li><code>./results/</code></li>
                    <li><code>./results/training/</code></li>
                    <li><code>./results/validation/</code></li>
                    <li><code>./results/reports/</code></li>
                    <li><code>./models/</code></li>
                </ul>
                <p>To include processed data files:</p>
                <ol>
                    <li>Run model training and validation scripts which will generate JSON outputs</li>
                    <li>Ensure that model training summaries and validation metrics are being saved</li>
                    <li>Check that you have proper write permissions to the results directory</li>
                </ol>
                <div class="action-buttons" style="text-align: left;">
                    <a href="javascript:void(0)" onclick="runScript('train_all_models.py')" class="button">
                        <i class="fas fa-cogs"></i> Run Training
                    </a>
                    <a href="javascript:void(0)" onclick="runScript('validate_model.py')" class="button">
                        <i class="fas fa-check-circle"></i> Run Validation
                    </a>
                </div>
            </div>
        """
    
    # Find existing expert HTML reports
    expert_reports = []
    search_dirs = [
        RESULTS_DIR, 
        REPORTS_DIR,
        os.path.join(".", "data", "uploads"),
        os.path.join(".", "data", "stocks"),
        os.path.join(".", "data", "mutual_funds")
    ]

    for root_dir in search_dirs:
        if os.path.exists(root_dir):
            for dirpath, _, filenames in os.walk(root_dir):
                for filename in filenames:
                    if filename.lower().endswith(('.html', '.htm')):
                        full_path = os.path.join(dirpath, filename)
                        if "dashboard" not in filename.lower() and os.path.getsize(full_path) > 1024:
                            expert_reports.append(full_path)
    
    # Check for existing investment report
    investment_report_path = os.path.join(RESULTS_DIR, "reports", "investment_report.html")
    has_investment_report = os.path.exists(investment_report_path)
    
    # Integrate with existing investment report generation script
    generate_investment_report_path = "generate_investment_report.py"
    has_report_generator = os.path.exists(generate_investment_report_path)
    
    # Load training parameters
    training_params = {}
    timeframes_file = os.path.join(MODELS_DIR, "training_summary.json")
    if os.path.exists(timeframes_file):
        try:
            with open(timeframes_file, 'r') as f:
                training_params = json.load(f)
        except Exception as e:
            print(f"Error loading training parameters: {e}")
    
    # Start building HTML content
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Investment Recommendation System Dashboard</title>
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
        <style>
            :root {{
                --primary: #2c3e50;
                --secondary: #3498db;
                --success: #2ecc71;
                --warning: #f39c12;
                --danger: #e74c3c;
                --light: #ecf0f1;
                --dark: #2c3e50;
                --bg-light: #f8f9fa;
                --bg-dark: #343a40;
                --text-light: #f8f9fa;
                --text-dark: #343a40;
            }}
            
            * {{
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }}
            
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                line-height: 1.6;
                color: var(--text-dark);
                background-color: var(--bg-light);
                overflow-x: hidden;
                transition: background-color 0.3s ease;
            }}
            
            body.dark-mode {{
                background-color: var(--bg-dark);
                color: var(--text-light);
            }}
            
            .container {{
                width: 100%;
                max-width: 1400px;
                margin: 0 auto;
                padding: 0 15px;
            }}
            
            #navbar {{
                background-color: rgba(255, 255, 255, 0.9);
                position: sticky;
                top: 0;
                width: 100%;
                z-index: 1000;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                display: flex;
                justify-content: space-between;
                align-items: center;
                padding: 10px 20px;
                height: 70px;
                backdrop-filter: blur(10px);
                transition: all 0.3s ease;
            }}

            body.dark-mode #navbar {{
                background-color: rgba(52, 58, 64, 0.9);
                box-shadow: 0 2px 10px rgba(0,0,0,0.3);
            }}
            
            #navbar.scrolled {{
                height: 60px;
                padding: 5px 20px;
            }}
            
            .logo {{
                display: flex;
                align-items: center;
                font-weight: bold;
                font-size: 1.2rem;
                color: var(--primary);
                text-decoration: none;
            }}
            
            body.dark-mode .logo {{
                color: var(--light);
            }}
            
            .theme-toggle {{
                background: transparent;
                border: none;
                cursor: pointer;
                font-size: 1.2rem;
                color: var(--primary);
                margin-left: 10px;
                display: flex;
                align-items: center;
                justify-content: center;
                width: 40px;
                height: 40px;
                border-radius: 50%;
                transition: background-color 0.3s;
            }}
            
            .theme-toggle:hover {{
                background-color: rgba(0,0,0,0.05);
            }}
            
            body.dark-mode .theme-toggle {{
                color: var(--light);
            }}
            
            body.dark-mode .theme-toggle:hover {{
                background-color: rgba(255,255,255,0.1);
            }}
            
            .tabs {{
                display: flex;
                flex-wrap: nowrap;
                overflow-x: auto;
                scrollbar-width: none;
                gap: 5px;
                align-items: center;
                padding: 0 10px;
                flex: 1;
                margin: 0 20px;
                justify-content: center;
            }}
            
            .tabs::-webkit-scrollbar {{
                display: none;
            }}
            
            /* Fix for tab cursor issue */
            .tab {{
                cursor: pointer;
                padding: 10px 15px;
                border-radius: 4px;
                font-weight: 500;
                white-space: nowrap;
                transition: all 0.3s ease;
                display: flex;
                align-items: center;
                gap: 5px;
                color: var(--primary);
                background-color: transparent;
                user-select: none; /* Prevent text selection */
                -webkit-user-select: none; /* Safari */
                -moz-user-select: none; /* Firefox */
                -ms-user-select: none; /* IE10+/Edge */
            }}
            
            body.dark-mode .tab {{
                color: var(--light);
            }}
            
            .tab:hover {{
                background-color: rgba(0,0,0,0.05);
            }}
            
            body.dark-mode .tab:hover {{
                background-color: rgba(255,255,255,0.1);
            }}
            
            .tab.active {{
                background-color: var(--primary);
                color: white;
            }}
            
            body.dark-mode .tab.active {{
                background-color: var(--secondary);
            }}
            
            .tab i {{
                font-size: 1rem;
            }}
            
            /* Content Styles */
            .content {{
                padding: 20px 0;
            }}
            
            .tab-content {{
                display: none;
            }}
            
            .tab-content.active {{
                display: block;
                animation: fadeIn 0.5s ease;
            }}
            
            @keyframes fadeIn {{
                from {{ opacity: 0; }}
                to {{ opacity: 1; }}
            }}
            
            .section {{
                margin-bottom: 30px;
                background-color: white;
                border-radius: 8px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.05);
                overflow: hidden;
                transition: box-shadow 0.3s ease;
            }}
            
            body.dark-mode .section {{
                background-color: #424242;
                box-shadow: 0 2px 10px rgba(0,0,0,0.2);
            }}
            
            .section h2 {{
                padding: 20px;
                border-bottom: 1px solid #eee;
                margin-bottom: 0;
                color: var(--primary);
                display: flex;
                align-items: center;
                gap: 10px;
            }}
            
            body.dark-mode .section h2 {{
                border-bottom: 1px solid #555;
                color: var(--text-light);
            }}
            
            .section-content {{
                padding: 20px;
            }}
            
            .card {{
                background-color: white;
                border-radius: 8px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.05);
                margin-bottom: 20px;
                padding: 20px;
                transition: transform 0.3s ease, box-shadow 0.3s ease;
            }}
            
            body.dark-mode .card {{
                background-color: #333;
                box-shadow: 0 2px 5px rgba(0,0,0,0.2);
            }}
            
            .card:hover {{
                transform: translateY(-2px);
                box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            }}
            
            body.dark-mode .card:hover {{
                box-shadow: 0 4px 8px rgba(0,0,0,0.3);
            }}
            
            .card h3 {{
                margin-top: 0;
                margin-bottom: 15px;
                color: var(--primary);
                display: flex;
                align-items: center;
                gap: 8px;
            }}
            
            body.dark-mode .card h3 {{
                color: var(--text-light);
            }}
            
            /* Table Styles */
            .table-container {{
                margin: 20px 0;
                overflow-x: auto;
            }}
            
            table {{
                width: 100%;
                border-collapse: collapse;
                margin-bottom: 0;
                background-color: white;
            }}
            
            body.dark-mode table {{
                background-color: #333;
                color: var(--text-light);
            }}
            
            th, td {{
                padding: 12px 15px;
                text-align: left;
                border-bottom: 1px solid #eee;
            }}
            
            body.dark-mode th, body.dark-mode td {{
                border-bottom: 1px solid #555;
            }}
            
            th {{
                background-color: var(--primary);
                color: white;
                position: relative;
                cursor: pointer;
            }}
            
            body.dark-mode th {{
                background-color: #555;
            }}
            
            tr:hover {{
                background-color: rgba(0,0,0,0.02);
            }}
            
            body.dark-mode tr:hover {{
                background-color: rgba(255,255,255,0.05);
            }}
            
            /* Button Styles */
            .button {{
                display: inline-block;
                padding: 8px 15px;
                background-color: var(--primary);
                color: white;
                border: none;
                border-radius: 4px;
                cursor: pointer;
                font-size: 0.9rem;
                transition: background-color 0.3s, transform 0.3s;
                text-decoration: none;
                margin-right: 5px;
                margin-bottom: 5px;
            }}
            
            .button:hover {{
                background-color: #34495e;
                transform: translateY(-1px);
            }}
            
            body.dark-mode .button {{
                background-color: var(--secondary);
            }}
            
            body.dark-mode .button:hover {{
                background-color: #2980b9;
            }}
            
            /* Search Styles */
            .search-container {{
                margin-bottom: 20px;
                position: relative;
            }}
            
            .search-icon {{
                position: absolute;
                left: 10px;
                top: 50%;
                transform: translateY(-50%);
                color: #aaa;
            }}
            
            .search-input {{
                width: 100%;
                padding: 10px 10px 10px 35px;
                border: 1px solid #ddd;
                border-radius: 4px;
                font-size: 1rem;
                transition: border-color 0.3s;
            }}
            
            .search-input:focus {{
                outline: none;
                border-color: var(--primary);
            }}
            
            body.dark-mode .search-input {{
                background-color: #424242;
                border-color: #555;
                color: var(--text-light);
            }}
            
            /* Filter Button Styles */
            .filter-container {{
                display: flex;
                flex-wrap: wrap;
                gap: 10px;
                margin-bottom: 20px;
            }}
            
            .filter-button {{
                padding: 8px 15px;
                background-color: #f1f1f1;
                border: none;
                border-radius: 4px;
                cursor: pointer;
                transition: background-color 0.3s;
            }}
            
            .filter-button.active {{
                background-color: var(--primary);
                color: white;
            }}
            
            body.dark-mode .filter-button {{
                background-color: #555;
                color: #ddd;
            }}
            
            body.dark-mode .filter-button.active {{
                background-color: var(--secondary);
            }}
            
            /* Dashboard Charts and Images */
            .image-container {{
                display: flex;
                flex-wrap: wrap;
                gap: 20px;
                justify-content: center;
            }}
            
            .plot-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
                gap: 20px;
                margin-top: 20px;
            }}
            
            .plot-item {{
                background-color: white;
                border-radius: 8px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.05);
                overflow: hidden;
                transition: transform 0.3s, box-shadow 0.3s;
            }}
            
            body.dark-mode .plot-item {{
                background-color: #333;
                box-shadow: 0 2px 5px rgba(0,0,0,0.2);
            }}
            
            .plot-item:hover {{
                transform: translateY(-5px);
                box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            }}
            
            body.dark-mode .plot-item:hover {{
                box-shadow: 0 5px 15px rgba(0,0,0,0.3);
            }}
            
            .plot-image {{
                width: 100%;
                height: auto;
                max-width: 600px;
                object-fit: contain;
                background-color: #f9f9f9;
                padding: 10px;
                border-bottom: 1px solid #eee;
            }}
            
            body.dark-mode .plot-image {{
                background-color: #444;
                border-bottom: 1px solid #555;
            }}
            
            .plot-name {{
                padding: 15px;
                font-weight: 500;
            }}
            
            /* Modal Styles */
            .modal {{
                display: none;
                position: fixed;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                background-color: rgba(0,0,0,0.5);
                z-index: 1001;
                overflow-y: auto;
            }}
            
            .modal-content {{
                background-color: white;
                margin: 50px auto;
                border-radius: 8px;
                box-shadow: 0 5px 15px rgba(0,0,0,0.3);
                width: 90%;
                max-width: 900px;
                position: relative;
                animation: modalFadeIn 0.3s;
            }}
            
            body.dark-mode .modal-content {{
                background-color: #333;
                color: var(--text-light);
            }}
            
            @keyframes modalFadeIn {{
                from {{ opacity: 0; transform: translateY(-30px); }}
                to {{ opacity: 1; transform: translateY(0); }}
            }}
            
            .modal-header {{
                padding: 15px 20px;
                border-bottom: 1px solid #eee;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }}
            
            body.dark-mode .modal-header {{
                border-bottom: 1px solid #555;
            }}
            
            .modal-body {{
                padding: 20px;
                max-height: 70vh;
                overflow-y: auto;
            }}
            
            .close {{
                color: #aaa;
                font-size: 28px;
                font-weight: bold;
                cursor: pointer;
            }}
            
            .close:hover {{
                color: var(--danger);
            }}
            
            /* JSON Viewer Styles */
            .json-content {{
                background-color: #f9f9f9;
                border-radius: 4px;
                padding: 15px;
                overflow-x: auto;
                font-family: monospace;
                color: #333;
            }}
            
            body.dark-mode .json-content {{
                background-color: #444;
                color: #e0e0e0;
            }}
            
            pre {{
                margin: 0;
                white-space: pre-wrap;
            }}
            
            /* File Grid Styles */
            .file-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
                gap: 20px;
                margin-top: 20px;
            }}
            
            .file-item {{
                background-color: white;
                border-radius: 8px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.05);
                padding: 15px;
                display: flex;
                flex-direction: column;
                align-items: center;
                transition: transform 0.3s, box-shadow 0.3s;
            }}
            
            body.dark-mode .file-item {{
                background-color: #333;
                box-shadow: 0 2px 5px rgba(0,0,0,0.2);
            }}
            
            .file-item:hover {{
                transform: translateY(-3px);
                box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            }}
            
            body.dark-mode .file-item:hover {{
                box-shadow: 0 4px 8px rgba(0,0,0,0.3);
            }}
            
            .file-icon {{
                font-size: 2.5rem;
                margin-bottom: 10px;
                color: var(--primary);
            }}
            
            body.dark-mode .file-icon {{
                color: var(--secondary);
            }}
            
            .file-name {{
                font-weight: 500;
                text-align: center;
                margin-bottom: 10px;
            }}
            
            .file-actions {{
                display: flex;
                gap: 10px;
                margin-top: auto;
            }}
            
            /* Sort Indicators */
            .sort-asc::after {{
                content: " \\25B2";
                font-size: 0.7em;
                vertical-align: middle;
            }}
            
            .sort-desc::after {{
                content: " \\25BC";
                font-size: 0.7em;
                vertical-align: middle;
            }}
            
            /* Responsive Styles */
            @media (max-width: 900px) {{
                .tabs {{
                    overflow-x: auto;
                    justify-content: flex-start;
                }}
                
                .logo-text {{
                    display: none;
                }}
                
                .plot-grid {{
                    grid-template-columns: repeat(auto-fill, minmax(240px, 1fr));
                }}
            }}
            
            @media (max-width: 600px) {{
                #navbar {{
                    padding: 10px;
                }}
                
                .tab {{
                    padding: 8px 10px;
                    font-size: 0.9rem;
                }}
                
                .theme-toggle {{
                    width: 30px;
                    height: 30px;
                }}
                
                .plot-grid {{
                    grid-template-columns: 1fr;
                }}
                
                .file-grid {{
                    grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
                }}
                
                .modal-content {{
                    width: 95%;
                    margin: 30px auto;
                }}
            }}
            
            .action-buttons {{
                display: flex;
                flex-wrap: wrap;
                gap: 10px;
                margin-top: 15px;
            }}
            
            /* Add styles for broken images */
            img.broken-image {{
                position: relative;
                min-height: 150px;
                background-color: #f1f1f1;
                display: flex;
                align-items: center;
                justify-content: center;
                font-style: italic;
                color: #888;
                border: 1px dashed #ccc;
            }}
            
            body.dark-mode img.broken-image {{
                background-color: #444;
                border: 1px dashed #666;
                color: #ccc;
            }}
            
            img.broken-image::before {{
                content: "Image failed to load";
                position: absolute;
            }}

            /* Investment Report iframe Fix - Enhanced */
            iframe {{
                width: 98%; /* Use percentage for responsiveness */
                height: 700px; /* Adjust height as needed */
                border: 1px solid #ddd;
                background-color: white; /* Ensure white background in light mode */
                display: block; /* Prevent scrollbar issues */
                margin: 20px auto; /* Center the iframe */
            }}

            body.dark-mode iframe {{
                background-color: #2d2d2d; /* Dark background for dark mode */
                border-color: #444;
            }}

            /* Expert Analysis Report Preview Iframe Fix - Enhanced Dark Mode */
            #report-preview iframe {{
                width: 100%;
                height: 600px;
                border: 1px solid #ddd;
                background-color: white; /* Light mode background */
            }}

            body.dark-mode #report-preview iframe {{
                background-color: #2d2d2d; /* Dark mode background */
                border-color: #444;
                color: #e0e0e0; /* Ensure text is readable in dark mode if report has dark text */
            }}

            /* JSON Viewer Styles - Dark Mode Contrast Fix - Re-apply */
            .json-content {{
                background-color: #f9f9f9;
                border-radius: 4px;
                padding: 15px;
                overflow-x: auto;
                font-family: monospace;
                color: #333;
            }}

            body.dark-mode .json-content {{
                background-color: #2d2d2d; /* Darker background for better dark mode */
                color: #e0e0e0;       /* Lighter text for dark mode */
                border: 1px solid #444; /* Dark border for dark mode */
            }}

            /* Chart Sizing Fix - Re-apply and adjust */
            .plot-image {{
                width: 100%;
                height: auto;
                max-width: 95%; /* Adjust max-width to be responsive within container */
                object-fit: contain;
                background-color: #f9f9f9;
                padding: 15px;
                margin: 10px auto;
                display: block;
                border-bottom: 1px solid #eee;
            }}

            body.dark-mode .plot-image {{
                background-color: #333;
                border-bottom: 1px solid #555;
            }}

            /* Plot Grid - Adjust for responsiveness */
            .plot-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fill, minmax(min(350px, 100%), 1fr)); /* Responsive grid columns */
                gap: 20px;
                margin-top: 20px;
            }}

            @media (max-width: 600px) {{
                .plot-grid {{
                    grid-template-columns: 1fr; /* Stack plots on smaller screens */
                }}
            }}

            /* Grid layout fix for charts - 2-3 per row */
            .plot-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fill, minmax(400px, 1fr));
                gap: 25px;
                margin-top: 20px;
                padding: 0 15px;
            }}
            
            .plot-item {{
                background-color: white;
                border-radius: 8px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.05);
                overflow: hidden;
                transition: transform 0.3s, box-shadow 0.3s;
                display: flex;
                flex-direction: column;
                height: 100%;
            }}
            
            body.dark-mode .plot-item {{
                background-color: #333;
                box-shadow: 0 2px 5px rgba(0,0,0,0.2);
            }}
            
            .plot-image {{
                width: 100%;
                height: auto;
                max-height: 350px;
                object-fit: contain;
                background-color: #f9f9f9;
                padding: 15px;
                margin: 0 auto;
                display: block;
            }}
            
            body.dark-mode .plot-image {{
                background-color: #333;
                border-bottom: 1px solid #555;
            }}
            
            /* Expert Analysis Preview - Fix dark mode readability */
            #report-preview {{
                background-color: white;
                padding: 20px;
                border-radius: 8px;
                margin-top: 20px;
            }}
            
            body.dark-mode #report-preview {{
                background-color: #e0e0e0;
                color: #333;
            }}
            
            /* Investment Report Integration */
            #investment-report-container {{
                width: 100%;
                height: 800px;
                border: 1px solid #ddd;
                background: white;
                margin: 20px 0;
                overflow: hidden;
            }}
            
            body.dark-mode #investment-report-container {{
                border-color: #444;
            }}
            
            /* JSON Viewer error handling */
            .error-message {{
                background-color: #ffecec;
                color: #e74c3c;
                padding: 15px;
                border-radius: 4px;
                border-left: 4px solid #e74c3c;
                margin-bottom: 20px;
                font-size: 0.9rem;
            }}
            
            body.dark-mode .error-message {{
                background-color: #4a2b2b;
                color: #ff8a80;
                border-left: 4px solid #ff5252;
            }}

            /* Add CSS for error messages */
            .error-message {{
                background-color: #ffecec;
                color: #e74c3c;
                padding: 15px;
                border-radius: 4px;
                border-left: 4px solid #e74c3c;
                margin-bottom: 20px;
                font-size: 0.9rem;
            }}
            
            body.dark-mode .error-message {{
                background-color: #4a2b2b;
                color: #ff8a80;
                border-left: 4px solid #ff5252;
            }}
            
            .error-message details {{
                margin-top: 10px;
            }}
            
            .error-message summary {{
                cursor: pointer;
                margin-bottom: 5px;
                font-weight: 500;
            }}
            
            .error-message pre {{
                background-color: #f8f8f8;
                padding: 10px;
                border-radius: 4px;
                overflow-x: auto;
                margin-top: 10px;
                color: #333;
            }}
            
            body.dark-mode .error-message pre {{
                background-color: #333;
                color: #ddd;
            }}
        </style>
    </head>
    <body>
        <nav id="navbar">
            <a href="#" class="logo">
                <i class="fas fa-chart-line"></i>
                <span class="logo-text">Investment Dashboard</span>
            </a>
            
            <div class="tabs">
                <div class="tab active" onclick="openTab(event, 'overview')">
                    <i class="fas fa-home"></i> Overview
                </div>
                <div class="tab" onclick="openTab(event, 'investment')">
                    <i class="fas fa-chart-pie"></i> Investment Reports
                </div>
                <div class="tab" onclick="openTab(event, 'expert')">
                    <i class="fas fa-user-tie"></i> Expert Analysis
                </div>
                <div class="tab" onclick="openTab(event, 'validation')">
                    <i class="fas fa-check-circle"></i> Model Validation
                </div>
                <div class="tab" onclick="openTab(event, 'processed-data')">
                    <i class="fas fa-database"></i> Processed Data
                </div>
                <div class="tab" onclick="openTab(event, 'raw-data')">
                    <i class="fas fa-file-csv"></i> Raw Data
                </div>
                <div class="tab" onclick="openTab(event, 'charts')">
                    <i class="fas fa-chart-bar"></i> Charts
                </div>
            </div>
            
            <button class="theme-toggle" onclick="toggleDarkMode()">
                <i class="fas fa-moon"></i>
            </button>
        </nav>

        <!-- Overview section -->
        <div id="overview" class="tab-content">
            <div class="dashboard-header">
                <h1><i class="fas fa-tachometer-alt"></i> Investment System Dashboard</h1>
                <p>Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="stats-container">
                <div class="card">
                    <h3><i class="fas fa-info-circle"></i> System Status Summary</h3>
                    <ul>
                        <li><strong>Total Reports:</strong> {len(json_files)} JSON reports</li>
                        <li><strong>Total Visualizations:</strong> {len(image_files)} charts and plots</li>
                        <li><strong>Expert Reports:</strong> {len(expert_reports)} HTML reports</li>
                        <li><strong>Last Updated:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</li>
                    </ul>
                </div>
    """
    
    # Add stock training summary from training_summary.json if it exists
    training_summary_file = os.path.join(REPORTS_DIRS["training"], "training_summary.json")
    if os.path.exists(training_summary_file):
        training_data = load_json_file(training_summary_file)
        if training_data:
            stocks_total = training_data.get('stocks', {}).get('total', 0)
            stocks_success = training_data.get('stocks', {}).get('success', 0)
            stocks_failed = training_data.get('stocks', {}).get('failed', 0)
            
            # Add success rate stats
            html_content += f"""
                    <div class="card">
                        <h3>Model Training Summary</h3>
                        <div class="stock-metrics">
                            <div class="metric-card">
                                <div class="metric-label">Total Stocks</div>
                                <div class="metric-value">{stocks_total}</div>
                            </div>
                            <div class="metric-card">
                                <div class="metric-label">Successfully Trained</div>
                                <div class="metric-value success">{stocks_success}</div>
                            </div>
                            <div class="metric-card">
                                <div class="metric-label">Failed</div>
                                <div class="metric-value failure">{stocks_failed}</div>
                            </div>
                            <div class="metric-card">
                                <div class="metric-label">Success Rate</div>
                                <div class="metric-value">{(stocks_success / max(stocks_total, 1)) * 100:.1f}%</div>
                            </div>
                        </div>
                    </div>
            """
    
    # Add model validation summary if it exists
    validation_summary_file = os.path.join(RESULTS_DIR, "validation_summary.json")
    if os.path.exists(validation_summary_file):
        validation_data = load_json_file(validation_summary_file)
        if validation_data:
            # Count how many stocks have validation data
            validated_count = len(validation_data)
            
            # Calculate average metrics across all models
            avg_metrics = {
                'rmse': 0,
                'mae': 0,
                'r2': 0,
                'mape': 0
            }
            
            for ticker, data in validation_data.items():
                metrics = data.get('metrics', {})
                for metric in avg_metrics:
                    avg_metrics[metric] += metrics.get(metric, 0)
            
            # Calculate averages
            if validated_count > 0:
                for metric in avg_metrics:
                    avg_metrics[metric] /= validated_count
                
                html_content += f"""
                    <div class="card">
                        <h3>Model Validation Summary</h3>
                        <p>Performance metrics averaged across {validated_count} validated models:</p>
                        <div class="stock-metrics">
                            <div class="metric-card">
                                <div class="metric-label">Avg RMSE</div>
                                <div class="metric-value">{avg_metrics['rmse']:.4f}</div>
                            </div>
                            <div class="metric-card">
                                <div class="metric-label">Avg MAE</div>
                                <div class="metric-value">{avg_metrics['mae']:.4f}</div>
                            </div>
                            <div class="metric-card">
                                <div class="metric-label">Avg R²</div>
                                <div class="metric-value">{avg_metrics['r2']:.4f}</div>
                            </div>
                            <div class="metric-card">
                                <div class="metric-label">Avg MAPE</div>
                                <div class="metric-value">{avg_metrics['mape']:.2f}%</div>
                            </div>
                        </div>
                    </div>
                """
    
    # Add a preview of some key plots
    key_plots = [f for f in image_files if "success_rate" in f.lower() or "prediction" in f.lower()][:3]
    if key_plots:
        html_content += """
                    <div class="card">
                        <h3>Key Visualizations</h3>
                        <div class="image-container">
        """
        
        for plot_file in key_plots:
            plot_name = os.path.basename(plot_file)
            img_data = file_to_base64(plot_file)
            if img_data:
                html_content += f"""
                            <div style="margin-bottom: 20px;">
                                <h4>{plot_name}</h4>
                                <img src="{img_data}" alt="{plot_name}" style="max-height: 300px;">
                            </div>
                """
        
        html_content += """
                        </div>
                    </div>
        """
    
    # Add action buttons for generating reports
    html_content += """
                    <div class="action-buttons">
                        <h3>Actions</h3>
    """
    
    if has_report_generator:
        html_content += """
                        <button class="button" onclick="runScript('generate_investment_report.py')">
                            <i class="fas fa-sync-alt"></i> Generate Investment Report
                        </button>
        """
    
    html_content += """
                    </div>
                </div>
            </div>
            
            <div id="investment" class="tab-content">
                <div class="section">
                    <h2>Investment Report</h2>
    """
    
    # Embed existing investment report if it exists
    if has_investment_report:
        html_content += f"""
                    <div class="card">
                        <h3>Investment Report</h3>
                        <p>This comprehensive report provides investment analysis, recommendations, and market insights.</p>
                        <iframe src="file://{os.path.abspath(investment_report_path)}" title="Investment Report"></iframe>
                        <p><a href="file://{os.path.abspath(investment_report_path)}" target="_blank">Open in new window</a></p>
                    </div>
        """
    else:
        html_content += """
                    <div class="card">
                        <h3>No Investment Report Found</h3>
                        <p>No investment report has been generated yet.</p>
        """
        
        if has_report_generator:
            html_content += """
                        <p>Use the button below to generate a new investment report.</p>
                        <div class="action-buttons">
                            <button class="button" onclick="runScript('generate_investment_report.py')">
                                <i class="fas fa-sync-alt"></i> Generate Investment Report
                            </button>
                        </div>
            """
        
        html_content += """
                    </div>
        """
    
    html_content += """
                </div>
            </div>
            
            <div id="expert" class="tab-content">
                <div class="section">
                    <h2>Expert Analysis & Predictions</h2>
    """
    
    if expert_reports:
        html_content += f"""
                    <div class="card">
                        <h3><i class="fas fa-user-tie"></i> Found {len(expert_reports)} Expert Reports</h3>
                        <div class="search-container">
                            <i class="fas fa-search search-icon"></i>
                            <input type="text" id="expertSearchInput" class="search-input" 
                                   onkeyup="filterTable('expertSearchInput', 'expertTable')" 
                                   placeholder="Search for reports...">
                        </div>
                        <div class="report-list">
                            <table class="data-table" id="expertTable">
                                <tr>
                                    <th onclick="sortTable('expertTable', 0)">Report Name <i class="fas fa-sort"></i></th>
                                    <th onclick="sortTable('expertTable', 1)">Location <i class="fas fa-sort"></i></th>
                                    <th onclick="sortTable('expertTable', 2)">Size <i class="fas fa-sort"></i></th>
                                    <th onclick="sortTable('expertTable', 3)">Last Modified <i class="fas fa-sort"></i></th>
                                    <th>Action</th>
                                </tr>
                    """
        
        for report_path in expert_reports:
            report_name = os.path.basename(report_path)
            try:
                file_size = os.path.getsize(report_path)
                size_str = f"{file_size/1024:.1f} KB" if file_size < 1024*1024 else f"{file_size/(1024*1024):.1f} MB"
                mod_time = datetime.fromtimestamp(os.path.getmtime(report_path)).strftime("%Y-%m-%d %H:%M")
            except:
                size_str = "N/A"
                mod_time = "N/A"
            
            html_content += f"""
                                <tr>
                                    <td>{report_name}</td>
                                    <td>{os.path.dirname(report_path)}</td>
                                    <td>{size_str}</td>
                                    <td>{mod_time}</td>
                                    <td>
                                        <a href="file://{os.path.abspath(report_path)}" class="button" target="_blank">Open</a>
                                        <button class="button" onclick="previewReport('{os.path.abspath(report_path)}')">Preview</button>
                                    </td>
                                </tr>
                    """
        
        html_content += """
                            </table>
                        </div>
                        <div id="report-preview" style="margin-top: 20px; border: 1px solid #ddd; padding: 10px;">
                            <h4>Report Preview</h4>
                            <iframe id="preview-frame" style="width: 100%; height: 600px; border: none;"></iframe>
                        </div>
                    </div>
    """
    else:
        html_content += """
                    <div class="card">
                        <h3>No Expert Reports Found</h3>
                        <p>We searched in these locations:</p>
                        <ul>
                            <li>./results/</li>
                            <li>./results/reports/</li>
                            <li>./data/uploads/</li>
                            <li>./data/stocks/</li>
                            <li>./data/mutual_funds/</li>
                        </ul>
                        <p>To include expert reports:</p>
                        <ol>
                            <li>Place HTML reports in any of the above directories</li>
                            <li>Ensure files are larger than 1KB</li>
                            <li>Files should have .html or .htm extension</li>
                        </ol>
                    </div>
    """
    
    html_content += """
                </div>
            </div>
            
            <div id="validation" class="tab-content">
                <div class="section">
                    <h2>Model Validation & Training Parameters</h2>
                    <p>This section provides details about how models were trained and validated.</p>
                    
                    <div class="card">
                        <h3>Model Training Parameters</h3>
                        
                        <h4>Timeframe Configurations</h4>
                        <table class="timeframe-table">
                            <tr>
                                <th>Timeframe</th>
                                <th>Sequence Length</th>
                                <th>Prediction Horizons (days)</th>
                            </tr>
                            <tr>
                                <td>Short-term</td>
                                <td>30 days</td>
                                <td>1, 3, 5</td>
                            </tr>
                            <tr>
                                <td>Medium-term</td>
                                <td>60 days</td>
                                <td>7, 14, 21</td>
                            </tr>
                            <tr>
                                <td>Long-term</td>
                                <td>90 days</td>
                                <td>30, 60, 90</td>
                            </tr>
                        </table>
                        
                        <h4>LSTM Model Architectures</h4>
                        <table class="data-table">
                            <tr>
                                <th>Architecture</th>
                                <th>Description</th>
                                <th>Parameters</th>
                            </tr>
                            <tr>
                                <td>Simple</td>
                                <td>Single LSTM layer with dropout</td>
                                <td>LSTM(64) → Dropout(0.2) → Dense</td>
                            </tr>
                            <tr>
                                <td>Medium</td>
                                <td>Two LSTM layers with dropout</td>
                                <td>LSTM(64, return_sequences=True) → Dropout(0.2) → LSTM(64) → Dropout(0.2) → Dense</td>
                            </tr>
                            <tr>
                                <td>Complex</td>
                                <td>Three LSTM layers with batch normalization</td>
                                <td>LSTM(128, return_sequences=True) → BatchNorm → Dropout(0.2) → LSTM(64, return_sequences=True) → BatchNorm → Dropout(0.2) → LSTM(32) → BatchNorm → Dropout(0.2) → Dense</td>
                            </tr>
                        </table>
                        
                        <h4>Training Procedure</h4>
                        <ul>
                            <li><strong>Optimizer:</strong> Adam</li>
                            <li><strong>Loss Function:</strong> Mean Squared Error (MSE)</li>
                            <li><strong>Early Stopping:</strong> Patience of 10 epochs, monitoring validation loss</li>
                            <li><strong>Validation Split:</strong> 20% of data</li>
                            <li><strong>Input Features:</strong> Price, Open, High, Low, Volume, Change%, Technical Indicators (when available)</li>
                            <li><strong>Data Normalization:</strong> MinMaxScaler (0-1 range)</li>
                        </ul>
                    </div>
                    
                    <div class="card">
                        <h3>Validation Methodology</h3>
                        <p>Models are validated using a train-test split approach:</p>
                        <ul>
                            <li><strong>Training Period:</strong> 8 months of historical data</li>
                            <li><strong>Testing Period:</strong> 2 months of out-of-sample data</li>
                            <li><strong>Validation Metrics:</strong> RMSE, MAE, R², MAPE</li>
                        </ul>
                        
                        <p>The validation process follows these steps:</p>
                        <ol>
                            <li>Split data into training (80%) and testing (20%) periods</li>
                            <li>Train models on the training data</li>
                            <li>Evaluate model performance on test data</li>
                            <li>After validation, retrain on the full dataset for production use</li>
                            <li>Models are tested against different prediction horizons to determine optimal forecasting periods</li>
                        </ol>
                    </div>
    """
    
    # Add validation results if available
    if os.path.exists(validation_summary_file):
        validation_data = load_json_file(validation_summary_file)
        if validation_data and len(validation_data) > 0:
            html_content += """
                    <div class="card">
                        <h3><i class="fas fa-check-double"></i> Validation Results by Stock</h3>
                        <div class="search-container">
                            <i class="fas fa-search search-icon"></i>
                            <input type="text" id="validationSearchInput" class="search-input" 
                                   onkeyup="filterTable('validationSearchInput', 'validationTable')" 
                                   placeholder="Search for stocks...">
                        </div>
                        <table class="data-table" id="validationTable">
                            <tr>
                                <th onclick="sortTable('validationTable', 0)">Stock <i class="fas fa-sort"></i></th>
                                <th onclick="sortTable('validationTable', 1)">RMSE <i class="fas fa-sort"></i></th>
                                <th onclick="sortTable('validationTable', 2)">MAE <i class="fas fa-sort"></i></th>
                                <th onclick="sortTable('validationTable', 3)">R² <i class="fas fa-sort"></i></th>
                                <th onclick="sortTable('validationTable', 4)">MAPE <i class="fas fa-sort"></i></th>
                                <th onclick="sortTable('validationTable', 5)">Best Horizon <i class="fas fa-sort"></i></th>
                            </tr>
                        """
            
            for ticker, data in validation_data.items():
                metrics = data.get('metrics', {})
                best_horizon = data.get('best_horizon', "N/A")
                
                html_content += f"""
                            <tr>
                                <td>{ticker}</td>
                                <td>{metrics.get('rmse', 'N/A'):.4f if isinstance(metrics.get('rmse'), (int, float)) else 'N/A'}</td>
                                <td>{metrics.get('mae', 'N/A'):.4f if isinstance(metrics.get('mae'), (int, float)) else 'N/A'}</td>
                                <td>{metrics.get('r2', 'N/A'):.4f if isinstance(metrics.get('r2'), (int, float)) else 'N/A'}</td>
                                <td>{metrics.get('mape', 'N/A'):.2f}% if isinstance(metrics.get('mape'), (int, float)) else 'N/A'</td>
                                <td>{best_horizon} days</td>
                            </tr>
                """
            
            html_content += """
                        </table>
                    </div>
            """
    
    # Add validation plots
    validation_plots = [f for f in image_files if "prediction" in f.lower() or "error" in f.lower()]
    if validation_plots:
        html_content += """
                    <div class="card">
                        <h3>Validation Plots</h3>
                        <div class="plot-grid">
        """
        
        for plot_file in validation_plots[:6]:  # Limit to 6 plots
            plot_name = os.path.basename(plot_file)
            img_data = file_to_base64(plot_file)
            if img_data:
                html_content += f"""
                            <div class="plot-item">
                                <h4>{plot_name}</h4>
                                <img src="{img_data}" alt="{plot_name}" style="width: 100%;">
                            </div>
                """
        
        html_content += """
                        </div>
                    </div>
        """
    
    html_content += """
                </div>
            </div>
            
            <!-- Create processed data content variable -->
            <div id="processed-data" class="tab-content">
                <div class="section">
                    <h2><i class="fas fa-database"></i> Processed & Cleaned Data</h2>
                    <div class="section-content">
                        <div class="search-container">
                            <i class="fas fa-search search-icon"></i>
                            <input type="text" id="processedDataSearch" class="search-input" 
                                   placeholder="Search processed data files..." 
                                   onkeyup="searchProcessedData()">
                        </div>
                        
                        <div class="filter-container">
                            <button class="filter-button active" data-type="all" onclick="filterProcessedDataByType('all')">All Files</button>
                            <button class="filter-button" data-type="validation" onclick="filterProcessedDataByType('validation')">Validation Data</button>
                            <button class="filter-button" data-type="training" onclick="filterProcessedDataByType('training')">Training Data</button>
                            <button class="filter-button" data-type="processed" onclick="filterProcessedDataByType('processed')">Processed Data</button>
                        </div>
                        
                        <div id="no-processed-results" style="display:none;" class="error-message">
                            <i class="fas fa-info-circle"></i> No processed data files match your search.
                        </div>
                        
                        <div class="file-grid" id="processed-data-grid">
<!-- Processed Data Items Will Be Generated Here -->
                        </div>
                        
<!-- Processed Data Warning Will Be Displayed If Necessary -->
                    </div>
                </div>
            </div>
            
            <div id="raw-data" class="tab-content">
                <div class="section">
                    <h2><i class="fas fa-file-alt"></i> Raw Data Files</h2>
                    <div class="section-content">
                        <div class="search-container">
                            <i class="fas fa-search search-icon"></i>
                            <input type="text" id="rawDataSearch" class="search-input" 
                                   placeholder="Search raw data files..." 
                                   onkeyup="searchRawData()">
                        </div>
                        
                        <div class="filter-container">
                            <button class="filter-button active" data-type="all" onclick="filterRawDataByType('all')">All Files</button>
                            <button class="filter-button" data-type="stocks" onclick="filterRawDataByType('stocks')">Stocks</button>
                            <button class="filter-button" data-type="mutual_funds" onclick="filterRawDataByType('mutual_funds')">Mutual Funds</button>
                            <button class="filter-button" data-type="uploads" onclick="filterRawDataByType('uploads')">Uploads</button>
                            <button class="filter-button" data-type="csv" onclick="filterRawDataByType('csv')">CSV Files</button>
                            <button class="filter-button" data-type="excel" onclick="filterRawDataByType('excel')">Excel Files</button>
                        </div>
                        
                        <div id="no-raw-results" style="display:none;" class="error-message">
                            <i class="fas fa-info-circle"></i> No raw data files match your search.
                        </div>
    """
    
    if raw_data_files:
        html_content += """
                        <div class="table-container">
                            <table id="rawDataTable">
                                <thead>
                                    <tr>
                                        <th onclick="sortTable('rawDataTable', 0)" class="sortable">File Name</th>
                                        <th onclick="sortTable('rawDataTable', 1)" class="sortable">Location</th>
                                        <th onclick="sortTable('rawDataTable', 2)" class="sortable">Type</th>
                                        <th onclick="sortTable('rawDataTable', 3)" class="sortable">Size</th>
                                        <th>Actions</th>
                                    </tr>
                                </thead>
                                <tbody>
        """
        
        for file_path in raw_data_files:
            file_name = os.path.basename(file_path)
            location = os.path.dirname(file_path)
            file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
            file_size_str = f"{file_size / 1024:.1f} KB" if file_size > 0 else "N/A"
            file_ext = os.path.splitext(file_name)[1][1:].upper()
            
            # Determine file type for filtering
            data_type = "other"
            if "stocks" in location.lower() or "stock" in file_name.lower():
                data_type = "stocks"
            elif "mutual_funds" in location.lower() or "fund" in file_name.lower():
                data_type = "mutual_funds"
            elif "uploads" in location.lower():
                data_type = "uploads"
                
            # Determine file format type
            format_type = "other"
            if file_ext.lower() in ["csv"]:
                format_type = "csv"
            elif file_ext.lower() in ["xlsx", "xls"]:
                format_type = "excel"
                
            html_content += f"""
                                <tr data-type="{data_type}" data-format="{format_type}">
                                    <td>{file_name}</td>
                                    <td>{location}</td>
                                    <td>{file_ext}</td>
                                    <td>{file_size_str}</td>
                                    <td>
                                        <a href="{file_path}" class="button" download>
                                            <i class="fas fa-download"></i> Download
                                        </a>
                                    </td>
                                </tr>
            """
        
        html_content += """
                                </tbody>
                            </table>
                        </div>
        """
    else:
        html_content += """
                        <div class="card">
                            <h3><i class="fas fa-exclamation-triangle"></i> No Raw Data Files Found</h3>
                            <p>No raw data files (.csv, .xlsx, .xls) were found in the following locations:</p>
                            <ul>
                                <li><code>./data/</code></li>
                                <li><code>./data/uploads/</code></li>
                                <li><code>./data/stocks/</code></li>
                                <li><code>./data/mutual_funds/</code></li>
                                <li><code>./data/processed/</code></li>
                                <li><code>./data/raw/</code></li>
                            </ul>
                            <p>To include raw data files:</p>
                            <ol>
                                <li>Upload stock and mutual fund CSV/Excel files to the appropriate directories</li>
                                <li>Ensure that the data files have the correct file extension (.csv, .xlsx)</li>
                                <li>Check that you have proper read permissions for the data directory</li>
                            </ol>
                        </div>
        """
    
    html_content += """
                    </div>
                </div>
            </div>
            
            <div id="charts" class="tab-content">
                <div class="section">
                    <h2><i class="fas fa-chart-bar"></i> Charts & Plots</h2>
                    <p>This section displays all visualization outputs from the system.</p>
                    
                    <div class="search-container">
                        <input type="text" id="plotSearchInput" onkeyup="filterPlots()" placeholder="Search for charts and plots...">
                    </div>
                    
                    <div class="filter-container">
                        <button class="filter-button active" onclick="filterPlotsByType('all')">All Plots</button>
                        <button class="filter-button" onclick="filterPlotsByType('prediction')">Prediction</button>
                        <button class="filter-button" onclick="filterPlotsByType('training')">Training</button>
                        <button class="filter-button" onclick="filterPlotsByType('risk')">Risk Analysis</button>
                    </div>
                    
                    <div class="plot-grid" id="plotGrid">
    """
    
    # Add ALL plots to this section
    if image_files:
        for plot_file in image_files:
            plot_name = os.path.basename(plot_file)
            img_data = file_to_base64(plot_file)
            if img_data:
                # Add plot type classes based on filename
                plot_classes = ""
                if "risk" in plot_name.lower():
                    plot_classes += " risk-plot"
                if "return" in plot_name.lower():
                    plot_classes += " return-plot"
                if "prediction" in plot_name.lower():
                    plot_classes += " prediction-plot"
                if "training" in plot_name.lower():
                    plot_classes += " training-plot"
                
                html_content += f"""
                        <div class="plot-item{plot_classes}" data-name="{plot_name.lower()}">
                            <h4>{plot_name}</h4>
                            <img src="{img_data}" alt="{plot_name}" onerror="handleImageError(this)" onclick="openImageModal('{img_data}', '{plot_name}')">
                        </div>
                """
    else:
        html_content += """
                        <div class="empty-state">
                            <i class="fas fa-chart-bar fa-3x"></i>
                            <h3>No Charts Available</h3>
                            <p>No charts or plots have been generated yet. Run the model training and validation scripts to generate visualizations.</p>
                        </div>
        """
    
    html_content += """
                    </div>
                </div>
            </div>
        </div>
        
        <!-- JSON Viewer Modal -->
        <div id="json-viewer" class="modal">
            <div class="modal-content">
                <div class="modal-header">
                    <h3 id="json-viewer-title">JSON Viewer</h3>
                    <span class="close" onclick="closeJsonViewer()">&times;</span>
                </div>
                <div id="json-content-container" class="modal-body"></div>
            </div>
        </div>
        
        <!-- JavaScript for the dashboard -->
        <script>
            // Function to open tab content
            function openTab(evt, tabName) {
                var i, tabcontent, tablinks;
                
                // Hide all tab content
                tabcontent = document.getElementsByClassName("tab-content");
                for (i = 0; i < tabcontent.length; i++) {
                    tabcontent[i].classList.remove("active");
                }
                
                // Remove active class from all tabs
                tablinks = document.getElementsByClassName("tab");
                for (i = 0; i < tablinks.length; i++) {
                    tablinks[i].classList.remove("active");
                }
                
                // Show the current tab and add active class
                document.getElementById(tabName).classList.add("active");
                evt.currentTarget.classList.add("active");
                
                // Save the active tab to localStorage
                localStorage.setItem('activeTab', tabName);
            }
            
            // Toggle dark mode
            function toggleDarkMode() {
                document.body.classList.toggle('dark-mode');
                const isDarkMode = document.body.classList.contains('dark-mode');
                localStorage.setItem('darkMode', isDarkMode ? 'enabled' : 'disabled');
                
                // Update icon
                const themeIcon = document.querySelector('.theme-toggle i');
                if (isDarkMode) {
                    themeIcon.classList.remove('fa-moon');
                    themeIcon.classList.add('fa-sun');
                } else {
                    themeIcon.classList.remove('fa-sun');
                    themeIcon.classList.add('fa-moon');
                }
            }
            
            // Handle scrolling for navbar
            window.onscroll = function() {
                var navbar = document.getElementById("navbar");
                if (document.body.scrollTop > 50 || document.documentElement.scrollTop > 50) {
                    navbar.classList.add("scrolled");
                } else {
                    navbar.classList.remove("scrolled");
                }
            };
            
            // Filter table based on search input
            function filterTable(inputId, tableId) {
                const input = document.getElementById(inputId);
                const filter = input.value.toUpperCase();
                const table = document.getElementById(tableId);
                const tr = table.getElementsByTagName('tr');
                
                for (let i = 1; i < tr.length; i++) {
                    let found = false;
                    const td = tr[i].getElementsByTagName('td');
                    
                    for (let j = 0; j < td.length; j++) {
                        if (td[j]) {
                            const txtValue = td[j].textContent || td[j].innerText;
                            if (txtValue.toUpperCase().indexOf(filter) > -1) {
                                found = true;
                                break;
                            }
                        }
                    }
                    
                    tr[i].style.display = found ? '' : 'none';
                }
            }
            
            // Sort table by column
            function sortTable(tableId, colIndex) {
                const table = document.getElementById(tableId);
                let rows, switching, i, x, y, shouldSwitch, dir = 'asc';
                switching = true;
                
                const sorted = table.getAttribute('data-sorted');
                if (sorted && sorted.split(',')[0] == colIndex) {
                    dir = sorted.split(',')[1] === 'asc' ? 'desc' : 'asc';
                }
                
                // Loop until no switching is needed
                while (switching) {
                    switching = false;
                    rows = table.rows;
                    
                    for (i = 1; i < (rows.length - 1); i++) {
                        shouldSwitch = false;
                        x = rows[i].getElementsByTagName('td')[colIndex];
                        y = rows[i + 1].getElementsByTagName('td')[colIndex];
                        
                        // Check if the two rows should switch
                        if (dir === 'asc') {
                            if (x.innerHTML.toLowerCase() > y.innerHTML.toLowerCase()) {
                                shouldSwitch = true;
                                break;
                            }
                        } else if (dir === 'desc') {
                            if (x.innerHTML.toLowerCase() < y.innerHTML.toLowerCase()) {
                                shouldSwitch = true;
                                break;
                            }
                        }
                    }
                    
                    if (shouldSwitch) {
                        rows[i].parentNode.insertBefore(rows[i + 1], rows[i]);
                        switching = true;
                    }
                }
                
                // Update the sorted attribute
                table.setAttribute('data-sorted', colIndex + ',' + dir);
                
                // Update sort indicators
                const headers = table.getElementsByTagName('th');
                for (i = 0; i < headers.length; i++) {
                    headers[i].classList.remove('sort-asc', 'sort-desc');
                }
                
                headers[colIndex].classList.add(dir === 'asc' ? 'sort-asc' : 'sort-desc');
            }
            
            // Filter data by type
            function filterByType(type) {
                const active = document.querySelector('.tab-content.active');
                const tableId = active.querySelector('table').id;
                const table = document.getElementById(tableId);
                const tr = table.getElementsByTagName('tr');
                const buttons = document.querySelectorAll('.filter-button');
                
                // Update active button
                buttons.forEach(button => {
                    button.classList.remove('active');
                    if (button.innerText.toLowerCase().includes(type.toLowerCase())) {
                        button.classList.add('active');
                    }
                });
                
                if (type === 'all') {
                    // Show all rows
                    for (let i = 1; i < tr.length; i++) {
                        tr[i].style.display = '';
                    }
                } else {
                    // Show only rows matching the type
                    for (let i = 1; i < tr.length; i++) {
                        const locationCol = tr[i].getElementsByTagName('td')[1]; // Location column
                        if (locationCol) {
                            const locationText = locationCol.textContent || locationCol.innerText;
                            tr[i].style.display = locationText.toLowerCase().includes(type.toLowerCase()) ? '' : 'none';
                        }
                    }
                }
            }
            
            // Function to view JSON file
            function viewJsonFile(filePath) {
                // Create an absolute path URL that works with fetch
                const fileUrl = 'file://' + filePath;
                
                // Use XMLHttpRequest which can handle file:// protocol
                const xhr = new XMLHttpRequest();
                xhr.onreadystatechange = function() {
                    if (xhr.readyState === 4) {
                        if (xhr.status === 200 || xhr.status === 0) {
                            try {
                                // Before parsing, check if content is empty
                                if (!xhr.responseText || xhr.responseText.trim() === '') {
                                    throw new Error('JSON file is empty');
                                }
                                
                                const data = JSON.parse(xhr.responseText);
                                const jsonViewer = document.getElementById('json-viewer');
                                jsonViewer.style.display = 'block';
                                
                                // Create the JSON viewer content
                                const jsonContent = document.createElement('div');
                                jsonContent.className = 'json-content';
                                jsonContent.innerHTML = `<pre>${JSON.stringify(data, null, 2)}</pre>`;
                                
                                // Clear previous content and add new
                                const viewerContent = document.getElementById('json-content-container');
                                viewerContent.innerHTML = '';
                                viewerContent.appendChild(jsonContent);
                                
                                // Add filename as title
                                const filename = filePath.split('/').pop();
                                document.getElementById('json-viewer-title').innerText = filename;
                            } catch (error) {
                                console.error('Error parsing JSON:', error, 'File:', filePath);
                                
                                // Create a detailed error message
                                const errorContent = document.createElement('div');
                                errorContent.className = 'error-message';
                                errorContent.innerHTML = `
                                    <strong>Failed to parse JSON file:</strong> ${error.message}<br>
                                    <p>The file may be corrupted or incomplete. Please check the file content and structure.</p>
                                    <p><strong>File path:</strong> ${filePath}</p>
                                    ${xhr.responseText ? 
                                        `<details>
                                            <summary>File content preview (first 200 chars)</summary>
                                            <pre>${xhr.responseText.substring(0, 200)}${xhr.responseText.length > 200 ? '...' : ''}</pre>
                                        </details>` : 
                                        '<p>The file appears to be empty.</p>'}
                                `;
                                
                                // Show the error in the viewer
                                const jsonViewer = document.getElementById('json-viewer');
                                jsonViewer.style.display = 'block';
                                
                                // Add filename as title
                                const filename = filePath.split('/').pop();
                                document.getElementById('json-viewer-title').innerText = `Error: ${filename}`;
                                
                                // Clear previous content and add error message
                                const viewerContent = document.getElementById('json-content-container');
                                viewerContent.innerHTML = '';
                                viewerContent.appendChild(errorContent);
                            }
                        } else {
                            console.error('Error fetching JSON file:', xhr.statusText);
                            
                            // Create a detailed error message for file access issues
                            const errorContent = document.createElement('div');
                            errorContent.className = 'error-message';
                            errorContent.innerHTML = `
                                <strong>Failed to load JSON file</strong><br>
                                <p>Status: ${xhr.status} ${xhr.statusText}</p>
                                <p>The file may be missing, moved, or you might not have permission to access it.</p>
                                <p><strong>File path:</strong> ${filePath}</p>
                                <p>Try regenerating the reports to create the missing files.</p>
                            `;
                            
                            // Show the error in the viewer
                            const jsonViewer = document.getElementById('json-viewer');
                            jsonViewer.style.display = 'block';
                            
                            // Add error as title
                            document.getElementById('json-viewer-title').innerText = 'File Access Error';
                            
                            // Clear previous content and add error message
                            const viewerContent = document.getElementById('json-content-container');
                            viewerContent.innerHTML = '';
                            viewerContent.appendChild(errorContent);
                        }
                    }
                };
                
                try {
                    xhr.open('GET', fileUrl, true);
                    xhr.send();
                } catch (error) {
                    console.error('Error opening request:', error);
                    
                    // Create a detailed error message
                    const errorContent = document.createElement('div');
                    errorContent.className = 'error-message';
                    errorContent.innerHTML = `
                        <strong>Failed to access file:</strong> ${error.message}<br>
                        <p>The browser may be blocking access to local files. If you're using Chrome, try Firefox or Safari instead.</p>
                        <p><strong>File path:</strong> ${filePath}</p>
                    `;
                    
                    // Show the error in the viewer
                    const jsonViewer = document.getElementById('json-viewer');
                    jsonViewer.style.display = 'block';
                    
                    // Add error as title
                    document.getElementById('json-viewer-title').innerText = 'Browser Security Error';
                    
                    // Clear previous content and add error message
                    const viewerContent = document.getElementById('json-content-container');
                    viewerContent.innerHTML = '';
                    viewerContent.appendChild(errorContent);
                }
            }
            
            // Function to close JSON viewer
            function closeJsonViewer() {
                document.getElementById('json-viewer').style.display = 'none';
            }
            
            // Function to filter raw data files by type
            function filterByType(type) {
                const rows = document.querySelectorAll('#rawDataTable tbody tr');
                document.querySelectorAll('.filter-button').forEach(btn => {
                    btn.classList.remove('active');
                });
                document.querySelector(`.filter-button[data-type="${type}"]`).classList.add('active');
                
                rows.forEach(row => {
                    const fileType = row.getAttribute('data-type');
                    if (type === 'all' || fileType === type) {
                        row.style.display = '';
                    } else {
                        row.style.display = 'none';
                    }
                });
            }
            
            // Function to run scripts
            function runScript(scriptName) {
                if (confirm('Run ' + scriptName + '? This may take a few minutes.')) {
                    // Create a simple status dialog
                    const statusDiv = document.createElement('div');
                    statusDiv.style.position = 'fixed';
                    statusDiv.style.top = '50%';
                    statusDiv.style.left = '50%';
                    statusDiv.style.transform = 'translate(-50%, -50%)';
                    statusDiv.style.padding = '20px';
                    statusDiv.style.background = 'rgba(0,0,0,0.8)';
                    statusDiv.style.color = 'white';
                    statusDiv.style.borderRadius = '10px';
                    statusDiv.style.zIndex = '9999';
                    statusDiv.innerHTML = `<p>Running ${scriptName}...</p><p>This may take several minutes.</p>`;
                    document.body.appendChild(statusDiv);
                    
                    // Create a form to submit
                    const form = document.createElement('form');
                    form.method = 'post';
                    form.action = 'run_script.php';
                    form.style.display = 'none';
                    
                    const input = document.createElement('input');
                    input.type = 'hidden';
                    input.name = 'script';
                    input.value = scriptName;
                    
                    form.appendChild(input);
                    document.body.appendChild(form);
                    
                    // Try to submit the form
                    try {
                        form.submit();
                    } catch (error) {
                        // If form submission fails, try opening a new window/tab
                        document.body.removeChild(statusDiv);
                        window.open('run_script.php?script=' + encodeURIComponent(scriptName), '_blank');
                    }
                }
            }
            
            // Function to preview HTML report
            function previewReport(url) {
                document.getElementById('preview-frame').src = url;
                document.getElementById('report-preview').scrollIntoView({behavior: 'smooth'});
            }
            
            // Initialize on page load
            document.addEventListener('DOMContentLoaded', function() {
                // Check for saved dark mode preference
                const darkMode = localStorage.getItem('darkMode');
                if (darkMode === 'enabled') {
                    document.body.classList.add('dark-mode');
                    const themeIcon = document.querySelector('.theme-toggle i');
                    themeIcon.classList.remove('fa-moon');
                    themeIcon.classList.add('fa-sun');
                }
                
                // Set active tab
                const activeTab = localStorage.getItem('activeTab');
                if (activeTab) {
                    const tab = document.querySelector(`.tab[onclick*="${activeTab}"]`);
                    if (tab) {
                        const event = { currentTarget: tab };
                        openTab(event, activeTab);
                    }
                }
            });
            
            // Function to filter plots by search text
            function filterPlots() {
                const input = document.getElementById('plotSearchInput');
                const filter = input.value.toLowerCase();
                const plotItems = document.querySelectorAll('#plotGrid .plot-item');
                
                plotItems.forEach(item => {
                    const name = item.getAttribute('data-name');
                    if (name && name.includes(filter)) {
                        item.style.display = '';
                    } else {
                        item.style.display = 'none';
                    }
                });
            }
            
            // Function to filter plots by type
            function filterPlotsByType(type) {
                const plotItems = document.querySelectorAll('#plotGrid .plot-item');
                const buttons = document.querySelectorAll('.plot-grid + .filter-container .filter-button');
                
                // Update active button
                document.querySelectorAll('.filter-button').forEach(button => {
                    button.classList.remove('active');
                    if (button.innerText.toLowerCase().includes(type.toLowerCase())) {
                        button.classList.add('active');
                    }
                });
                
                if (type === 'all') {
                    // Show all plots
                    plotItems.forEach(item => {
                        item.style.display = '';
                    });
                } else {
                    // Show only plots matching the type
                    plotItems.forEach(item => {
                        if (item.classList.contains(`${type}-plot`)) {
                            item.style.display = '';
                        } else {
                            item.style.display = 'none';
                        }
                    });
                }
            }
            
            // Handle image loading errors
            function handleImageError(img) {
                img.onerror = null; // Prevent infinite loops
                img.src = 'data:image/png;base64,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';
                img.alt = 'Image not available';
                img.title = 'Image not available';
                img.classList.add('broken-image');
            }
            
            // Function to search processed data files
            function searchProcessedData() {
                const input = document.getElementById('processedDataSearch');
                if (!input) {
                    console.error('Search input not found');
                    return;
                }
                
                const filter = input.value.toLowerCase(); // Use toLowerCase instead of toUpperCase for consistency
                const fileItems = document.querySelectorAll('#processed-data .file-grid .file-item');
                
                console.log(`Filtering processed data with: "${filter}" (${fileItems.length} items found)`);
                
                let found = 0;
                fileItems.forEach(item => {
                    const name = item.getAttribute('data-name') || '';
                    const type = item.getAttribute('data-type') || '';
                    const searchText = (name + ' ' + type).toLowerCase();
                    
                    if (searchText.includes(filter)) {
                        item.style.display = '';
                        found++;
                    } else {
                        item.style.display = 'none';
                    }
                });
                
                // Display message if no results found
                const noResultsElement = document.getElementById('no-processed-results');
                if (noResultsElement) {
                    if (found === 0 && filter.length > 0) {
                        noResultsElement.style.display = 'block';
                    } else {
                        noResultsElement.style.display = 'none';
                    }
                }
            }
            
            // Function to search raw data files
            function searchRawData() {
                const input = document.getElementById('rawDataSearch');
                if (!input) {
                    console.error('Raw data search input not found');
                    return;
                }
                
                const filter = input.value.toLowerCase();
                const rows = document.querySelectorAll('#rawDataTable tbody tr');
                
                console.log(`Filtering raw data with: "${filter}" (${rows.length} items found)`);
                
                let found = 0;
                rows.forEach(row => {
                    const cells = row.getElementsByTagName('td');
                    let found = false;
                    
                    for (let i = 0; i < cells.length; i++) {
                        const cell = cells[i];
                        if (cell) {
                            const txtValue = cell.textContent || cell.innerText;
                            if (txtValue.toLowerCase().indexOf(filter) > -1) {
                                found = true;
                                break;
                            }
                        }
                    }
                    
                    row.style.display = found ? '' : 'none';
                });
                
                // Display message if no results found
                const noResultsElement = document.getElementById('no-raw-results');
                if (noResultsElement) {
                    if (Array.from(rows).every(row => row.style.display === 'none') && filter.length > 0) {
                        noResultsElement.style.display = 'block';
                    } else {
                        noResultsElement.style.display = 'none';
                    }
                }
            }
            
            // Function to filter processed data by type
            function filterProcessedDataByType(type) {
                console.log(`Filtering processed data by type: ${type}`);
                
                // Update active button
                document.querySelectorAll('#processed-data .filter-button').forEach(btn => {
                    btn.classList.remove('active');
                    if (btn.getAttribute('data-type') === type) {
                        btn.classList.add('active');
                    }
                });
                
                const fileItems = document.querySelectorAll('#processed-data .file-grid .file-item');
                let found = 0;
                
                fileItems.forEach(item => {
                    const itemType = item.getAttribute('data-type');
                    if (type === 'all' || itemType === type) {
                        item.style.display = '';
                        found++;
                    } else {
                        item.style.display = 'none';
                    }
                });
                
                // Handle no results message
                const noResultsElement = document.getElementById('no-processed-results');
                if (noResultsElement) {
                    if (found === 0) {
                        noResultsElement.style.display = 'block';
                    } else {
                        noResultsElement.style.display = 'none';
                    }
                }
            }
            
            // Function to filter raw data by type
            function filterRawDataByType(type) {
                console.log(`Filtering raw data by type: ${type}`);
                
                // Update active button
                document.querySelectorAll('#raw-data .filter-button').forEach(btn => {
                    btn.classList.remove('active');
                    if (btn.getAttribute('data-type') === type) {
                        btn.classList.add('active');
                    }
                });
                
                const rows = document.querySelectorAll('#rawDataTable tbody tr');
                let found = 0;
                
                rows.forEach(row => {
                    const dataType = row.getAttribute('data-type');
                    const formatType = row.getAttribute('data-format');
                    
                    if (type === 'all' || 
                        dataType === type || 
                        (type === 'csv' && formatType === 'csv') || 
                        (type === 'excel' && formatType === 'excel')) {
                        row.style.display = '';
                        found++;
                    } else {
                        row.style.display = 'none';
                    }
                });
                
                // Handle no results message
                const noResultsElement = document.getElementById('no-raw-results');
                if (noResultsElement) {
                    if (found === 0) {
                        noResultsElement.style.display = 'block';
                    } else {
                        noResultsElement.style.display = 'none';
                    }
                }
            }
        </script>
    </body>
    </html>
    """
    
    # Write the HTML file
    with open(OUTPUT_FILE, 'w') as f:
        # Replace placeholder comments with actual content
        html_content = html_content.replace('<!-- Processed Data Items Will Be Generated Here -->', processed_data_items)
        html_content = html_content.replace('<!-- Processed Data Warning Will Be Displayed If Necessary -->', processed_data_warning)
        
        f.write(html_content)
    
    # Create PHP helper to run scripts (if supported)
    run_php_path = os.path.join(OUTPUT_DIR, "run.php")
    php_content = """<?php
header('Content-Type: text/plain');

// Sanitize input to prevent command injection
$script = isset($_GET['script']) ? $_GET['script'] : '';
$script = basename($script); // Only allow the filename, not a path

$allowed_scripts = array(
    'generate_investment_report.py',
    'validate_model.py',
    'train_models.py',
    'train_all_models.py'
);

if (!in_array($script, $allowed_scripts)) {
    echo "Error: Script not allowed";
    exit(1);
}

// Execute the script
$command = "python ../" . escapeshellarg($script) . " 2>&1";
$output = array();
$return_var = 0;
exec($command, $output, $return_var);

echo implode("\\n", $output);
echo "\\nExit code: " . $return_var;
?>
"""
    
    try:
        with open(run_php_path, 'w') as f:
            f.write(php_content)
    except:
        print("Note: Couldn't create PHP helper file (not critical)")
    
    print(f"Dashboard generated at: {OUTPUT_FILE}")
    
    def debug_tabs():
        """Check all tab IDs to ensure they match their onClick handlers."""
        # This function will be called at the end of generate_dashboard()
        html_file = OUTPUT_FILE
        if not os.path.exists(html_file):
            print("HTML file not found for debugging")
            return
        
        try:
            with open(html_file, 'r') as f:
                content = f.read()
                
            # Find all tab definitions
            tab_pattern = r'<div class="tab.*?onclick="openTab\(event, \'(.*?)\'\)"'
            tabs = re.findall(tab_pattern, content)
            
            # Find all tab content divs
            content_pattern = r'<div id="(.*?)" class="tab-content"'
            contents = re.findall(content_pattern, content)
            
            print("\n--- Tab Debugging ---")
            print(f"Found {len(tabs)} tab handlers: {tabs}")
            print(f"Found {len(contents)} content divs: {contents}")
            
            # Check for mismatches
            missing_tabs = [c for c in contents if c not in tabs]
            missing_contents = [t for t in tabs if t not in contents]
            
            if missing_tabs:
                print(f"WARNING: Found content divs with no tab handlers: {missing_tabs}")
            if missing_contents:
                print(f"WARNING: Found tab handlers with no content divs: {missing_contents}")
                
            if not missing_tabs and not missing_contents:
                print("All tabs match their content divs.")
        except Exception as e:
            print(f"Error analyzing tabs: {e}")
    
    # Debug tab issues
    debug_tabs()
    
    return OUTPUT_FILE

def main():
    """Generate dashboard and open it in browser."""
    ensure_directories()
    dashboard_file = generate_dashboard()
    
    print(f"Opening dashboard in browser...")
    abs_path = os.path.abspath(dashboard_file)
    webbrowser.open(f'file://{abs_path}')
    
    print("Dashboard successfully generated and opened!")

if __name__ == "__main__":
    main() 