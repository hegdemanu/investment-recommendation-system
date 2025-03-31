"""
Dashboard Generator Module

This module generates the HTML dashboard that integrates all reports, visualizations,
and data into a single interactive interface with a modular architecture.
"""
import os
import json
import glob
import base64
import re
import webbrowser
import shutil
from datetime import datetime
from pathlib import Path

# Add project root to path to import settings
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
# Add archive path to system path for imports
archive_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if archive_path not in sys.path:
    sys.path.append(archive_path)

try:
    from settings import RESULTS_DIR
except ImportError:
    RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'results')

try:
    from config.settings import (
        DASHBOARD_FILE, DASHBOARD_DIR, DASHBOARD_JSON_DIR, TRAINING_DIR, VALIDATION_DIR, 
        REPORTS_DIR, MODELS_DIR, DATA_DIR
    )
except ImportError:
    # Define fallback paths if settings can't be imported
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    DASHBOARD_FILE = os.path.join(base_dir, "results", "dashboard", "dashboard.html")
    DASHBOARD_DIR = os.path.join(base_dir, "results", "dashboard")
    DASHBOARD_JSON_DIR = os.path.join(DASHBOARD_DIR, "json")
    TRAINING_DIR = os.path.join(base_dir, "results", "training")
    VALIDATION_DIR = os.path.join(base_dir, "results", "validation")
    REPORTS_DIR = os.path.join(base_dir, "results", "reports")
    MODELS_DIR = os.path.join(base_dir, "models")
    DATA_DIR = os.path.join(base_dir, "data")

# Fix imports with local utility functions
def file_to_base64(file_path):
    """Convert file to base64 encoding."""
    try:
        with open(file_path, 'rb') as f:
            encoded = base64.b64encode(f.read()).decode('utf-8')
            return encoded
    except Exception as e:
        print(f"Error encoding file {file_path}: {e}")
        return ""

def load_json_file(file_path):
    """Load and parse a JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading JSON file {file_path}: {e}")
        return None

# Sample generator functions (simplified)
def create_sample_json_file(file_path, data=None):
    """Create a sample JSON file."""
    if data is None:
        data = {
            "timestamp": datetime.now().isoformat(),
            "sample": True,
            "metrics": {
                "accuracy": 0.85,
                "precision": 0.83,
                "recall": 0.81,
                "f1_score": 0.82
            },
            "predictions": [
                {"date": "2023-01-01", "value": 100},
                {"date": "2023-01-02", "value": 105},
                {"date": "2023-01-03", "value": 103}
            ]
        }
    
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        return True
    except Exception as e:
        print(f"Error creating sample JSON file {file_path}: {e}")
        return False

def create_sample_csv_file(file_path, data=None):
    """Create a sample CSV file."""
    if data is None:
        # Create simple sample data
        sample_data = "date,open,high,low,close,volume\n"
        sample_data += "2023-01-01,100,105,98,102,1000000\n"
        sample_data += "2023-01-02,102,107,100,105,1200000\n"
        sample_data += "2023-01-03,105,108,103,107,1100000\n"
    else:
        sample_data = data
    
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(sample_data)
        return True
    except Exception as e:
        print(f"Error creating sample CSV file {file_path}: {e}")
        return False

# Define paths for modular structure
DASHBOARD_CSS_DIR = os.path.join(DASHBOARD_DIR, "css")
DASHBOARD_JS_DIR = os.path.join(DASHBOARD_DIR, "js")
DASHBOARD_JS_MODULES_DIR = os.path.join(DASHBOARD_JS_DIR, "modules")

# Create symlink to index.html for compatibility
INDEX_HTML = os.path.join(DASHBOARD_DIR, "index.html")

# Explicitly set dashboard file path
DASHBOARD_FILE = os.path.join(DASHBOARD_DIR, "dashboard.html")

def ensure_directories():
    """Ensure all necessary directories exist."""
    # Data directories
    os.makedirs(DASHBOARD_DIR, exist_ok=True)
    os.makedirs(DASHBOARD_JSON_DIR, exist_ok=True)
    os.makedirs(TRAINING_DIR, exist_ok=True)
    os.makedirs(VALIDATION_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(os.path.join(DATA_DIR, "stocks"), exist_ok=True)
    os.makedirs(os.path.join(DATA_DIR, "mutual_funds"), exist_ok=True)
    os.makedirs(os.path.join(DATA_DIR, "uploads"), exist_ok=True)
    os.makedirs(os.path.join(DATA_DIR, "processed"), exist_ok=True)
    os.makedirs(os.path.join(DATA_DIR, "raw"), exist_ok=True)
    os.makedirs(os.path.join(DATA_DIR, "sector_info"), exist_ok=True)
    
    # Dashboard modular structure directories
    os.makedirs(DASHBOARD_CSS_DIR, exist_ok=True)
    os.makedirs(DASHBOARD_JS_DIR, exist_ok=True)
    os.makedirs(DASHBOARD_JS_MODULES_DIR, exist_ok=True)
    
    # Create a symlink from index.html to dashboard.html if it doesn't exist
    if not os.path.exists(INDEX_HTML):
        try:
            os.symlink(os.path.basename(DASHBOARD_FILE), INDEX_HTML)
            print(f"Created symlink: {INDEX_HTML} -> {os.path.basename(DASHBOARD_FILE)}")
        except Exception as e:
            print(f"Error creating symlink: {e}")
            # Fallback to copy if symlink fails
            try:
                shutil.copy2(DASHBOARD_FILE, INDEX_HTML)
                print(f"Copied {DASHBOARD_FILE} to {INDEX_HTML} (symlink failed)")
            except Exception as copy_e:
                print(f"Error copying file: {copy_e}")

def get_image_files():
    """Get all image files from the results and models directories."""
    image_files = []
    image_types = ['.png', '.jpg', '.jpeg', '.svg', '.gif']
    directories = [TRAINING_DIR, VALIDATION_DIR, REPORTS_DIR, MODELS_DIR]
    
    for directory in directories:
        for img_type in image_types:
            files = glob.glob(os.path.join(directory, f"**/*{img_type}"), recursive=True)
            if files:
                print(f"Found {len(files)} {img_type} files in {directory}")
            image_files.extend(files)
    
    return image_files

def get_sentiment_model_visualizations():
    """Get sentiment model comparison visualizations."""
    sentiment_visuals = []
    sentiment_dir = os.path.join(RESULTS_DIR, "sentiment_models")
    
    if os.path.exists(sentiment_dir):
        for img_type in ['.png', '.jpg', '.jpeg', '.svg', '.gif']:
            files = glob.glob(os.path.join(sentiment_dir, f"*{img_type}"))
            if files:
                print(f"Found {len(files)} sentiment model visualizations")
            sentiment_visuals.extend(files)
    
    return sentiment_visuals

def get_json_files():
    """Get all JSON files from the results and models directories."""
    json_files = []
    directories = [TRAINING_DIR, VALIDATION_DIR, REPORTS_DIR, MODELS_DIR, RESULTS_DIR]
    
    for directory in directories:
        files = glob.glob(os.path.join(directory, "**/*.json"), recursive=True)
        if files:
            print(f"Found {len(files)} JSON files in {directory}")
        json_files.extend(files)
    
    return json_files

def get_raw_data_files():
    """Get all raw data files."""
    data_files = []
    data_types = ['.xlsx', '.xls', '.csv']
    directories = [DATA_DIR, RESULTS_DIR]
    
    for directory in directories:
        for data_type in data_types:
            files = glob.glob(os.path.join(directory, f"**/*{data_type}"), recursive=True)
            if files:
                print(f"Found {len(files)} {data_type} files in {directory}")
            data_files.extend(files)
    
    # Add specific subdirectories to check
    subdirs = ["uploads", "stocks", "mutual_funds", "processed", "raw"]
    for subdir in subdirs:
        for data_type in data_types:
            files = glob.glob(os.path.join(DATA_DIR, subdir, f"**/*{data_type}"), recursive=True)
            if files:
                print(f"Found {len(files)} {data_type} files in {os.path.join(DATA_DIR, subdir)}")
            data_files.extend(files)
    
    # Remove potential duplicates
    data_files = list(set(data_files))
    
    print(f"After recursive search: Found {len(data_files)} total raw data files")
    if data_files:
        print("First 5 raw data files:")
        for i, file in enumerate(data_files[:5]):
            print(f"  {i+1}. {file}")
    
    return data_files

def get_processed_data():
    """Get all processed data files."""
    processed_data = []
    processed_dir = os.path.join(DATA_DIR, "processed")
    
    # Check for processed data files
    data_types = ['.csv', '.xlsx', '.xls', '.json']
    for data_type in data_types:
        files = glob.glob(os.path.join(processed_dir, f"**/*{data_type}"), recursive=True)
        processed_data.extend(files)
    
    return processed_data

def get_sector_data():
    """Get sector information for the pie chart."""
    sector_data = {}
    stock_sectors_file = os.path.join(DATA_DIR, "sector_info", "stock_sectors.json")
    mf_sectors_file = os.path.join(DATA_DIR, "sector_info", "mutual_fund_sectors.json")
    
    # Initialize with default sector data if files don't exist
    if not os.path.exists(stock_sectors_file) or not os.path.exists(mf_sectors_file):
        # Default sectors with colors - using more professional names
        sector_data = {
            "Information Technology": {"value": 25, "color": "#3498db"},
            "Financial Services": {"value": 15, "color": "#f39c12"},
            "Energy & Utilities": {"value": 10, "color": "#e74c3c"},
            "Healthcare": {"value": 10, "color": "#2ecc71"},
            "Consumer Discretionary": {"value": 8, "color": "#9b59b6"},
            "Industrials": {"value": 12, "color": "#1abc9c"},
            "Consumer Staples": {"value": 10, "color": "#34495e"},
            "Diversified": {"value": 10, "color": "#7f8c8d"}
        }
        return sector_data
    
    # Load the sector data files
    try:
        stock_sectors = load_json_file(stock_sectors_file)
        mf_sectors = load_json_file(mf_sectors_file)
        
        # Professional sector name mapping
        sector_name_mapping = {
            "Technology": "Information Technology",
            "Banking": "Financial Services",
            "Energy": "Energy & Utilities",
            "Pharmaceuticals": "Healthcare",
            "Retail": "Consumer Discretionary",
            "Manufacturing": "Industrials",
            "FMCG": "Consumer Staples",
            "Automotive": "Industrials",
            "Paints": "Materials",
            "Equity": "Equity Funds",
            "Debt": "Fixed Income",
            "Hybrid": "Balanced Funds"
        }
        
        # Get training summary to understand which stocks/mfs are used
        training_summary_file = os.path.join(RESULTS_DIR, "training", "training_summary.json")
        if os.path.exists(training_summary_file):
            training_summary = load_json_file(training_summary_file)
            
            # Count stocks by sector
            sector_counts = {}
            
            # Process stocks
            if training_summary and "stocks" in training_summary:
                stock_details = training_summary["stocks"].get("details", {})
                for ticker in stock_details:
                    # Find which sector this stock belongs to
                    for sector, stocks in stock_sectors["sectors"].items():
                        if ticker in stocks:
                            # Use professional sector name if available
                            prof_sector = sector_name_mapping.get(sector, sector)
                            if prof_sector not in sector_counts:
                                sector_counts[prof_sector] = 0
                            sector_counts[prof_sector] += 1
            
            # Process mutual funds
            if training_summary and "mutual_funds" in training_summary:
                mf_details = training_summary["mutual_funds"].get("details", {})
                for fund in mf_details:
                    # Find which sector this mutual fund belongs to
                    for sector, funds in mf_sectors["sectors"].items():
                        if fund in funds:
                            # Use professional sector name if available
                            prof_sector = sector_name_mapping.get(sector, sector)
                            if prof_sector not in sector_counts:
                                sector_counts[prof_sector] = 0
                            sector_counts[prof_sector] += 1
            
            # If no data is found, use default values with professional names
            if not sector_counts:
                sector_counts = {
                    "Information Technology": 2,
                    "Financial Services": 1,
                    "Energy & Utilities": 1,
                    "Healthcare": 1,
                    "Consumer Discretionary": 1,
                    "Industrials": 2,
                    "Consumer Staples": 1,
                    "Diversified": 3
                }
            
            # Assign colors to sectors for the pie chart - use a professional color palette
            colors = [
                "#0075C9", "#FFB81C", "#DA291C", "#00A551", 
                "#702F8A", "#00ADEF", "#414042", "#231F20",
                "#D35213", "#002855", "#553592", "#00C389"
            ]
            
            # Convert counts to percentages and assign colors
            total = sum(sector_counts.values())
            for i, (sector, count) in enumerate(sector_counts.items()):
                percentage = round((count / total) * 100)
                sector_data[sector] = {
                    "value": percentage,
                    "color": colors[i % len(colors)]
                }
            
            return sector_data
        else:
            # Default sectors with colors with professional names
            sector_data = {
                "Information Technology": {"value": 25, "color": "#0075C9"},
                "Financial Services": {"value": 15, "color": "#FFB81C"},
                "Energy & Utilities": {"value": 10, "color": "#DA291C"},
                "Healthcare": {"value": 10, "color": "#00A551"},
                "Consumer Discretionary": {"value": 8, "color": "#702F8A"},
                "Industrials": {"value": 12, "color": "#00ADEF"},
                "Consumer Staples": {"value": 10, "color": "#414042"},
                "Diversified": {"value": 10, "color": "#231F20"}
            }
            return sector_data
            
    except Exception as e:
        print(f"Error loading sector data: {str(e)}")
        # Default sectors with professional names and colors
        sector_data = {
            "Information Technology": {"value": 25, "color": "#0075C9"},
            "Financial Services": {"value": 15, "color": "#FFB81C"},
            "Energy & Utilities": {"value": 10, "color": "#DA291C"},
            "Healthcare": {"value": 10, "color": "#00A551"},
            "Consumer Discretionary": {"value": 8, "color": "#702F8A"},
            "Industrials": {"value": 12, "color": "#00ADEF"},
            "Consumer Staples": {"value": 10, "color": "#414042"},
            "Diversified": {"value": 10, "color": "#231F20"}
        }
        return sector_data

def generate_sector_pie_data():
    """Generate JavaScript data for sector pie chart without any problematic formatting."""
    # Get sector data (this will retrieve allocation percentages)
    sector_data = get_sector_data()
    
    # Generate JavaScript objects for each sector in a clean format
    js_objects = []
    for sector, data in sector_data.items():
        js_objects.append('{label:"' + sector + '",value:' + str(data["value"]) + ',color:"' + data["color"] + '"}')
    
    # Return clean comma-separated string without square brackets
    return ",".join(js_objects)

def generate_chart_items(image_files):
    """Generate HTML for chart items."""
    if not image_files:
        return """<div class="no-results">No chart files found. Generate reports first.</div>"""
    
    items_html = ""
    for file in image_files:
        file_name = os.path.basename(file)
        relative_path = os.path.relpath(file)
        file_directory = os.path.dirname(relative_path)
        
        # Create a title from the filename
        title = os.path.splitext(file_name)[0]
        title = title.replace("_", " ").replace("-", " ")
        title = ' '.join(word.capitalize() for word in title.split())
        
        # Add directory info to categorize
        category = ""
        if "training" in file_directory.lower():
            category = "Training"
        elif "validation" in file_directory.lower():
            category = "Validation"
        elif "reports" in file_directory.lower():
            category = "Report"
        elif "models" in file_directory.lower():
            category = "Model"
        elif "predictions" in file_directory.lower():
            category = "Prediction"
        elif "sentiment_models" in file_directory.lower():
            category = "Sentiment Analysis"
        
        if category:
            title = f"{category}: {title}"
        
        try:
            data_url = file_to_base64(file)
            items_html += f"""
            <div class="chart-card">
                <img src="{data_url}" alt="{title}" class="chart-img">
                <div style="padding: 1rem;">
                    <h3>{title}</h3>
                    <p>Path: {relative_path}</p>
                </div>
            </div>
            """
        except Exception as e:
            items_html += f"""
            <div class="chart-card">
                <div style="padding: 1rem; color: var(--error-color);">
                    <h3>Error Loading: {title}</h3>
                    <p>Path: {relative_path}</p>
                    <p>Error: {str(e)}</p>
                </div>
            </div>
            """
    
    return items_html

def generate_json_items(json_files):
    """Generate HTML for JSON items."""
    if not json_files:
        return """<div class="no-results">No JSON report files found. Generate reports first.</div>"""
    
    items_html = ""
    dashboard_json_files = []
    
    # Copy JSON files to dashboard/json directory for easier access
    for file in json_files:
        file_name = os.path.basename(file)
        dashboard_json_path = os.path.join(DASHBOARD_JSON_DIR, file_name)
        
        # Copy file to dashboard/json directory
        try:
            shutil.copy2(file, dashboard_json_path)
            dashboard_json_files.append(dashboard_json_path)
        except Exception as e:
            print(f"Warning: Could not copy {file} to dashboard directory: {str(e)}")
    
    # Use both original and dashboard JSON files
    all_json_files = list(set(json_files + dashboard_json_files))
    
    for file in all_json_files:
        file_name = os.path.basename(file)
        relative_path = os.path.relpath(file)
        file_directory = os.path.dirname(relative_path)
        
        # Create a title from the filename
        title = os.path.splitext(file_name)[0]
        title = title.replace("_", " ").replace("-", " ")
        title = ' '.join(word.capitalize() for word in title.split())
        
        # Add directory info to categorize
        category = ""
        if "training" in file_directory.lower():
            category = "Training"
        elif "validation" in file_directory.lower():
            category = "Validation"
        elif "reports" in file_directory.lower():
            category = "Report"
        elif "models" in file_directory.lower():
            category = "Model"
        elif "dashboard/json" in file_directory.lower():
            # Skip duplicate dashboard files in the UI
            continue
        
        if category:
            title = f"{category}: {title}"
        
        # Try to load the JSON to get a preview
        preview = ""
        try:
            data = load_json_file(file)
            if data:
                # Find all keys at the root level
                root_keys = list(data.keys())
                preview = "Contains: " + ", ".join(root_keys[:3])
                if len(root_keys) > 3:
                    preview += f" and {len(root_keys) - 3} more"
        except Exception as e:
            preview = f"Error loading preview: {str(e)}"
        
        onclick = f"openJsonModal('{relative_path}', '{title}')"
        
        items_html += f"""
        <div class="json-card" onclick="{onclick}">
            <h3>{title}</h3>
            <p>{preview}</p>
            <p class="mt-1">Path: {relative_path}</p>
        </div>
        """
    
    return items_html

def generate_raw_data_items(data_files):
    """Generate HTML for raw data items."""
    if not data_files:
        return """<tr><td colspan="4" class="no-results">No data files found. Add data files to the data directory.</td></tr>"""
    
    items_html = ""
    for file in data_files:
        file_name = os.path.basename(file)
        relative_path = os.path.relpath(file)
        
        # Determine file type
        file_ext = os.path.splitext(file_name)[1].lower()
        file_type = "Unknown"
        if file_ext == '.csv':
            file_type = "CSV"
        elif file_ext in ['.xlsx', '.xls']:
            file_type = "Excel"
        elif file_ext == '.json':
            file_type = "JSON"
        
        # Determine file size
        try:
            file_size = os.path.getsize(file)
            if file_size < 1024:
                file_size_str = f"{file_size} B"
            elif file_size < 1024 * 1024:
                file_size_str = f"{file_size / 1024:.1f} KB"
            else:
                file_size_str = f"{file_size / (1024 * 1024):.1f} MB"
        except:
            file_size_str = "Unknown"
        
        items_html += f"""
        <tr>
            <td>{file_name}</td>
            <td>{relative_path}</td>
            <td>{file_type}</td>
            <td>{file_size_str}</td>
        </tr>
        """
    
    return items_html

def generate_processed_data_items(processed_data):
    """Generate HTML for processed data items."""
    if not processed_data:
        warning = """<div class="no-results" style="color: var(--warning-color);">
            No processed data files found. Process raw data first or upload processed data files to the data/processed directory.
        </div>"""
        return warning, """<div class="no-results">No processed data available.</div>"""
    
    items_html = ""
    for file in processed_data:
        file_name = os.path.basename(file)
        relative_path = os.path.relpath(file)
        
        # Create a title from the filename
        title = os.path.splitext(file_name)[0]
        title = title.replace("_", " ").replace("-", " ")
        title = ' '.join(word.capitalize() for word in title.split())
        
        # Determine file type and custom display for different processed data types
        file_ext = os.path.splitext(file_name)[1].lower()
        
        if file_ext == '.json':
            # For JSON files, try to load and display a preview
            try:
                data = load_json_file(file)
                preview = ""
                if data:
                    # Try to find key data points to display
                    if isinstance(data, dict):
                        preview = "Contains: " + ", ".join(list(data.keys())[:3])
                    elif isinstance(data, list):
                        preview = f"Contains {len(data)} items"
                
                onclick = f"openJsonModal('{relative_path}', '{title}')"
                
                items_html += f"""
                <div class="chart-card">
                    <div style="padding: 1rem;">
                        <h3>{title}</h3>
                        <p>{preview}</p>
                        <p class="mt-1">Path: {relative_path}</p>
                        <button class="filter-btn" onclick="{onclick}">View JSON</button>
                    </div>
                </div>
                """
            except Exception as e:
                items_html += f"""
                <div class="chart-card">
                    <div style="padding: 1rem; color: var(--error-color);">
                        <h3>Error Loading: {title}</h3>
                        <p>Path: {relative_path}</p>
                        <p>Error: {str(e)}</p>
                    </div>
                </div>
                """
        else:
            # For CSV/Excel files, just show basic info
            items_html += f"""
            <div class="chart-card">
                <div style="padding: 1rem;">
                    <h3>{title}</h3>
                    <p>File Type: {file_ext[1:].upper()}</p>
                    <p class="mt-1">Path: {relative_path}</p>
                </div>
            </div>
            """
    
    # Return empty warning if we have processed data
    return "", items_html

def get_rag_reports() -> list:
    """Get AI-generated reports from the reports directory."""
    report_dir = os.path.join(RESULTS_DIR, 'reports')
    reports = []
    
    if os.path.exists(report_dir):
        for filename in os.listdir(report_dir):
            if filename.endswith('.json'):
                try:
                    report_path = os.path.join(report_dir, filename)
                    with open(report_path, 'r') as f:
                        report = json.load(f)
                        reports.append(report)
                except Exception as e:
                    print(f"Error loading report {filename}: {str(e)}")
    
    return reports

def generate_report_form() -> str:
    """Generate the HTML form for creating new AI reports."""
    return """
    <div class="report-generate-form">
        <h3>Generate AI Financial Analysis</h3>
        <form id="reportForm" onsubmit="submitReportForm(event)">
            <div class="form-control">
                <label for="reportQuery">Financial Query</label>
                <textarea id="reportQuery" required placeholder="Ask a financial question or specify what you want to analyze..."></textarea>
            </div>
            <div class="form-control">
                <button type="submit" class="report-btn">Generate AI Analysis</button>
                <span id="reportStatus" style="margin-left: 1rem; display: none;">Generating analysis...</span>
            </div>
        </form>
    </div>
    """

def generate_report_section(reports: list) -> str:
    """Generate HTML for the RAG reports section."""
    if not reports:
        return """
        <div class="report-section">
            <h2>AI Financial Analysis</h2>
            <div class="no-results">No analysis reports found. Use the form above to generate AI-powered financial insights.</div>
        </div>
        """
    
    html = '<div class="report-section"><h2>AI Financial Analysis</h2><div class="report-grid">'
    
    # Sort reports by generation date, newest first
    sorted_reports = sorted(reports, 
                           key=lambda x: x.get("generated_at", ""), 
                           reverse=True)
    
    for report in sorted_reports:
        # Determine risk level class based on content
        risk_class = ""
        risk_assessment = report.get("risk_assessment", "").lower()
        if "high risk" in risk_assessment or "severe" in risk_assessment:
            risk_class = "risk-high"
        elif "moderate risk" in risk_assessment or "medium risk" in risk_assessment:
            risk_class = "risk-medium"
        elif "low risk" in risk_assessment or "minimal risk" in risk_assessment:
            risk_class = "risk-low"
        
        html += f'''
        <div class="report-card">
            <h3>{report.get("title", "Analysis Report")}</h3>
            <div class="summary">{report.get("context_summary", report.get("summary", ""))}</div>
            <div class="recommendations">{report.get("recommendations", "")}</div>
            <div class="risk-assessment {risk_class}">{report.get("risk_assessment", "")}</div>
            <p class="model-info">Generated: {report.get("generated_at", "Unknown")} | Model: {report.get("model", "DeepSeek-Qwen")}</p>
        </div>'''
    
    html += "</div></div>"
    return html

def generate_metrics_section():
    """Generate HTML for the metrics section. Currently returns empty content."""
    return ""

def generate_stocks_section():
    """Generate HTML for the stocks section. Currently returns empty content."""
    return ""

def generate_analysis_section():
    """Generate HTML for the analysis section. Currently returns empty content."""
    return ""

def generate_predictions_section():
    """Generate HTML for the predictions section. Currently returns empty content."""
    return ""

def create_dashboard_html():
    """Create the dashboard HTML file with the base structure."""
    dashboard_html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Investment Analysis Dashboard</title>
    
    <!-- CSS Imports - Only include files that exist -->
    <link rel="stylesheet" href="css/base.css">
    <link rel="stylesheet" href="css/layout.css">
    <link rel="stylesheet" href="css/header.css">
    <link rel="stylesheet" href="css/main.css">
    <link rel="stylesheet" href="css/modals.css">
    <link rel="stylesheet" href="css/themes.css">
    
    <!-- External libraries if needed -->
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</head>
<body>
    <header>
        <div class="logo">
            <h1>Investment Analysis Dashboard</h1>
        </div>
        <div class="header-controls">
            <div class="timestamp">
                Last Updated: <span id="last-updated">-</span>
            </div>
            <div class="theme-switcher">
                <label for="theme-switch">Dark Mode</label>
                <input type="checkbox" id="theme-switch">
            </div>
        </div>
        <nav>
            <button class="nav-button active" data-target="home-section">Home</button>
            <button class="nav-button" data-target="charts-section">Charts</button>
            <button class="nav-button" data-target="json-section">JSON Reports</button>
            <button class="nav-button" data-target="sentiment-section">Sentiment Analysis</button>
            <button class="nav-button" data-target="data-section">Data</button>
            <button class="nav-button" data-target="processed-section">Processed Data</button>
            <button class="nav-button" data-target="reports-section">Reports</button>
            <button class="nav-button" data-target="expert-section">Expert Analysis</button>
        </nav>
    </header>
    
    <main>
        <!-- Home Section -->
        <section id="home-section">
            <div class="section-header">
                <h2>Investment Portfolio Overview</h2>
            </div>
            <div class="content-grid">
                <div class="card summary-card">
                    <h3>Portfolio Summary</h3>
                    <div class="card-content">
                        <p>This dashboard provides a comprehensive overview of your investment portfolio analysis.</p>
                        <button class="action-button" data-modal="diversification-modal">View Diversification</button>
                    </div>
                </div>
                
                <div class="card chart-card">
                    <h3>Portfolio Allocation</h3>
                    <div class="card-content">
                        <canvas id="portfolio-chart"></canvas>
                    </div>
                </div>
                
                <div class="card performance-card">
                    <h3>Performance Metrics</h3>
                    <div class="card-content">
                        <div class="metric">
                            <span class="metric-label">Total Return</span>
                            <span class="metric-value" id="total-return">-</span>
                        </div>
                        <div class="metric">
                            <span class="metric-label">Annual Growth</span>
                            <span class="metric-value" id="annual-growth">-</span>
                        </div>
                        <div class="metric">
                            <span class="metric-label">Risk Score</span>
                            <span class="metric-value" id="risk-score">-</span>
                        </div>
                    </div>
                </div>
            </div>
        </section>
        
        <!-- Charts Section -->
        <section id="charts-section">
            <div class="section-header">
                <h2>Charts &amp; Visualizations</h2>
                <div class="controls">
                    <select id="chart-filter">
                        <option value="all">All Charts</option>
                        <option value="stocks">Stocks</option>
                        <option value="mutual-funds">Mutual Funds</option>
                        <option value="predictions">Predictions</option>
                    </select>
                </div>
            </div>
            <div id="charts-container" class="content-grid">
                <!-- Charts will be dynamically inserted here -->
            </div>
        </section>
        
        <!-- JSON Reports Section -->
        <section id="json-section">
            <div class="section-header">
                <h2>JSON Reports</h2>
                <div class="controls">
                    <select id="json-filter">
                        <option value="all">All Reports</option>
                        <option value="stocks">Stocks</option>
                        <option value="mutual-funds">Mutual Funds</option>
                        <option value="predictions">Predictions</option>
                    </select>
                </div>
            </div>
            <div id="json-container" class="content-grid">
                <!-- JSON cards will be dynamically inserted here -->
            </div>
        </section>
        
        <!-- Sentiment Analysis Section -->
        <section id="sentiment-section">
            <div class="section-header">
                <h2>Sentiment Analysis</h2>
            </div>
            <div class="content-grid">
                <div class="card full-width">
                    <h3>Market Sentiment Overview</h3>
                    <div class="card-content">
                        <p>This section displays sentiment analysis from news articles and social media.</p>
                    </div>
                </div>
                
                <div id="sentiment-charts" class="content-grid">
                    <!-- Sentiment charts will be inserted here -->
                </div>
            </div>
        </section>
        
        <!-- Data Section -->
        <section id="data-section">
            <div class="section-header">
                <h2>Raw Data</h2>
                <div class="controls">
                    <select id="data-filter">
                        <option value="all">All Data</option>
                        <option value="stocks">Stocks</option>
                        <option value="mutual-funds">Mutual Funds</option>
                        <option value="other">Other</option>
                    </select>
                </div>
            </div>
            <div id="data-container" class="content-grid">
                <!-- Data items will be dynamically inserted here -->
                <p class="no-data">Data files will appear here.</p>
            </div>
        </section>
        
        <!-- Processed Data Section -->
        <section id="processed-section">
            <div class="section-header">
                <h2>Processed Data</h2>
            </div>
            <div id="processed-data-container" class="content-grid">
                <!-- Processed data items will be dynamically inserted here -->
                <p class="no-data">Processed data files will appear here.</p>
            </div>
        </section>
        
        <!-- Reports Section -->
        <section id="reports-section">
            <div class="section-header">
                <h2>Generated Reports</h2>
            </div>
            <div class="content-grid">
                <div class="card full-width">
                    <h3>Generate New Report</h3>
                    <div class="card-content">
                        <form id="report-form">
                            <div class="form-group">
                                <label for="report-type">Report Type:</label>
                                <select id="report-type">
                                    <option value="portfolio">Portfolio Analysis</option>
                                    <option value="stock">Stock Analysis</option>
                                    <option value="mutual-fund">Mutual Fund Analysis</option>
                                    <option value="comparison">Comparison</option>
                                </select>
                            </div>
                            <div class="form-group">
                                <label for="report-target">Target:</label>
                                <input type="text" id="report-target" placeholder="Enter stock/fund symbol or 'all'">
                            </div>
                            <div class="form-group">
                                <label for="report-period">Time Period:</label>
                                <select id="report-period">
                                    <option value="1m">1 Month</option>
                                    <option value="3m">3 Months</option>
                                    <option value="6m">6 Months</option>
                                    <option value="1y" selected>1 Year</option>
                                    <option value="3y">3 Years</option>
                                    <option value="5y">5 Years</option>
                                </select>
                            </div>
                            <div class="form-actions">
                                <button type="submit" class="primary-button">Generate Report</button>
                            </div>
                        </form>
                    </div>
                </div>
                
                <div id="reports-list" class="content-grid">
                    <!-- Reports will be dynamically inserted here -->
                    <p class="no-data">No reports available. Generate a report using the form above.</p>
                </div>
            </div>
        </section>
        
        <!-- Expert Analysis Section -->
        <section id="expert-section">
            <div class="section-header">
                <h2>Expert Analysis</h2>
            </div>
            <div class="content-grid">
                <div class="card full-width">
                    <h3>AI Investment Advisor</h3>
                    <div class="card-content">
                        <div id="advisor-chat">
                            <div class="chat-history" id="chat-history">
                                <div class="chat-message system">
                                    <p>Hello! I'm your AI investment advisor. How can I help you today?</p>
                                </div>
                            </div>
                            <div class="chat-input">
                                <input type="text" id="user-query" placeholder="Ask a question about your investments...">
                                <button id="send-query" class="primary-button">Send</button>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </section>
    </main>
    
    <!-- Modals -->
    <div id="json-viewer-modal" class="modal">
        <div class="modal-header">
            <h2 id="modal-json-title" class="modal-title">JSON Viewer</h2>
            <button class="modal-close">&times;</button>
        </div>
        <div class="modal-body">
            <div id="json-viewer-content">
                <!-- JSON will be displayed here -->
            </div>
        </div>
    </div>
    
    <div id="image-preview-modal" class="modal">
        <div class="modal-header">
            <h2 id="modal-image-title" class="modal-title">Chart View</h2>
            <button class="modal-close">&times;</button>
        </div>
        <div class="modal-body">
            <div class="modal-image-container">
                <img id="modal-image" src="" alt="Chart Preview">
            </div>
        </div>
    </div>
    
    <div id="diversification-modal" class="modal">
        <div class="modal-header">
            <h2 class="modal-title">Portfolio Diversification</h2>
            <button class="modal-close">&times;</button>
        </div>
        <div class="modal-body">
            <div id="diversification-content">
                <!-- Diversification info will be displayed here -->
            </div>
        </div>
    </div>
    
    <!-- Scripts - Use regular script tags instead of modules to avoid CORS issues -->
    <script src="js/data.js"></script>
    <script src="js/charts-data.js"></script>
    <script src="js/debug.js"></script>
    <script src="js/app.js"></script>
</body>
</html>"""

    try:
        # Make sure the directory exists
        os.makedirs(os.path.dirname(DASHBOARD_FILE), exist_ok=True)
        
        # Write the HTML to the file
        with open(DASHBOARD_FILE, 'w') as f:
            f.write(dashboard_html)
        
        # Check if the file was created and has content
        if os.path.exists(DASHBOARD_FILE) and os.path.getsize(DASHBOARD_FILE) > 0:
            print(f"Created dashboard HTML: {DASHBOARD_FILE} ({os.path.getsize(DASHBOARD_FILE)} bytes)")
        else:
            print(f"ERROR: Dashboard HTML file was not created properly: {DASHBOARD_FILE}")
    except Exception as e:
        print(f"Error creating dashboard HTML file: {str(e)}")
    
    return DASHBOARD_FILE

def generate_dashboard_data():
    """
    Generate dynamic data for the dashboard.
    This creates a JS file with data instead of embedding in HTML.
    """
    # Ensure the js directory exists
    os.makedirs(DASHBOARD_JS_DIR, exist_ok=True)
    
    # Get data to include
    sector_data = get_sector_data()
    
    # Create a data.js file
    data_js_content = f"""/**
 * Dashboard Data Module
 * Contains data for the dashboard
 */
 
// Portfolio sector data
window.portfolioSectorData = {json.dumps(sector_data, indent=2)};

// Last updated timestamp
window.lastUpdated = "{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}";
"""
    
    # Write the data.js file
    data_js_path = os.path.join(DASHBOARD_JS_DIR, "data.js")
    with open(data_js_path, 'w') as f:
        f.write(data_js_content)
    
    print(f"Generated dashboard data.js: {data_js_path}")

def generate_dashboard(open_browser=False):
    """Generate the dashboard HTML file and supporting files."""
    print("Generating investment dashboard...")
    ensure_directories()
    
    # Always recreate the dashboard HTML file to ensure it's up to date
    create_dashboard_html()
    
    # Generate data.js with dynamic data for the dashboard
    generate_dashboard_data()
    
    # Create sample files if necessary for demonstration
    create_sample_files_if_needed()
    
    # Update the chart and JSON data
    update_dashboard_data()
    
    # Create symlink from index.html to dashboard.html if it doesn't exist
    if not os.path.exists(INDEX_HTML) and os.path.exists(DASHBOARD_FILE):
        try:
            rel_path = os.path.relpath(DASHBOARD_FILE, os.path.dirname(INDEX_HTML))
            os.symlink(rel_path, INDEX_HTML)
            print(f"Created symlink: {INDEX_HTML} -> {rel_path}")
        except Exception as e:
            print(f"Error creating symlink: {e}")
            # Fallback to copy
            try:
                shutil.copy2(DASHBOARD_FILE, INDEX_HTML)
                print(f"Copied {DASHBOARD_FILE} to {INDEX_HTML} (symlink failed)")
            except Exception as copy_e:
                print(f"Error copying file: {copy_e}")
    
    print(f"Investment dashboard generated at: {DASHBOARD_FILE}")
    
    # Open the dashboard in browser if requested
    if open_browser:
        dashboard_url = f"file://{os.path.abspath(DASHBOARD_FILE)}"
        print(f"Opening dashboard in browser: {dashboard_url}")
        webbrowser.open(dashboard_url)
    
    return DASHBOARD_FILE

def create_sample_files_if_needed():
    """Create sample JSON and CSV files if no data exists."""
    json_files = get_json_files()
    data_files = get_raw_data_files()
    
    # Create sample JSON file if no JSON files found
    if not json_files:
        print("No JSON files found. Creating a sample JSON file...")
        sample_json_path = os.path.join(DASHBOARD_JSON_DIR, "sample_data.json")
        create_sample_json_file(sample_json_path)
    
    # Create sample CSV file if no data files found
    if not data_files:
        print("No data files found. Creating a sample CSV file...")
        sample_csv_path = os.path.join(DATA_DIR, "raw", "sample_data.csv")
        create_sample_csv_file(sample_csv_path)

def update_dashboard_data():
    """Update the dashboard with latest chart and JSON data."""
    # Get image files for chart section
    image_files = get_image_files()
    
    # Get JSON files for reports section
    json_files = get_json_files()
    
    # Get sentiment visualizations
    sentiment_visuals = get_sentiment_model_visualizations()
    
    # Get raw data files
    data_files = get_raw_data_files()
    
    # Get processed data files
    processed_data = get_processed_data()
    
    # Generate chart cards data.js
    charts_data = []
    for img_file in image_files:
        filename = os.path.basename(img_file)
        title = os.path.splitext(filename)[0].replace('_', ' ').title()
        relative_path = os.path.relpath(img_file, DASHBOARD_DIR)
        charts_data.append({
            "src": relative_path,
            "title": title,
            "path": relative_path
        })
    
    # Generate JSON cards data.js
    json_data = []
    for json_file in json_files:
        filename = os.path.basename(json_file)
        title = os.path.splitext(filename)[0].replace('_', ' ').title()
        relative_path = os.path.relpath(json_file, DASHBOARD_DIR)
        json_data.append({
            "path": relative_path,
            "title": title
        })
    
    # Create a charts-data.js file
    charts_js_content = f"""/**
 * Charts Data Module
 * Contains chart data for the dashboard
 */
 
// Chart cards data
window.chartsData = {json.dumps(charts_data, indent=2)};

// JSON reports data
window.jsonReportsData = {json.dumps(json_data, indent=2)};
"""
    
    # Write the charts-data.js file
    charts_js_path = os.path.join(DASHBOARD_JS_DIR, "charts-data.js")
    with open(charts_js_path, 'w') as f:
        f.write(charts_js_content)
    
    print(f"Generated charts data.js: {charts_js_path}")

def debug_tabs(html_content):
    """Ensure that all tab IDs match their onClick handlers in the HTML."""
    # Find all tab buttons
    tab_buttons = re.findall(r'onclick="openTab\(\'(.*?)\'\)"', html_content)
    
    # Find all content section IDs
    content_sections = re.findall(r'id="(home|charts|json|data|processed|expert)"', html_content)
    
    # Compare to make sure all tabs have matching content divs
    all_tabs_matched = True
    for tab in tab_buttons:
        if tab not in content_sections:
            print(f"Warning: Tab '{tab}' does not have a matching content div")
            all_tabs_matched = False
    
    for section in content_sections:
        if section not in tab_buttons:
            print(f"Warning: Content div '{section}' does not have a matching tab button")
            all_tabs_matched = False
    
    if all_tabs_matched:
        print("All tabs match their content divs")

def main():
    """Main function to generate the dashboard and open it in a browser."""
    generate_dashboard(open_browser=True)
    
    return True

if __name__ == "__main__":
    main() 