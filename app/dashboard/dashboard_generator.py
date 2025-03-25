"""
Dashboard Generator Module

This module generates the HTML dashboard that integrates all reports, visualizations,
and data into a single interactive interface.
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

from config.settings import (
    DASHBOARD_FILE, DASHBOARD_DIR, DASHBOARD_JSON_DIR, TRAINING_DIR, VALIDATION_DIR, 
    REPORTS_DIR, MODELS_DIR, RESULTS_DIR, DATA_DIR
)
from app.utils.file_utils import file_to_base64, load_json_file
from app.utils.sample_generator import create_sample_json_file, create_sample_csv_file

# Import the template
from app.dashboard.dashboard_template import get_dashboard_template

def ensure_directories():
    """Ensure all necessary directories exist."""
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
    """Generate the JavaScript data structure for the sector pie chart."""
    sector_data = get_sector_data()
    
    # Generate the JavaScript array for the pie chart
    js_data = []
    for sector, data in sector_data.items():
        js_data.append(f'{{ label: "{sector}", value: {data["value"]}, color: "{data["color"]}" }}')
    
    return ",\n                ".join(js_data)

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

def generate_dashboard():
    """Generate the main dashboard HTML file."""
    ensure_directories()
    
    # Collect all image files
    image_files = get_image_files()
    
    # Collect all JSON report files
    json_files = get_json_files()
    
    # Collect all raw data files
    data_files = get_raw_data_files()
    
    # Collect all processed data
    processed_data = get_processed_data()
    
    # If no files were found, create sample files
    if not image_files and not json_files and not data_files:
        print("No files found. Creating sample files...")
        create_sample_json_file()
        create_sample_csv_file()
        
        # Refresh file lists
        image_files = get_image_files()
        json_files = get_json_files()
        data_files = get_raw_data_files()
    
    # Check for validation summary
    validation_summary = os.path.join(RESULTS_DIR, "validation_summary.json")
    if os.path.exists(validation_summary):
        # Add validation summary to the JSON files if it's not already there
        if validation_summary not in json_files:
            json_files.append(validation_summary)
    
    # Print a summary of what was found
    print(f"Total files found: {len(image_files)} images, {len(json_files)} JSONs, {len(data_files)} data files")
    
    # Generate HTML content
    chart_items = generate_chart_items(image_files)
    json_items = generate_json_items(json_files)
    raw_data_items = generate_raw_data_items(data_files)
    processed_data_warning, processed_data_items = generate_processed_data_items(processed_data)
    sector_pie_data = generate_sector_pie_data()
    
    # Get template and insert dynamic content
    template = get_dashboard_template()
    html_content = template.format(
        chart_items=chart_items,
        json_items=json_items,
        raw_data_items=raw_data_items,
        processed_data_warning=processed_data_warning,
        processed_data_items=processed_data_items,
        date=datetime.now().strftime("%Y-%m-%d"),
        sector_pie_data=sector_pie_data
    )
    
    # Write the HTML file
    with open(DASHBOARD_FILE, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    # Debug tabs to make sure they match their content divs
    debug_tabs(html_content)
    
    return DASHBOARD_FILE

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
    dashboard_file = generate_dashboard()
    
    if dashboard_file:
        # Open the dashboard in a web browser
        webbrowser.open('file://' + os.path.abspath(dashboard_file))
        return True
    
    return False

if __name__ == "__main__":
    main() 