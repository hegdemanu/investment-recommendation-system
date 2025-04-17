#!/usr/bin/env python3
"""
Script to check if training report exists and can be opened.
This helps diagnose issues with the report generation.
"""
import os
import json
from pathlib import Path

# Define the expected report locations
RESULTS_DIR = "./results"
REPORT_PATH = os.path.join(RESULTS_DIR, "training", "training_summary.json")
CHART_PATH = os.path.join(RESULTS_DIR, "training", "training_success_rate.png")

def check_report():
    """Check if training report exists and can be opened."""
    print("Checking training report...")
    
    # Check if directories exist
    for dir_path in [RESULTS_DIR, os.path.join(RESULTS_DIR, "training")]:
        path = Path(dir_path)
        if not path.exists():
            print(f"❌ Directory not found: {dir_path}")
            print(f"Creating directory: {dir_path}")
            path.mkdir(parents=True, exist_ok=True)
        else:
            print(f"✅ Directory exists: {dir_path}")
    
    # Check JSON report
    report_path = Path(REPORT_PATH)
    if not report_path.exists():
        print(f"❌ Report file not found: {REPORT_PATH}")
    else:
        print(f"✅ Report file exists: {REPORT_PATH}")
        try:
            # Try to open and parse the JSON file
            with open(report_path, 'r') as f:
                report_data = json.load(f)
            print(f"✅ Report file successfully parsed.")
            print(f"Report contents summary:")
            print(f"- Timestamp: {report_data.get('timestamp', 'N/A')}")
            print(f"- Stocks: {report_data.get('stocks', {}).get('total', 0)} total, "
                  f"{report_data.get('stocks', {}).get('success', 0)} successful")
            print(f"- Mutual Funds: {report_data.get('mutual_funds', {}).get('total', 0)} total, "
                  f"{report_data.get('mutual_funds', {}).get('success', 0)} successful")
        except json.JSONDecodeError:
            print(f"❌ Report file exists but contains invalid JSON.")
            # Print the first few lines to help diagnose
            with open(report_path, 'r') as f:
                first_lines = ''.join([f.readline() for _ in range(5)])
            print(f"First few lines of file:\n{first_lines}")
        except Exception as e:
            print(f"❌ Error reading report file: {str(e)}")
    
    # Check chart file
    chart_path = Path(CHART_PATH)
    if not chart_path.exists():
        print(f"❌ Chart file not found: {CHART_PATH}")
    else:
        print(f"✅ Chart file exists: {CHART_PATH}")
        print(f"Chart file size: {chart_path.stat().st_size} bytes")
    
    # List all files in the training directory
    training_dir = Path(os.path.join(RESULTS_DIR, "training"))
    if training_dir.exists():
        print("\nAll files in training directory:")
        for file_path in training_dir.iterdir():
            print(f"- {file_path.name} ({file_path.stat().st_size} bytes)")

if __name__ == "__main__":
    check_report() 