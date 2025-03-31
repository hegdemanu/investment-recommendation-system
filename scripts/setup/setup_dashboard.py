#!/usr/bin/env python3
"""
Setup Modular Dashboard Script

This script archives the old monolithic dashboard HTML files and sets up the new modular dashboard.
"""
import os
import shutil
import datetime
from pathlib import Path

# Define paths
ROOT_DIR = Path(__file__).parent
DASHBOARD_DIR = ROOT_DIR / "results" / "dashboard"
ARCHIVE_DIR = ROOT_DIR / "archive" / "dashboard"
OLD_DASHBOARD_FILES = [
    "investment_dashboard.html",
    "investment_dashboard_backup.html",
    "index.html"
]
DASHBOARD_NEW_FILE = DASHBOARD_DIR / "dashboard.html"
DASHBOARD_INDEX_FILE = DASHBOARD_DIR / "index.html"

def archive_old_dashboard():
    """Archive old dashboard files with timestamp."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print(f"Creating archive directory: {ARCHIVE_DIR}")
    os.makedirs(ARCHIVE_DIR, exist_ok=True)
    
    for file in OLD_DASHBOARD_FILES:
        old_file = DASHBOARD_DIR / file
        if old_file.exists():
            archived_file = ARCHIVE_DIR / f"{file.replace('.html', '')}_{timestamp}.html"
            print(f"Archiving {old_file} to {archived_file}")
            shutil.copy2(old_file, archived_file)
            print(f"Removing {old_file}")
            old_file.unlink()

def setup_new_dashboard():
    """Set up the new modular dashboard."""
    if not DASHBOARD_NEW_FILE.exists():
        print(f"Error: New dashboard file not found: {DASHBOARD_NEW_FILE}")
        return False
    
    # Create symlink from index.html to dashboard.html
    print(f"Creating symlink: {DASHBOARD_INDEX_FILE} -> {DASHBOARD_NEW_FILE.name}")
    if DASHBOARD_INDEX_FILE.exists():
        DASHBOARD_INDEX_FILE.unlink()
    DASHBOARD_INDEX_FILE.symlink_to(DASHBOARD_NEW_FILE.name)
    
    # Ensure all necessary directories exist
    css_dir = DASHBOARD_DIR / "css"
    js_dir = DASHBOARD_DIR / "js"
    js_modules_dir = js_dir / "modules"
    
    os.makedirs(css_dir, exist_ok=True)
    os.makedirs(js_dir, exist_ok=True)
    os.makedirs(js_modules_dir, exist_ok=True)
    
    print("New modular dashboard has been set up successfully!")
    return True

def main():
    """Main function to run the setup."""
    print("Starting dashboard modernization...")
    archive_old_dashboard()
    if setup_new_dashboard():
        print("Dashboard modernization completed successfully!")
    else:
        print("Dashboard modernization completed with errors. Please check the logs.")

if __name__ == "__main__":
    main() 