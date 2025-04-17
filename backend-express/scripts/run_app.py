#!/usr/bin/env python3
"""
Interactive application to generate, manage, and view reports for investment recommendation system.

This script:
1. Discovers all available reports in the repository
2. Provides a menu to select and open specific reports
3. Can regenerate reports on demand
4. Includes an option to open all reports at once

Usage: python run_app.py
"""
import os
import json
import subprocess
import platform
import glob
from pathlib import Path
import webbrowser
import argparse

# Define directory paths
RESULTS_DIR = "./results"
REPORTS_DIRS = {
    "training": os.path.join(RESULTS_DIR, "training"),
    "metrics": os.path.join(RESULTS_DIR, "metrics"),
    "plots": os.path.join(RESULTS_DIR, "plots"),
    "validation": RESULTS_DIR  # For validation_summary.json
}

# Define report types and their extensions
REPORT_TYPES = {
    "html": [".html"],
    "json": [".json"],
    "images": [".png", ".jpg", ".jpeg", ".svg"],
    "csv": [".csv"]
}

def ensure_directories():
    """Ensure all report directories exist."""
    for directory in REPORTS_DIRS.values():
        os.makedirs(directory, exist_ok=True)
    print("✅ Verified all report directories exist")

def find_reports():
    """Find all reports in the repository."""
    reports = {}
    
    for category, directory in REPORTS_DIRS.items():
        if not os.path.exists(directory):
            continue
            
        reports[category] = []
        
        # Find all files in the directory
        for ext_type, extensions in REPORT_TYPES.items():
            for ext in extensions:
                pattern = os.path.join(directory, f"*{ext}")
                matching_files = glob.glob(pattern)
                
                for file_path in matching_files:
                    file_name = os.path.basename(file_path)
                    reports[category].append({
                        "name": file_name,
                        "path": file_path,
                        "type": ext_type,
                        "size": os.path.getsize(file_path),
                        "extension": ext
                    })
    
    return reports

def open_file(file_path):
    """Open a file with the default application."""
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return False
        
    try:
        if platform.system() == 'Darwin':  # macOS
            subprocess.run(['open', file_path], check=True)
        elif platform.system() == 'Windows':
            os.startfile(file_path)
        else:  # Linux or other
            subprocess.run(['xdg-open', file_path], check=True)
        print(f"✅ Opened: {file_path}")
        return True
    except Exception as e:
        print(f"❌ Error opening file: {str(e)}")
        
        # Try using webbrowser module as fallback for HTML files
        if file_path.endswith(('.html', '.htm')):
            try:
                abs_path = os.path.abspath(file_path)
                webbrowser.open('file://' + abs_path)
                print(f"✅ Opened in web browser: {file_path}")
                return True
            except Exception as e2:
                print(f"❌ Error opening in browser: {str(e2)}")
                
        return False

def generate_reports():
    """Run scripts to generate/regenerate reports."""
    scripts = {
        "Training Report": "train_all_models.py",
        "Validation Report": "validate_model.py",
        "Model Training (Timeframes)": "train_models.py",
        "HTML Dashboard": "generate_html_report.py"
    }
    
    print("\n=== Report Generation ===")
    print("Select a script to run for generating reports:")
    
    options = list(scripts.keys())
    for i, option in enumerate(options, 1):
        print(f"{i}. {option}")
    print(f"{len(options) + 1}. Run all scripts")
    print("0. Back to main menu")
    
    choice = input("\nEnter your choice (0-{}): ".format(len(options) + 1))
    
    try:
        choice = int(choice)
        if choice == 0:
            return
        elif 1 <= choice <= len(options):
            script = scripts[options[choice - 1]]
            print(f"\nRunning {script}...")
            subprocess.run(['python', script], check=True)
            print(f"✅ {script} completed")
            
            # After running any report, offer to open the dashboard
            if script != "generate_html_report.py" and os.path.exists("generate_html_report.py"):
                if input("\nWould you like to open the HTML dashboard? (y/n): ").lower() == 'y':
                    print("Generating dashboard...")
                    subprocess.run(['python', 'generate_html_report.py'], check=True)
        elif choice == len(options) + 1:
            for name, script in scripts.items():
                if os.path.exists(script):
                    print(f"\nRunning {script}...")
                    try:
                        subprocess.run(['python', script], check=True)
                        print(f"✅ {script} completed")
                    except subprocess.CalledProcessError:
                        print(f"❌ Error running {script}")
                else:
                    print(f"❌ Script not found: {script}")
        else:
            print("Invalid choice. Please try again.")
    except ValueError:
        print("Please enter a valid number.")

def display_report_menu(reports):
    """Display interactive menu for report selection."""
    while True:
        print("\n=== Investment Recommendation System Reports ===")
        
        # Check if dashboard exists and offer to open it directly
        dashboard_path = os.path.join(RESULTS_DIR, "dashboard", "investment_dashboard.html")
        if os.path.exists(dashboard_path):
            print("\n💡 HTML Dashboard Available!")
            if input("Would you like to open the comprehensive HTML dashboard? (y/n): ").lower() == 'y':
                open_file(dashboard_path)
                if input("\nReturn to menu? (y/n): ").lower() != 'y':
                    return
        elif os.path.exists("generate_html_report.py"):
            print("\n💡 HTML Dashboard can be generated!")
            if input("Would you like to generate and open the comprehensive HTML dashboard? (y/n): ").lower() == 'y':
                subprocess.run(['python', 'generate_html_report.py'], check=True)
                if input("\nReturn to menu? (y/n): ").lower() != 'y':
                    return
        
        # Count total reports
        total_reports = sum(len(reports[category]) for category in reports)
        if total_reports == 0:
            print("No reports found. Generate reports first.")
            if input("Generate reports now? (y/n): ").lower() == 'y':
                generate_reports()
                reports = find_reports()
                continue
            else:
                return
        
        # Display categories
        categories = list(reports.keys())
        valid_categories = [cat for cat in categories if reports[cat]]
        
        print("\nReport Categories:")
        for i, category in enumerate(valid_categories, 1):
            report_count = len(reports[category])
            print(f"{i}. {category.title()} Reports ({report_count})")
        
        print(f"{len(valid_categories) + 1}. Open All Reports")
        print(f"{len(valid_categories) + 2}. Generate/Regenerate Reports")
        print("0. Exit")
        
        choice = input("\nEnter your choice (0-{}): ".format(len(valid_categories) + 2))
        
        try:
            choice = int(choice)
            if choice == 0:
                break
            elif 1 <= choice <= len(valid_categories):
                selected_category = valid_categories[choice - 1]
                display_category_reports(reports[selected_category], selected_category)
            elif choice == len(valid_categories) + 1:
                open_all_reports(reports)
            elif choice == len(valid_categories) + 2:
                generate_reports()
                reports = find_reports()  # Refresh reports list
            else:
                print("Invalid choice. Please try again.")
        except ValueError:
            print("Please enter a valid number.")

def display_category_reports(category_reports, category_name):
    """Display reports for a specific category."""
    while True:
        print(f"\n=== {category_name.title()} Reports ===")
        
        # Group by type for better organization
        grouped_reports = {}
        for report in category_reports:
            report_type = report['type']
            if report_type not in grouped_reports:
                grouped_reports[report_type] = []
            grouped_reports[report_type].append(report)
        
        # Display reports by type
        all_reports = []
        print("")
        for report_type, reports in grouped_reports.items():
            print(f"-- {report_type.upper()} Files --")
            for i, report in enumerate(reports, len(all_reports) + 1):
                size_kb = report['size'] / 1024
                print(f"{i}. {report['name']} ({size_kb:.1f} KB)")
                all_reports.append(report)
            print("")
        
        print(f"{len(all_reports) + 1}. Open All {category_name.title()} Reports")
        print("0. Back to main menu")
        
        choice = input("\nEnter your choice (0-{}): ".format(len(all_reports) + 1))
        
        try:
            choice = int(choice)
            if choice == 0:
                break
            elif 1 <= choice <= len(all_reports):
                selected_report = all_reports[choice - 1]
                open_file(selected_report['path'])
            elif choice == len(all_reports) + 1:
                for report in all_reports:
                    open_file(report['path'])
            else:
                print("Invalid choice. Please try again.")
        except ValueError:
            print("Please enter a valid number.")

def open_all_reports(reports):
    """Open all reports across all categories."""
    print("\nOpening all reports...")
    
    count = 0
    for category, category_reports in reports.items():
        if not category_reports:
            continue
            
        print(f"\nOpening {category.title()} Reports:")
        for report in category_reports:
            if open_file(report['path']):
                count += 1
    
    print(f"\n✅ Opened {count} reports")

def main():
    """Main function to run the application."""
    parser = argparse.ArgumentParser(description='Investment Recommendation System Reports Manager')
    parser.add_argument('--generate', action='store_true', help='Generate reports')
    parser.add_argument('--open-all', action='store_true', help='Open all reports')
    parser.add_argument('--report', help='Open a specific report by name')
    parser.add_argument('--dashboard', action='store_true', help='Generate and open HTML dashboard')
    
    args = parser.parse_args()
    
    # Ensure all directories exist
    ensure_directories()
    
    # Process dashboard request first
    if args.dashboard:
        dashboard_path = os.path.join(RESULTS_DIR, "dashboard", "investment_dashboard.html")
        if os.path.exists(dashboard_path):
            print(f"Opening existing dashboard: {dashboard_path}")
            open_file(dashboard_path)
        elif os.path.exists("generate_html_report.py"):
            print("Generating and opening HTML dashboard...")
            subprocess.run(['python', 'generate_html_report.py'], check=True)
        else:
            print("Dashboard generator not found. Please run with --generate first.")
        return
    
    # Find all reports
    reports = find_reports()
    
    # Process command-line arguments
    if args.generate:
        generate_reports()
        reports = find_reports()  # Refresh reports list
    
    if args.report:
        # Try to find the specified report
        found = False
        for category, category_reports in reports.items():
            for report in category_reports:
                if args.report.lower() in report['name'].lower():
                    open_file(report['path'])
                    found = True
                    break
            if found:
                break
        
        if not found:
            print(f"Report '{args.report}' not found")
            print("Available reports:")
            for category, category_reports in reports.items():
                for report in category_reports:
                    print(f"  - {report['name']}")
    
    if args.open_all:
        open_all_reports(reports)
    
    # If no specific action arguments provided, show interactive menu
    if not (args.open_all or args.report or args.generate or args.dashboard):
        display_report_menu(reports)

if __name__ == "__main__":
    main() 