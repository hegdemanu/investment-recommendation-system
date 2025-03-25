#!/usr/bin/env python3
"""
Investment Recommendation System Launcher
Just double-click this file to run the entire application
"""

import os
import sys
import time
import subprocess
import webbrowser
from datetime import datetime

def print_colored(text, color):
    """Print colored text"""
    colors = {
        'green': '\033[92m',
        'blue': '\033[94m',
        'red': '\033[91m',
        'yellow': '\033[93m',
        'end': '\033[0m'
    }
    print(f"{colors.get(color, '')}{text}{colors['end']}")

def print_header(message):
    """Print a formatted header message"""
    border = "=" * (len(message) + 4)
    print(f"\n{border}")
    print_colored(f"| {message} |", 'blue')
    print(f"{border}\n")

def run_command(command, description):
    """Run a command with proper output formatting"""
    print_header(description)
    print(f"Running: {command}")
    start_time = time.time()
    
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        
        execution_time = time.time() - start_time
        
        if result.returncode == 0:
            print_colored(f"✅ Success! (Completed in {execution_time:.2f}s)", 'green')
            print(result.stdout)
            return True
        else:
            print_colored(f"❌ Error! (After {execution_time:.2f}s)", 'red')
            print(f"Error details: {result.stderr}")
            return False
    except Exception as e:
        print_colored(f"❌ Error executing command: {e}", 'red')
        return False

def main():
    """Run the complete investment recommendation system pipeline"""
    # Create necessary directories
    os.makedirs("results/reports", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    print_header("INVESTMENT RECOMMENDATION SYSTEM")
    print_colored(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 'yellow')
    print(f"Working directory: {os.getcwd()}")
    
    # Install requirements if they exist
    if os.path.exists("requirements.txt"):
        print_header("CHECKING DEPENDENCIES")
        try:
            import importlib
            
            # Try to import some common required packages
            required_packages = ["pandas", "numpy", "matplotlib"]
            missing_packages = []
            
            for package in required_packages:
                try:
                    importlib.import_module(package)
                except ImportError:
                    missing_packages.append(package)
            
            if missing_packages:
                print_colored("Some required packages are missing. Installing now...", 'yellow')
                run_command("pip install -r requirements.txt", "INSTALLING DEPENDENCIES")
            else:
                print_colored("All required packages are installed.", 'green')
        except Exception as e:
            print_colored(f"Error checking dependencies: {e}", 'red')
            
    # Run training if scripts exist
    if os.path.exists("train_stock_models.py"):
        run_command("python train_stock_models.py", "TRAINING STOCK MODELS")
    
    if os.path.exists("train_mutual_fund_models.py"):
        run_command("python train_mutual_fund_models.py", "TRAINING MUTUAL FUND MODELS")
    
    # Generate the investment report
    success = run_command("python generate_investment_report.py", "GENERATING INVESTMENT REPORT")
    
    if success:
        # Open the report in a browser
        report_path = os.path.join("results", "reports", "investment_report.html")
        if os.path.exists(report_path):
            print_header("OPENING REPORT")
            print("Opening report in your default browser...")
            
            # Convert to absolute file path for the browser
            abs_path = os.path.abspath(report_path)
            webbrowser.open(f'file://{abs_path}')
        else:
            print_colored(f"Error: Report file not found at {report_path}", 'red')
    
    print_header("PROCESS COMPLETE")
    print_colored(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 'green')
    
    # Keep console open on Windows
    if os.name == 'nt':
        print("\nPress Enter to exit...")
        input()

if __name__ == "__main__":
    main() 