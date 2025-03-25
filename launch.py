#!/usr/bin/env python3
"""
Simple Python launcher for Investment Recommendation System
This will work on any platform with Python installed
"""

import os
import sys
import subprocess
import platform

def main():
    """Main launcher function"""
    print("=== Investment Recommendation System ===")
    
    # Change to the directory where this script is located
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # Make sure the main script exists
    if not os.path.exists("run_investment_system.py"):
        print("Error: run_investment_system.py not found in the current directory.")
        input("Press Enter to exit.")
        return 1
    
    # Determine the Python executable to use
    if platform.system() == "Windows":
        local_python = os.path.join(".conda", "bin", "python.exe")
    else:
        local_python = os.path.join(".conda", "bin", "python")
    
    python_cmd = local_python if os.path.exists(local_python) else sys.executable
    
    # Print information about the Python being used
    print(f"Using Python: {python_cmd}")
    
    # Run the system
    print("Starting the dashboard...")
    try:
        result = subprocess.run([python_cmd, "run_investment_system.py", "dashboard"], 
                              check=True)
        success = result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"Error running the system: {e}")
        success = False
    except FileNotFoundError:
        print(f"Error: Python executable '{python_cmd}' not found.")
        print("Trying system Python...")
        try:
            result = subprocess.run([sys.executable, "run_investment_system.py", "dashboard"], 
                                  check=True)
            success = result.returncode == 0
        except Exception as e:
            print(f"Error running with system Python: {e}")
            success = False
    
    # Print completion message
    if success:
        print("\nProcess completed successfully!")
    else:
        print("\nProcess completed with errors. Check the output above for details.")
    
    if platform.system() == "Windows":
        input("\nPress Enter to exit.")
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main()) 