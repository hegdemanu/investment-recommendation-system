#!/usr/bin/env python3
"""
Simple Python launcher for Investment Recommendation System
This will work on any platform with Python installed
"""

import os
import sys
import subprocess
import platform
import argparse

def main():
    """Main launcher function"""
    print("=== Investment Recommendation System ===")
    
    # Change to the directory where this script is located
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Launch Investment Recommendation System")
    parser.add_argument("--mode", choices=["dashboard", "web", "sample"], 
                      default="dashboard", help="Mode to run (dashboard, web server, or generate samples)")
    parser.add_argument("--port", type=int, default=8000, help="Port to run web server on")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to run web server on")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")
    args = parser.parse_args()
    
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
    
    # Prepare command based on mode
    if args.mode == "dashboard":
        print("Starting the dashboard...")
        cmd = [python_cmd, "run_investment_system.py", "dashboard"]
    elif args.mode == "web":
        print(f"Starting the web server on http://{args.host}:{args.port}/")
        cmd = [
            python_cmd, "run_investment_system.py", "web",
            "--port", str(args.port),
            "--host", args.host
        ]
        if args.debug:
            cmd.append("--debug")
    elif args.mode == "sample":
        print("Generating sample data...")
        cmd = [python_cmd, "run_investment_system.py", "sample"]
    else:
        print(f"Unknown mode: {args.mode}")
        return 1
    
    # Run the system
    try:
        result = subprocess.run(cmd, check=True)
        success = result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"Error running the system: {e}")
        success = False
    except FileNotFoundError:
        print(f"Error: Python executable '{python_cmd}' not found.")
        print("Trying system Python...")
        try:
            # Replace python command with system python
            cmd[0] = sys.executable
            result = subprocess.run(cmd, check=True)
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