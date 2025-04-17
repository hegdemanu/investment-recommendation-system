#!/usr/bin/env python3
"""
Simple script to test opening a report file directly.
This script attempts to open report files using multiple methods.
"""
import os
import subprocess
import platform
import webbrowser
import time

def open_with_subprocess(file_path):
    """Try to open file using subprocess."""
    print(f"Trying to open {file_path} with subprocess...")
    
    try:
        if platform.system() == 'Darwin':  # macOS
            subprocess.run(['open', file_path], check=True)
            return True
        elif platform.system() == 'Windows':
            os.startfile(file_path)
            return True
        else:  # Linux
            subprocess.run(['xdg-open', file_path], check=True)
            return True
    except Exception as e:
        print(f"Error with subprocess: {e}")
        return False

def open_with_webbrowser(file_path):
    """Try to open file using webbrowser module."""
    print(f"Trying to open {file_path} with webbrowser module...")
    
    try:
        abs_path = os.path.abspath(file_path)
        file_url = f'file://{abs_path}'
        print(f"Opening URL: {file_url}")
        webbrowser.open(file_url)
        return True
    except Exception as e:
        print(f"Error with webbrowser: {e}")
        return False

def main():
    """Test opening report files."""
    # List of report files to try
    report_files = [
        "./results/training/training_summary.json",
        "./results/validation_summary.json",
        "./results/training/training_success_rate.png"
    ]
    
    for file_path in report_files:
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            continue
            
        print(f"\nFile found: {file_path} ({os.path.getsize(file_path)} bytes)")
        
        # Try opening with subprocess
        if open_with_subprocess(file_path):
            print("Successfully opened with subprocess")
        else:
            print("Failed to open with subprocess")
            
        # Wait a moment before trying another method
        time.sleep(2)
        
        # Try opening with webbrowser module
        if open_with_webbrowser(file_path):
            print("Successfully opened with webbrowser module")
        else:
            print("Failed to open with webbrowser module")
            
        print("\nMoving to next file...\n")
        time.sleep(2)
    
    print("Test complete")

if __name__ == "__main__":
    main() 