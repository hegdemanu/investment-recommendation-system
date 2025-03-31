#!/bin/bash
# Investment Recommendation System Launcher
# For macOS and Linux systems

echo "Starting Investment Recommendation System..."
echo "============================================"

# Change to the directory where this script is located
cd "$(dirname "$0")"

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Python 3 is not installed. Please install Python 3 and try again."
    exit 1
fi

# Run the full analysis
python3 run_investment_system.py full-analysis

echo "============================================"
echo "Process complete. Press Enter to exit."
read 