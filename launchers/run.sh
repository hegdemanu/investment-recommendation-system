#!/bin/bash
# Simple launcher script that uses the local conda environment

# Change to the directory where this script is located
cd "$(dirname "$0")"

echo "=== Investment Recommendation System ==="

# Use the local conda Python
PYTHON_PATH=".conda/bin/python"

# Check if the Python executable exists
if [ ! -f "$PYTHON_PATH" ]; then
    echo "Error: Local Python not found at $PYTHON_PATH"
    echo "Press Enter to exit."
    read
    exit 1
fi

# Check if run_investment_system.py exists
if [ ! -f "run_investment_system.py" ]; then
    echo "Error: run_investment_system.py not found in the current directory."
    echo "Press Enter to exit."
    read
    exit 1
fi

# Run the system
echo "Starting the system..."
"$PYTHON_PATH" run_investment_system.py dashboard

echo "Press Enter to exit."
read 