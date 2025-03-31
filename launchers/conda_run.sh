#!/bin/bash
# Conda environment launcher for Investment Recommendation System

# Change to the directory where this script is located
cd "$(dirname "$0")"

echo "=== Investment Recommendation System ==="
echo "Activating conda environment..."

# Check if conda exists
if [ -f ".conda/bin/conda" ]; then
    CONDA_PATH=".conda/bin/conda"
elif command -v conda &> /dev/null; then
    CONDA_PATH="conda"
else
    echo "Error: conda not found. Make sure conda is installed."
    echo "Press Enter to exit."
    read
    exit 1
fi

# Run directly with the conda python
PYTHON_PATH=".conda/bin/python"
if [ -f "$PYTHON_PATH" ]; then
    echo "Using local conda Python..."
    "$PYTHON_PATH" run_investment_system.py dashboard
else
    echo "Trying to activate conda environment..."
    # Try to activate the environment
    source .conda/bin/activate &> /dev/null || source "$(conda info --base)/etc/profile.d/conda.sh" &> /dev/null
    
    # Run with the activated environment
    python run_investment_system.py dashboard
fi

echo "Press Enter to exit."
read 