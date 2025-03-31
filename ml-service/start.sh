#!/bin/bash

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
else
    echo "No virtual environment found. Consider creating one with:"
    echo "python -m venv venv"
    echo "source venv/bin/activate"
fi

# Install dependencies if requirements.txt exists
if [ -f "requirements.txt" ]; then
    echo "Installing dependencies..."
    pip install -r requirements.txt
else
    echo "No requirements.txt found. Installing minimal dependencies..."
    pip install fastapi uvicorn numpy pandas scikit-learn tensorflow prophet arch yfinance ta torch transformers
fi

# Start the FastAPI server
echo "Starting ML service..."
uvicorn src.api.app:app --host 0.0.0.0 --port 5001 --reload 