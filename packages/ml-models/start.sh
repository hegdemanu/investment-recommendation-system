#!/bin/bash

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Install dependencies if requirements.txt exists
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
fi

# Start the FastAPI server using uvicorn
uvicorn src.api.app:app --host 0.0.0.0 --port 5001 --reload 