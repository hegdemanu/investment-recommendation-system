#!/bin/bash

# Investment Recommendation System Setup Script
# This script prepares the environment for running the investment recommendation system

echo "======================================================"
echo "Investment Recommendation System - Setup Script"
echo "======================================================"

# Check if Python 3.8+ is installed
python_version=$(python3 --version 2>&1 | awk '{print $2}')
python_major=$(echo $python_version | cut -d. -f1)
python_minor=$(echo $python_version | cut -d. -f2)

if [ -z "$python_version" ] || [ "$python_major" -lt 3 ] || ([ "$python_major" -eq 3 ] && [ "$python_minor" -lt 8 ]); then
    echo "Error: Python 3.8 or higher is required"
    echo "Current version: $python_version"
    echo "Please install Python 3.8+ and try again"
    exit 1
fi

echo "✅ Python $python_version detected"

# Create and activate virtual environment
echo ""
echo "Creating virtual environment..."

# Check if virtualenv is installed
if ! command -v pip3 &> /dev/null; then
    echo "Error: pip3 is not installed"
    exit 1
fi

# Create virtual environment directory
if [ ! -d "venv" ]; then
    python3 -m venv venv
    if [ $? -ne 0 ]; then
        echo "Error: Failed to create virtual environment"
        exit 1
    fi
    echo "✅ Virtual environment created"
else
    echo "✅ Virtual environment already exists"
fi

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source venv/bin/activate
if [ $? -ne 0 ]; then
    echo "Error: Failed to activate virtual environment"
    exit 1
fi
echo "✅ Virtual environment activated"

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "Error: Failed to install dependencies"
    exit 1
fi
echo "✅ Dependencies installed"

# Create required directories
echo ""
echo "Creating required directories..."
mkdir -p data
mkdir -p results

echo "✅ Directories created"

# Set up environment variables
echo ""
echo "Setting up environment variables..."

if [ ! -f ".env" ]; then
    echo "Creating .env file..."
    cat > .env << EOF
# NewsAPI key for sentiment analysis
# Get your key at https://newsapi.org/
NEWSAPI_KEY=your_newsapi_key_here

# Default settings
DEFAULT_FORECAST_DAYS=30
DEFAULT_HISTORY_YEARS=5
DEFAULT_NEWS_DAYS=7
EOF
    echo "✅ .env file created (please edit with your API keys)"
else
    echo "✅ .env file already exists"
fi

# Check for optional dependencies
echo ""
echo "Checking for optional dependencies..."

# Check for Prophet (can be difficult to install on some systems)
if python -c "import prophet" &> /dev/null; then
    echo "✅ Prophet is installed"
else
    echo "⚠️  Prophet is not installed"
    echo "  The Prophet package is required for forecasting features"
    echo "  If installation fails, try:"
    echo "  pip install prophet"
fi

# Check for transformers and torch (for sentiment analysis)
if python -c "import transformers" &> /dev/null; then
    echo "✅ Transformers is installed"
else
    echo "⚠️  Transformers is not installed"
    echo "  The transformers package is required for sentiment analysis features"
fi

if python -c "import torch" &> /dev/null; then
    echo "✅ PyTorch is installed"
else
    echo "⚠️  PyTorch is not installed"
    echo "  The torch package is required for sentiment analysis features"
fi

# Download the FinBERT model to avoid first-run delays
echo ""
echo "Downloading FinBERT model (this may take a few minutes)..."
python -c "from transformers import AutoTokenizer, AutoModelForSequenceClassification; tokenizer = AutoTokenizer.from_pretrained('ProsusAI/finbert'); model = AutoModelForSequenceClassification.from_pretrained('ProsusAI/finbert')" &> /dev/null
if [ $? -eq 0 ]; then
    echo "✅ FinBERT model downloaded"
else
    echo "⚠️  Failed to download FinBERT model"
    echo "  The model will be downloaded when first running sentiment analysis"
fi

# Set up permissions
echo ""
echo "Setting up permissions..."
chmod +x run_analysis.py
chmod +x trading_engine/*.py
echo "✅ Permissions set"

# Done
echo ""
echo "======================================================"
echo "🎉 Setup complete! 🎉"
echo "======================================================"
echo ""
echo "To use the system, run:"
echo "  source venv/bin/activate"
echo "  python run_analysis.py analyze --tickers AAPL --api_key YOUR_NEWSAPI_KEY"
echo ""
echo "For more information, see README.md"
echo ""

# Deactivate virtual environment
deactivate 