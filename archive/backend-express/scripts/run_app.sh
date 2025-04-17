#!/bin/bash
# Investment Recommendation System Launcher
# Just click this file to run the entire application

# Set to exit immediately if any command fails
set -e

# Display colorful output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function for formatted output
print_header() {
    echo -e "\n${BLUE}======================================${NC}"
    echo -e "${BLUE}    $1${NC}"
    echo -e "${BLUE}======================================${NC}\n"
}

# Check if Python is installed
if ! command -v python &> /dev/null; then
    echo -e "${RED}Error: Python is not installed or not in PATH${NC}"
    echo "Please install Python 3.6+ and try again"
    exit 1
fi

# Create necessary directories
mkdir -p results/reports logs

# Print welcome message
clear
print_header "INVESTMENT RECOMMENDATION SYSTEM"
echo -e "${YELLOW}Starting application...${NC}"
echo -e "Current directory: $(pwd)"
echo -e "Started at: $(date)"
echo ""

# Check for conda/virtual environment
if [ -f "environment.yml" ]; then
    print_header "CHECKING CONDA ENVIRONMENT"
    if command -v conda &> /dev/null; then
        echo "Conda found, activating environment..."
        # Create environment if it doesn't exist
        if ! conda env list | grep -q "investment-env"; then
            echo "Creating conda environment from environment.yml..."
            conda env create -f environment.yml
        fi
        # Activate the environment
        source "$(conda info --base)/etc/profile.d/conda.sh"
        conda activate investment-env || echo -e "${YELLOW}Warning: Could not activate conda environment${NC}"
    else
        echo -e "${YELLOW}Warning: Conda not found. Using system Python.${NC}"
    fi
elif [ -f "requirements.txt" ]; then
    print_header "CHECKING PYTHON DEPENDENCIES"
    echo "Installing required packages..."
    pip install -r requirements.txt
fi

# Run the application
print_header "RUNNING INVESTMENT RECOMMENDATION SYSTEM"

# Run training if scripts exist
if [ -f "train_stock_models.py" ]; then
    echo -e "${GREEN}Training stock models...${NC}"
    python train_stock_models.py
fi

if [ -f "train_mutual_fund_models.py" ]; then
    echo -e "${GREEN}Training mutual fund models...${NC}"
    python train_mutual_fund_models.py
fi

# Generate the investment report
echo -e "${GREEN}Generating investment report...${NC}"
python generate_investment_report.py

# Open the report in the browser
if [ -f "results/reports/investment_report.html" ]; then
    print_header "OPENING REPORT"
    echo "Opening report in your default browser..."
    
    # Different commands based on OS
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        open results/reports/investment_report.html
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Linux
        xdg-open results/reports/investment_report.html
    else
        # Windows or others
        python -m webbrowser results/reports/investment_report.html
    fi
else
    echo -e "${RED}Error: Report file not found${NC}"
fi

# Complete
print_header "PROCESS COMPLETE"
echo -e "${GREEN}Investment Recommendation System has completed successfully!${NC}"
echo -e "Finished at: $(date)"
echo ""
echo -e "Press any key to exit..."
read -n 1 -s

exit 0 