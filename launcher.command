#!/bin/bash
# Investment Recommendation System Launcher
# For macOS - double-clickable from Finder

# Set terminal title
echo -e "\033]0;Investment Recommendation System\007"

# Display banner
echo "╔════════════════════════════════════════════════╗"
echo "║       Investment Recommendation System         ║"
echo "╚════════════════════════════════════════════════╝"
echo

# Change to the directory where this script is located
cd "$(dirname "$0")"

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Python 3 is not installed. Please install Python 3 and try again."
    echo
    echo "Press Enter to exit."
    read
    exit 1
fi

# Run the full analysis
echo "Starting full analysis..."
echo
python3 run_investment_system.py full-analysis

echo
echo "╔════════════════════════════════════════════════╗"
echo "║                Process complete                ║"
echo "╚════════════════════════════════════════════════╝"
echo
echo "Press Enter to close this window."
read 