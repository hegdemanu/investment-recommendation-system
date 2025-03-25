#!/bin/bash
# Investment Recommendation System Launcher
# For macOS - user-friendly double-clickable from Finder

# Set terminal title
echo -e "\033]0;Investment Recommendation System\007"

# Clear screen
clear

# Display colorful banner
echo -e "\033[1;36m"
echo "╔═══════════════════════════════════════════════════════════╗"
echo "║                                                           ║"
echo "║             INVESTMENT RECOMMENDATION SYSTEM              ║"
echo "║                                                           ║"
echo "╚═══════════════════════════════════════════════════════════╝"
echo -e "\033[0m"
echo

# Change to the directory where this script is located
cd "$(dirname "$0")"

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo -e "\033[1;31mPython 3 is not installed!\033[0m"
    echo -e "Please install Python 3 and try again."
    echo
    echo "Press Enter to exit."
    read
    exit 1
fi

# Check if the run_investment_system.py script exists
if [ ! -f "run_investment_system.py" ]; then
    echo -e "\033[1;31mError: run_investment_system.py not found!\033[0m"
    echo -e "Make sure you're running this launcher from the correct directory."
    echo
    echo "Press Enter to exit."
    read
    exit 1
fi

# Run the full analysis
echo -e "\033[1;32mStarting full analysis...\033[0m"
echo
echo -e "\033[1;33mThis may take a few moments. Please be patient.\033[0m"
echo

python3 run_investment_system.py full-analysis

# Check exit status
if [ $? -eq 0 ]; then
    echo
    echo -e "\033[1;32m╔═══════════════════════════════════════════════════════════╗"
    echo "║                                                           ║"
    echo "║                    PROCESS COMPLETE!                      ║"
    echo "║                                                           ║"
    echo "╚═══════════════════════════════════════════════════════════╝\033[0m"
else
    echo
    echo -e "\033[1;31m╔═══════════════════════════════════════════════════════════╗"
    echo "║                                                           ║"
    echo "║       AN ERROR OCCURRED DURING EXECUTION. CHECK LOGS.     ║"
    echo "║                                                           ║"
    echo "╚═══════════════════════════════════════════════════════════╝\033[0m"
fi

echo
echo -e "\033[1mPress Enter to close this window.\033[0m"
read 