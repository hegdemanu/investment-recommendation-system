#!/bin/bash
# Investment Recommendation System Launcher for macOS
# Just double-click this file to run the application

# Change to the directory where this script is located
cd "$(dirname "$0")"

# Make sure we're in the right directory
echo "Working directory: $(pwd)"

# Print welcome message
echo ""
echo "====================================="
echo "  INVESTMENT RECOMMENDATION SYSTEM   "
echo "====================================="
echo ""
echo "Starting application..."
echo "$(date)"
echo ""

# Run the report generator directly
echo "Generating investment report..."
python generate_investment_report.py

# Generate expert analysis if available
if [ -f "generate_expert_analysis.py" ]; then
    echo ""
    echo "Generating expert analysis..."
    python generate_expert_analysis.py
fi

# Open the reports in browser
echo ""
echo "Opening reports in browser..."
open ./results/reports/investment_report.html

# Open expert analysis summary if it exists
if [ -f "./results/reports/expert/expert_reports_summary.html" ]; then
    echo "Opening expert analysis summary..."
    open ./results/reports/expert/expert_reports_summary.html
fi

echo ""
echo "Process complete! $(date)"
echo ""
# Keep the terminal window open so user can see the output
read -p "Press Enter to exit..." 