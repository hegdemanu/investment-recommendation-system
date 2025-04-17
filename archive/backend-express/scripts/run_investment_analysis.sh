#!/bin/bash

# Investment Recommendation System - Complete Analysis Pipeline
# This script runs the entire pipeline from model validation to prediction and report generation

echo "========= Investment Recommendation System ========="
echo "Starting comprehensive analysis pipeline..."
echo

# Step 1: Validate models using training/testing split
echo "Step 1/3: Validating models with training/testing data split"
python validate_model.py
echo

# Step 2: Generate predictions for the next 30 days
echo "Step 2/3: Generating predictions for all stocks"
echo "Using 100% LSTM model for first 15 days, then weighted ensemble after"
python make_predictions.py
echo

# Step 3: Create comprehensive investment report
echo "Step 3/3: Creating investment recommendation report"
python generate_report.py
echo

echo "Analysis pipeline complete!"
echo "Results available in:"
echo "- Model validation: results/validation_summary.json"
echo "- Predictions: results/predictions/"
echo "- Investment report: results/analysis/investment_report.html"
echo
echo "Note: Stock predictions now use 100% LSTM for first 15 days, then"
echo "transition to a weighted ensemble similar to mutual funds."
echo
echo "To view the HTML report, open results/analysis/investment_report.html in a web browser."
echo "========= Analysis Complete =========" 