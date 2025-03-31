#!/bin/bash

# Shell script to analyze multiple stocks using archived stock data

# Create results directory if it doesn't exist
mkdir -p results

# List of stocks to analyze 
# We're using the stocks found in data_pipeline/stocks directory
STOCKS=("ADNA" "Apollo" "asian_paints" "bajaj_fin" "icici" "infosys" "jsw_steel" "nestle" "ongc" "titan" "ultratech" "wipro")

echo "======================================================"
echo "Starting Stock Analysis using Archived Data"
echo "======================================================"

# Loop through each stock and run the analysis
for stock in "${STOCKS[@]}"
do
  echo ""
  echo "Analyzing $stock..."
  python trading_engine/analyze_archived_stocks.py --ticker $stock --days 30
  echo "----------------------------------------------------"
done

# Also run analysis for Tata Motors and M&M using their NSE symbols
echo ""
echo "Analyzing additional Indian stocks..."
python trading_engine/analyze_archived_stocks.py --ticker TATAMOTORS.NS --days 30
python trading_engine/analyze_archived_stocks.py --ticker "M&M.BO" --days 30

# Generate summary of all recommendations
echo "======================================================"
echo "Generating summary of all recommendations"
echo "======================================================"

# Create summary file
SUMMARY_FILE="results/stock_recommendations_summary.txt"
echo "Stock Recommendations Summary ($(date))" > $SUMMARY_FILE
echo "=====================================================" >> $SUMMARY_FILE
echo "" >> $SUMMARY_FILE

# Combine all individual recommendation files into a summary
for file in results/*_recommendation.txt; do
  ticker=$(basename "$file" _recommendation.txt)
  echo "TICKER: $ticker" >> $SUMMARY_FILE
  grep "Expected Movement\|Trading Recommendation" "$file" >> $SUMMARY_FILE
  echo "" >> $SUMMARY_FILE
done

echo "Analysis complete! Summary saved to $SUMMARY_FILE"
echo "All results and visualizations saved to the results directory" 