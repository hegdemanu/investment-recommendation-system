# Getting Started

This guide will help you get started with the Investment Recommendation System.

## Prerequisites

- Python 3.9+
- Conda or venv for environment management
- Git for cloning the repository

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd investment-recommendation-system
```

2. Create and activate a conda environment:
```bash
conda create -n investment-env python=3.9
conda activate investment-env
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### Running the Full Analysis Pipeline

The easiest way to get started is to run the complete analysis pipeline:

```bash
./run_investment_analysis.sh
```

This script will:
1. Validate models with training/testing split
2. Generate predictions for all stocks
3. Create a comprehensive investment report

After running, you can view the results in:
- `results/validation_summary.json`: Model validation metrics
- `results/predictions/`: Stock price predictions and plots
- `results/analysis/investment_report.html`: Comprehensive HTML report

### Using the Web Application

To start the web application:

```bash
python app.py
```

Then navigate to `http://localhost:5003` in your browser to access the web interface.

## Working with Individual Scripts

### Model Validation

To validate model performance on historical data:

```bash
python validate_model.py
```

### Generating Predictions

To generate predictions for the next 30 days:

```bash
python make_predictions.py
```

### Creating Reports

To create a comprehensive investment report:

```bash
python generate_report.py
```

## Using the API

If the Flask application is running, you can use the following API endpoints:

- `GET /api/health`: Check server status
- `POST /api/train-models`: Train prediction models
- `POST /api/multi-timeframe-predictions`: Generate predictions for multiple timeframes
- `POST /api/backtracking-analysis`: Run backtracking analysis
- `POST /api/risk-based-recommendations`: Get risk-based investment recommendations
- `POST /api/peg-analysis`: Perform PEG ratio analysis

Example API call:
```bash
curl -X POST http://localhost:5003/api/risk-based-recommendations \
  -H "Content-Type: application/json" \
  -d '{"ticker": "ADNA", "risk_appetite": "moderate"}'
```

## Using Custom Data

To use your own stock data:

1. Format your CSV files with the required columns: Date, Price, Open, High, Low, Vol., Change %
2. Save the files in the `data/stocks/` directory with naming format `TICKER_Sorted.csv`
3. Run the analysis pipeline or individual scripts as needed

## Next Steps

- Check the [Project Summary](SUMMARY.md) for an overview of the system capabilities
- Explore the source code in the `src/` directory for more details
- Modify the risk parameters in `src/risk_analyzer.py` to customize recommendations 