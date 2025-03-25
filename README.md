# Investment Recommendation System

A comprehensive system for analyzing stock and mutual fund performance, training prediction models, and generating investment recommendations.

## How to Run the Application

### Quick Start (Choose one option)

#### Option 1: Python Script (Works on All Platforms)
Simply double-click on `run_app.py` or run from terminal:
```bash
python run_app.py
```

#### Option 2: Windows Users
Double-click on `run_app.bat` to run the application.

#### Option 3: macOS/Linux Users
1. Make sure the script is executable:
   ```bash
   chmod +x run_app.sh
   ```
2. Double-click on `run_app.sh` or run from terminal:
   ```bash
   ./run_app.sh
   ```

## System Components

The Investment Recommendation System includes the following components:

1. **Data Collection**: Historical stock and mutual fund data acquisition
2. **Model Training**: LSTM-based prediction models for price forecasting
3. **Performance Analysis**: Evaluation of model accuracy and training efficiency
4. **Report Generation**: Comprehensive HTML reports with visualizations
5. **Investment Recommendations**: Data-driven suggestions for optimal investments

## System Requirements

- Python 3.6 or higher
- Required Python packages:
  - pandas
  - numpy
  - matplotlib
  - seaborn
  - tensorflow (for LSTM models)

## Visualizations

The system generates the following visualizations:
- Training performance metrics
- Prediction accuracy comparison
- Training size vs. accuracy relationship
- Stock and mutual fund performance trends

## Output

After running the application, you'll find:
- Trained models in the `models/` directory
- Training summaries in `results/training/`
- HTML reports in `results/reports/`
- Log files in `logs/`

The main output is an interactive HTML report that opens automatically after the process completes.

---

© 2023 Investment Recommendation System

## Features

- **LSTM Model Training**: Trains deep learning models to predict future stock prices using historical data
- **Hybrid Model Weighting**: Uses 100% LSTM for first 15 days of stock predictions, then transitions to a weighted ensemble
- **Sliding Window Approach**: Uses a 6-month sliding window with 1-day steps for optimal prediction horizons
- **Multiple Time Frame Analysis**: Provides short-term, medium-term, and long-term predictions
- **Backtracking Analysis**: Evaluates model performance on historical data
- **Risk-Based Recommendations**: Tailors investment suggestions based on risk appetite
- **PEG Ratio Analysis**: Incorporates Price/Earnings to Growth ratio for fundamental analysis
- **Comprehensive Reporting**: Generates detailed HTML reports with visualizations

## Repository Structure

```
investment-recommendation-system/
├── data/                            # Data files
│   ├── stocks/                      # Stock historical data (CSV files)
│   ├── mutual_funds/                # Mutual fund data files
│   ├── raw/                         # Original data files
│   └── processed/                   # Processed data files
├── models/                          # Trained LSTM models and metadata
├── results/                         # Analysis outputs
│   ├── predictions/                 # Price predictions and plots
│   ├── analysis/                    # Investment recommendation reports
│   └── validation/                  # Model validation metrics
├── src/                             # Core code modules
│   ├── model_trainer.py             # LSTM model training utilities
│   ├── data_processor.py            # Data preprocessing utilities
│   ├── risk_analyzer.py             # Risk assessment functions
│   ├── recommendation_engine.py     # Investment recommendation logic
│   └── report_generator.py          # Report generation utilities
├── static/                          # Static web resources
│   ├── css/                         # CSS stylesheets
│   └── images/                      # Image files and plots
├── templates/                       # Flask HTML templates
├── docs/                            # Documentation
│   └── SUMMARY.md                   # Project summary
├── notebooks/                       # Jupyter notebooks
├── app.py                           # Flask web application
├── validate_model.py                # Script to validate model performance
├── make_predictions.py              # Script to generate predictions
├── generate_report.py               # Script to create investment reports
├── run_investment_analysis.sh       # Full analysis pipeline script
└── requirements.txt                 # Project dependencies
```

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

Run the complete analysis pipeline with a single command:

```bash
./run_investment_analysis.sh
```

This will:
1. Validate models with training/testing split
2. Generate predictions for all stocks
3. Create a comprehensive investment report

## Running Individual Components

- **Model Validation**:
```bash
python validate_model.py
```

- **Generate Predictions**:
```bash
python make_predictions.py
```

- **Create Investment Report**:
```bash
python generate_report.py
```

- **Start Web Application**:
```bash
python app.py
```

## Web Application

The system includes a Flask web application that provides a user-friendly interface to the investment recommendation system. The app offers the following API endpoints:

- **Health Check**: `/api/health`
- **Train Models**: `/api/train-models`
- **Multi-timeframe Predictions**: `/api/multi-timeframe-predictions`
- **Backtracking Analysis**: `/api/backtracking-analysis`
- **Risk-based Recommendations**: `/api/risk-based-recommendations`
- **PEG Analysis**: `/api/peg-analysis`

## Model Details

The system uses LSTM (Long Short-Term Memory) neural networks to predict stock prices. Key features:

- **Input Features**: Price, Volume, Technical indicators (RSI, MACD, EMA, Bollinger Bands)
- **Sequence Length**: Variable (optimized per stock)
- **Prediction Horizons**: 1, 3, 5, 7, 14, 21, and 30 days
- **Training Data**: Minimum 6 months of historical data
- **Validation**: Train-test split (80/20) with sliding window evaluation

### Model Weighting

The system uses a sophisticated model weighting strategy:

- **Stocks**:
  - First 15 days: 100% LSTM model predictions
  - After 15 days: Weighted ensemble (20% LSTM, 40% ARIMA-GARCH, 40% Prophet)
  
- **Mutual Funds**:
  - Short-term (1-5 days): 50% LSTM, 30% ARIMA-GARCH, 20% Prophet
  - Medium-term (7-21 days): 40% LSTM, 30% ARIMA-GARCH, 30% Prophet
  - Long-term (30+ days): 20% LSTM, 40% ARIMA-GARCH, 40% Prophet

This hybrid approach leverages the strengths of each model for different time horizons, with LSTM proving more effective for short-term predictions and statistical models for longer-term forecasts.

## Risk Profiles

The system provides recommendations for three investor profiles:

1. **Conservative**: Focus on stable returns with lower volatility
2. **Moderate**: Balance between risk and return
3. **Aggressive**: Higher potential returns with higher volatility

## Results

The system generates the following outputs:

1. **Model Validation**: Performance metrics for each trained model
2. **Stock Predictions**: 30-day price forecasts with expected return calculations
3. **Risk Analysis**: Volatility and risk-adjusted return metrics
4. **Investment Recommendations**: Tailored stock suggestions for different risk profiles
5. **HTML Report**: Comprehensive visualization and analysis

## Disclaimer

This system is for educational and research purposes only. It does not constitute financial advice, and past performance is not indicative of future results. Always conduct your own research before making investment decisions.

