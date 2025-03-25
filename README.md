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

© 2025 Investment Recommendation System

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
├── app/                            # Main application code
│   ├── core/                       # Core business logic
│   │   ├── model_trainer.py        # LSTM model training utilities
│   │   ├── data_processor.py       # Data preprocessing utilities
│   │   ├── risk_analyzer.py        # Risk assessment functions
│   │   └── recommendation_engine.py # Investment recommendation logic
│   ├── dashboard/                  # Dashboard generation
│   │   ├── dashboard_generator.py  # Dashboard HTML generation
│   │   └── dashboard_template.py   # HTML template for dashboard
│   ├── api/                        # API endpoints for web interface
│   ├── utils/                      # Utility functions
│   │   ├── file_utils.py           # File operations utilities
│   │   └── data_utils.py           # Data processing utilities
│   ├── templates/                  # Flask HTML templates
│   └── static/                     # Static web resources
│       ├── css/                    # CSS stylesheets
│       ├── js/                     # JavaScript files
│       └── images/                 # Image files and plots
├── config/                         # Configuration files
│   └── settings.py                 # Central configuration settings
├── data/                           # Data files
│   ├── stocks/                     # Stock historical data (CSV files)
│   ├── mutual_funds/               # Mutual fund data files
│   ├── uploads/                    # User-uploaded data files
│   ├── raw/                        # Original data files
│   └── processed/                  # Processed data files
├── models/                         # Trained LSTM models and metadata
├── results/                        # Analysis outputs
│   ├── predictions/                # Price predictions and plots
│   ├── reports/                    # Generated reports
│   ├── dashboard/                  # Dashboard HTML files
│   ├── training/                   # Training metrics and summaries
│   └── validation/                 # Model validation metrics
├── docs/                           # Documentation
├── run_investment_system.py        # Unified command-line interface
├── CHANGELOG.md                    # Version history and changes
├── archive/                        # Legacy code (pre-refactoring)
│   ├── scripts/                    # Old script files
│   ├── src/                        # Old source code
│   └── web/                        # Old templates and static files
└── requirements.txt                # Project dependencies
```

## Installation

1. Clone the repository:
```bash
git clone <https://github.com/hegdemanu/investment-recommendation-system>
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
python run_investment_system.py full-analysis
```

This will:
1. Validate models with training/testing split
2. Generate predictions for all stocks
3. Create a comprehensive investment report
4. Open the dashboard in your browser

## Running Individual Components

The system now uses a unified command-line interface through `run_investment_system.py`:

- **Model Validation**:
```bash
python run_investment_system.py validate
```

- **Generate Predictions**:
```bash
python run_investment_system.py predict
```

- **Create Dashboard**:
```bash
python run_investment_system.py dashboard
```

- **Generate Sample Data** (for demonstration):
```bash
python run_investment_system.py sample
```

- **Start Web Application**:
```bash
python run_investment_system.py webapp
```

## Easy Launch Options

Several launcher files are provided for quick startup:

### Simple Python Launcher (Recommended for All Platforms)
- `launch.py` - Python script that works on all platforms
- Automatically detects the correct Python environment
- Usage: 
  ```bash
  # On macOS/Linux
  ./launch.py
  
  # On Windows
  python launch.py
  ```

### Shell Scripts for macOS/Linux
- `run.sh` - Basic launcher that uses the local conda environment
- `conda_run.sh` - Advanced launcher that properly activates the conda environment
- Usage:
  ```bash
  # Make executable first (one-time setup)
  chmod +x run.sh conda_run.sh
  
  # Then run
  ./run.sh
  # or
  ./conda_run.sh
  ```

### Batch File for Windows
- `run.bat` - Windows batch file that works with both system Python and local conda
- Usage: Double-click the file in Windows Explorer or run from Command Prompt
  ```
  run.bat
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

## Directory Structure

The project has been reorganized into a more modular and maintainable structure:

```
investment-recommendation-system/
├── app/                    # Main application code
│   ├── core/               # Core business logic
│   ├── dashboard/          # Dashboard generation
│   ├── api/                # API endpoints
│   ├── utils/              # Utility functions
│   ├── templates/          # HTML templates
│   └── static/             # Static assets (CSS, JS, images)
├── config/                 # Configuration files
├── data/                   # Data storage
│   ├── stocks/             # Stock data
│   ├── mutual_funds/       # Mutual fund data
│   ├── uploads/            # User-uploaded data
│   ├── processed/          # Processed data files
│   └── raw/                # Raw data files
├── models/                 # Trained models
├── results/                # Analysis results
│   ├── reports/            # Generated reports
│   ├── training/           # Training metrics
│   ├── validation/         # Validation metrics
│   ├── predictions/        # Prediction results
│   └── dashboard/          # Dashboard output
├── docs/                   # Documentation
└── run_investment_system.py  # Main entry point
```

## Running the System

The system now provides a streamlined interface for running different commands:

### Dashboard Generation

To generate and view the dashboard:

```bash
# On Unix-like systems:
./run_investment_system.py dashboard

# On Windows:
python run_investment_system.py dashboard

# To generate the dashboard without opening the browser:
python run_investment_system.py dashboard --no-browser
```

### Sample Data Generation

To generate sample data for demonstration purposes:

```bash
python run_investment_system.py sample
```

### Model Training (Coming Soon)

To train prediction models:

```bash
# Train models for all available stocks:
python run_investment_system.py train --all

# Train models for specific stocks:
python run_investment_system.py train --stocks AAPL MSFT GOOGL
```

### Report Generation (Coming Soon)

To generate investment recommendation reports:

```bash
# Generate a report with medium risk profile (default):
python run_investment_system.py report

# Generate a report with specific risk profile:
python run_investment_system.py report --risk-profile low
```

### Web Interface

To run the web interface:

```bash
# Run the web interface on the default port (8000):
python run_investment_system.py web

# Run the web interface on a custom port:
python run_investment_system.py web --port 5000
```

