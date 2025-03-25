# Investment Recommendation System

A comprehensive system for analyzing stock and mutual fund performance, training prediction models, and generating investment recommendations with interactive dashboards.

## How to Run the Application

### Quick Start

```bash
# Run the dashboard
python run_investment_system.py dashboard

# Run full analysis (training, predictions, reports)
python run_investment_system.py full-analysis
```

Alternatively, use the platform-independent launcher:
```bash
# On macOS/Linux
./launch.py

# On Windows
python launch.py
```

## System Components

The Investment Recommendation System includes the following components:

1. **Data Collection**: Historical stock and mutual fund data acquisition
2. **Model Training**: LSTM, GRU, and Ensemble models for price forecasting
3. **Performance Analysis**: Evaluation of model accuracy and training efficiency
4. **Dashboard Generation**: Interactive HTML dashboard with visualizations
5. **Investment Recommendations**: Data-driven suggestions for optimal investments based on risk profile

## System Requirements

- Python 3.8 or higher
- Required Python packages (listed in requirements.txt):
  - pandas
  - numpy
  - matplotlib
  - seaborn
  - tensorflow (for deep learning models)
  - flask (for web interface)

## Features

- **Multi-Model Prediction**: Trains LSTM, GRU, and Ensemble models to predict future stock prices using historical data
- **Time Horizon Optimization**: Uses LSTM for short-term, GRU for medium-term, and Ensemble models for long-term predictions
- **Sliding Window Approach**: Uses a 6-month sliding window with 1-day steps for optimal prediction horizons
- **Multiple Time Frame Analysis**: Provides short-term, medium-term, and long-term predictions
- **Backtracking Analysis**: Evaluates model performance on historical data
- **Risk-Based Recommendations**: Tailors investment suggestions based on risk appetite
- **PEG Ratio Analysis**: Incorporates Price/Earnings to Growth ratio for fundamental analysis
- **Interactive Dashboard**: Generates detailed interactive dashboard with visualizations

## Repository Structure

```
investment-recommendation-system/
├── app/                            # Main application code
│   ├── core/                       # Core business logic
│   │   ├── model_trainer.py        # Model training utilities for LSTM, GRU, and Ensemble models
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
├── config/                         # Configuration files
│   └── settings.py                 # Central configuration settings
├── data/                           # Data files
│   ├── stocks/                     # Stock historical data (CSV files)
│   ├── mutual_funds/               # Mutual fund data files
│   ├── uploads/                    # User-uploaded data files
├── models/                         # Trained prediction models and metadata
├── results/                        # Analysis outputs
│   ├── predictions/                # Price predictions and plots
│   ├── reports/                    # Generated reports
│   ├── dashboard/                  # Dashboard HTML files
│   │   └── json/                   # JSON files for dashboard
│   ├── training/                   # Training metrics and summaries
│   └── validation/                 # Model validation metrics
├── docs/                           # Documentation
├── logs/                           # Log files
├── run_investment_system.py        # Unified command-line interface
├── launch.py                       # Platform-independent launcher
├── CHANGELOG.md                    # Version history and changes
├── archive/                        # Legacy code and deprecated scripts
└── requirements.txt                # Project dependencies
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/hegdemanu/investment-recommendation-system
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

## Running Individual Components

The system uses a unified command-line interface through `run_investment_system.py`:

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

## Dashboard Features

The interactive dashboard provides:

- **Portfolio Diversification View**: Visual breakdown of investments by sector
- **Model Performance Metrics**: Accuracy, RMSE, and other training metrics
- **Prediction Visualizations**: Graphical representations of price forecasts
- **Risk Analysis**: Risk assessment based on portfolio composition
- **Recommendation Engine**: Investment suggestions based on model predictions
- **JSON Report Viewer**: Interactive viewer for detailed JSON reports
- **Data Browser**: Access to raw and processed data files

## Model Details

The system uses three complementary models for price prediction:

1. **LSTM (Long Short-Term Memory)**: Primary model for short-term predictions
   - Best for: 1-7 day horizons
   - Features: Price, Volume, Technical indicators (RSI, MACD, EMA, Bollinger Bands)

2. **GRU (Gated Recurrent Unit)**: Optimized for medium-term predictions
   - Best for: 7-14 day horizons
   - More efficient training time than LSTM
   - Better handling of irregular patterns

3. **Ensemble Model**: Weighted combination for long-term predictions
   - Best for: 14-30 day horizons
   - Combines LSTM, GRU and statistical methods
   - Adaptive weighting based on historical accuracy

**Common Model Parameters**:
- **Sequence Length**: Variable (optimized per stock)
- **Prediction Horizons**: 1, 3, 5, 7, 14, 21, and 30 days
- **Training Data**: Minimum 6 months of historical data
- **Validation**: Train-test split (80/20) with sliding window evaluation

## Dashboard Preview

![Dashboard Overview](docs/images/dashboard_preview.png)

The investment recommendation system generates an interactive dashboard with:

- **Price Predictions**: Visual display of model predictions across different time horizons
- **Portfolio Allocation**: Interactive pie charts for sector-based diversification
- **Performance Metrics**: Accuracy visualization and comparison between models
- **JSON Viewer**: Built-in viewer for examining detailed model metadata and results
- **Responsive Design**: Adapts to different screen sizes with mobile-friendly controls

For a live demo, run: `python run_investment_system.py dashboard`

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

