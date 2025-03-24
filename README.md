# Investment Recommendation System

A comprehensive system that analyzes stocks and mutual funds to provide personalized investment recommendations using machine learning models.

## Features

- **Data Analysis**
  - Stock price analysis
  - Mutual fund NAV analysis
  - Technical indicators calculation
  - Feature engineering

- **Machine Learning Models**
  - LSTM for short-term predictions
  - ARIMA-GARCH for medium-term predictions
  - Prophet for long-term forecasting

- **Risk Analysis**
  - Volatility calculation
  - Sharpe ratio analysis
  - Risk classification
  - Portfolio risk assessment

- **Portfolio Optimization**
  - Modern Portfolio Theory implementation
  - Risk-adjusted allocation
  - Balanced portfolio strategy
  - Asset allocation optimization

- **Reporting and Visualization**
  - Price predictions
  - Risk profiles
  - Portfolio recommendations
  - Interactive visualizations

## Repository Structure

```
investment-recommendation-system/
├── notebooks/
│   └── Investment_Recommendation_System.ipynb
├── data/
│   ├── uploads/           # User uploaded CSV files
│   ├── raw/              # Original data files
│   └── processed/        # Processed data files
├── models/
│   ├── stock_models/     # Trained stock prediction models
│   └── mf_models/        # Trained mutual fund models
├── results/
│   ├── stock_portfolio_allocation.csv
│   ├── mf_portfolio_allocation.csv
│   └── portfolio_summary.csv
├── src/
│   ├── data_processor.py
│   ├── model_trainer.py
│   ├── risk_analyzer.py
│   ├── recommendation_engine.py
│   └── report_generator.py
├── static/
│   └── css/
├── templates/
│   └── index.html
├── app.py
├── requirements.txt
└── README.md
```

## Prerequisites

- Python 3.8+
- Google Colab with GPU runtime
- Required CSV files:
  - 12 stock files (stock_1.csv to stock_12.csv)
  - 6 mutual fund files (mf_1.csv to mf_6.csv)

## Data Format Requirements

### Stock CSV Files
Required columns:
- Date
- Price
- Open
- High
- Low
- Volume
- Change %

### Mutual Fund CSV Files
Required columns:
- Date
- NAV

## Installation

1. Clone the repository:
```bash
git clone [https://github.com/hegdemanu/investment-recommendation-system/blob/main/notebooks/Investment_Recommendation_System.ipynb](https://github.com/hegdemanu/investment-recommendation-system/tree/main)
cd investment-recommendation-system
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

### Using Google Colab Notebook

1. Upload the notebook to Google Colab
2. Select GPU runtime
3. Upload your CSV files
4. Run cells in sequence
5. Check results in the 'results' directory

### Using Local Environment

1. Set up your environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

2. Run the application:
```bash
python app.py
```

3. Access the web interface at `http://localhost:5000`

## Model Details

### LSTM Model
- Architecture: 2 LSTM layers with dropout
- Features: Price, volume, technical indicators
- Prediction horizon: Short-term (1-7 days)

### ARIMA-GARCH Model
- Purpose: Medium-term predictions
- Features: Price volatility and trends
- Prediction horizon: Medium-term (1-3 months)

### Prophet Model
- Purpose: Long-term forecasting
- Features: Seasonal patterns and trends
- Prediction horizon: Long-term (3-12 months)

## Results

The system generates the following outputs:

1. **Portfolio Allocation**
   - Stock weights
   - Mutual fund weights
   - Risk-adjusted allocation

2. **Risk Analysis**
   - Volatility metrics
   - Sharpe ratios
   - Risk classifications

3. **Predictions**
   - Price forecasts
   - NAV predictions
   - Trend analysis

4. **Visualizations**
   - Portfolio allocation charts
   - Risk-return plots
   - Price prediction graphs

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request


## Acknowledgments

- TensorFlow and Keras for deep learning models
- Prophet for time series forecasting
- Pandas and NumPy for data processing
- Matplotlib and Plotly for visualizations

## Contact

B G Manu- f20212393@goa.bits-pilani.ac.in
Project Link: [https://github.com/hegdemanu/investment-recommendation-system/blob/main/notebooks/Investment_Recommendation_System.ipynb](https://github.com/hegdemanu/investment-recommendation-system/tree/main)
