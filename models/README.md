# Models Directory

This directory stores all trained machine learning models and associated artifacts used by the Investment Recommendation System.

## Model Types

### LSTM Models
- Stored as `.h5` files (TensorFlow/Keras format)
- Naming convention: `{ticker}_lstm.h5`
- Used for short-term predictions (1 day, 1 week, 1 month)
- Trained on sequence data with multiple features

### ARIMA-GARCH Models
- Stored as serialized pickle files (`.pkl`)
- Naming convention: `{ticker}_arima_garch.pkl`
- Used for medium-term predictions (3-6 months)
- Captures both trend and volatility patterns

### Prophet Models
- Stored as serialized pickle files (`.pkl`)
- Naming convention: `{ticker}_prophet.pkl`
- Used for long-term forecasting (1+ year)
- Specialized for handling seasonality and holidays

## Associated Files

### Scalers
- Stored as serialized pickle files (`.pkl`)
- Naming convention: `{ticker}_scaler.pkl`
- Contains fitted MinMaxScaler or StandardScaler instances
- Required to preprocess new data and inverse transform predictions

### Feature Lists
- Stored as serialized pickle files (`.pkl`)
- Naming convention: `{ticker}_features.pkl`
- Contains list of features used during model training
- Ensures consistency between training and prediction

### Training Metrics
- Stored as CSV files
- Naming convention: `{ticker}_metrics.csv`
- Contains performance metrics from model training
- Useful for tracking model improvement over time

### Data Statistics
- File: `data_stats.csv`
- Contains statistical properties of training data
- Used for data drift detection
- Helps determine when retraining is necessary

## Model Versioning

Models are versioned using timestamps in their filenames if multiple versions exist:
- `RELIANCE.NS_lstm_20230615.h5`
- `RELIANCE.NS_lstm_20230930.h5`

The system automatically uses the most recent model unless specified otherwise.

## Visualization Artifacts

- Training loss plots are stored as PNG files
- Naming convention: `{ticker}_training_loss.png`
- Show training and validation loss over epochs
- Useful for diagnosing overfitting or underfitting

## Storage Requirements

- LSTM models: ~5-20 MB per ticker
- ARIMA-GARCH models: ~1-5 MB per ticker
- Prophet models: ~1-5 MB per ticker
- Associated files: ~1 MB per ticker
- Plan storage accordingly for the number of tickers

## Using Saved Models

To use saved models for prediction:
```python
from tensorflow.keras.models import load_model
import pickle

# Load LSTM model
model = load_model(f'models/{ticker}_lstm.h5')

# Load scaler
with open(f'models/{ticker}_scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Load features
with open(f'models/{ticker}_features.pkl', 'rb') as f:
    features = pickle.load(f)
```

## Model Maintenance

- Periodically check model performance against recent data
- Retrain models when data drift is detected
- Archive old models instead of deleting them
- Consider ensemble approaches combining multiple model versions 