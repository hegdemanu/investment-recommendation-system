# Core Modules

This directory contains the core Python modules for the Investment Recommendation System.

## Module Overview

- **model_trainer.py**: Contains the `ModelTrainer` class for training LSTM models and making predictions
- **data_processor.py**: Handles data loading, cleaning, and preprocessing functions
- **prediction_utils.py**: Utilities for making predictions, backtracking analysis, and risk assessment
- **risk_analyzer.py**: Functions for analyzing investment risk and volatility
- **recommendation_engine.py**: Algorithm for generating investment recommendations based on predictions
- **report_generator.py**: Functions for generating reports and visualizations

## Usage

These modules are typically used by the main scripts in the repository root:

```python
from src.model_trainer import ModelTrainer
from src.prediction_utils import predict_multi_timeframe, generate_risk_based_recommendations

# Create a model trainer instance
model_trainer = ModelTrainer(models_dir="./models")

# Train models
trained_models = model_trainer.train_lstm_models(data)

# Load a model for predictions
model, scaler, features = model_trainer.load_model("TICKER")

# Make predictions for multiple timeframes
predictions = predict_multi_timeframe(model, last_sequence, scaler, features)

# Generate recommendations based on risk appetite
recommendations = generate_risk_based_recommendations(predictions, last_price, "moderate")
```

## Dependencies

These modules depend on the following packages:

- numpy
- pandas
- tensorflow/keras
- scikit-learn
- matplotlib
- seaborn 