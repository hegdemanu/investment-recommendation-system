# ML Service for Investment Recommendation System

This service provides ML model prediction, training, and management for the Investment Recommendation System.

## Features

- Stock price prediction using LSTM, ARIMA+GARCH, and Prophet models
- Model training and management with versioning
- Sentiment analysis for stock symbols
- Model recommendation based on performance metrics
- API endpoints for frontend integration

## Setup and Installation

1. Make sure you have Python 3.8+ installed
2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Starting the Service

Run the startup script:

```bash
./start.sh
```

Or manually:

```bash
uvicorn src.api.app:app --host 0.0.0.0 --port 5001 --reload
```

The API will be available at http://localhost:5001

## API Documentation

Once the service is running, you can access the interactive API documentation at:
- Swagger UI: http://localhost:5001/docs
- ReDoc: http://localhost:5001/redoc

## API Endpoints

### Prediction

- `POST /predict` - Generate predictions for a stock symbol

Example request:
```json
{
  "symbol": "AAPL",
  "modelType": "LSTM",
  "horizon": 5,
  "includeMetadata": true
}
```

### Model Management

- `GET /models` - Get all available models
- `GET /models/symbol/{symbol}` - Get models for a specific symbol
- `GET /models/compare/{symbol}` - Compare models for a symbol
- `DELETE /models/{model_id}` - Delete a model by ID
- `POST /retrain` - Retrain a model for a symbol

### Sentiment Analysis

- `GET /sentiment/{symbol}` - Analyze sentiment for a symbol
- `GET /sentiment/{symbol}/detailed` - Get detailed sentiment analysis

### Model Selection

- `GET /selector/{symbol}` - Get model recommendation for a symbol

### RAG (Retrieval Augmented Generation)

- `POST /rag/query` - Query the RAG system

Example request:
```json
{
  "query": "What are the growth prospects for Apple?",
  "context": "AAPL"
}
```

## Integration with Frontend

The ML service is designed to integrate with the frontend via API calls. The frontend components in `apps/web/src/hooks` are already set up to communicate with these endpoints.

To ensure proper communication:

1. Make sure the ML service is running on port 5001
2. Verify that the API base URL in the frontend is correctly set to `http://localhost:5001`

## Training New Models

To train a new model for a stock symbol:

```bash
curl -X POST http://localhost:5001/retrain \
  -H "Content-Type: application/json" \
  -d '{"symbol": "AAPL", "modelType": "LSTM"}'
```

This will start training in the background. You can check the status by querying the models endpoint.

## Model Registry

Models are stored in the `ml-service/models` directory, with metadata tracking versions, performance metrics, and hyperparameters. 