# Investment ML Service

Machine learning service for stock predictions and sentiment analysis.

## Structure

```
ml-service/
├── src/
│   ├── api/        # FastAPI application
│   ├── models/     # ML models
│   ├── utils/      # Utilities
│   └── data/       # Data processing
├── tests/          # Unit tests
├── data/           # Training data
└── requirements.txt
```

## Setup

1. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Start the service:
```bash
./start.sh  # On Windows: start.bat
```

The service will be available at `http://localhost:5001`

## API Endpoints

- `GET /health`: Health check
- `POST /predict`: Get predictions for a symbol
- `POST /train/{symbol}`: Train model for a symbol
- `GET /models`: List available models

## Models

- LSTM for price prediction
- FinBERT for sentiment analysis
- Technical indicators 