# Investment Recommendation System API

A FastAPI-based backend service for generating and managing investment recommendations.

## Features

- User authentication and authorization
- Portfolio management
- Investment recommendations based on Modern Portfolio Theory
- Real-time market data integration
- Portfolio rebalancing suggestions
- RESTful API with OpenAPI documentation

## Prerequisites

- Python 3.9+
- PostgreSQL 15+
- Redis 7+
- Docker and Docker Compose (optional)

## Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd apps/api
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

5. Start the database and Redis (using Docker):
```bash
docker-compose up -d
```

6. Run the application:
```bash
uvicorn src.main:app --reload
```

The API will be available at `http://localhost:8000`.
API documentation will be available at `http://localhost:8000/docs`.

## API Endpoints

### Authentication
- `POST /api/v1/auth/register` - Register a new user
- `POST /api/v1/auth/login` - Login and get access token

### Portfolios
- `GET /api/v1/portfolios` - List user's portfolios
- `POST /api/v1/portfolios` - Create a new portfolio
- `GET /api/v1/portfolios/{id}` - Get portfolio details
- `PUT /api/v1/portfolios/{id}` - Update portfolio
- `DELETE /api/v1/portfolios/{id}` - Delete portfolio

### Portfolio Holdings
- `GET /api/v1/portfolios/{id}/holdings` - List portfolio holdings
- `POST /api/v1/portfolios/{id}/holdings` - Add holding to portfolio
- `PUT /api/v1/portfolios/{id}/holdings/{holding_id}` - Update holding
- `DELETE /api/v1/portfolios/{id}/holdings/{holding_id}` - Remove holding

### Recommendations
- `POST /api/v1/recommendations/generate` - Generate investment recommendations
- `POST /api/v1/recommendations/{portfolio_id}/rebalance` - Get rebalancing suggestions

## Development

### Project Structure

```
src/
├── api/
│   ├── deps.py
│   └── v1/
│       ├── endpoints/
│       └── router.py
├── core/
│   ├── config.py
│   ├── security.py
│   └── logger.py
├── crud/
│   ├── base.py
│   ├── user.py
│   └── portfolio.py
├── db/
│   └── session.py
├── models/
│   ├── base.py
│   ├── user.py
│   └── portfolio.py
├── services/
│   ├── market_data.py
│   ├── portfolio_optimizer.py
│   └── recommendation.py
└── main.py
```

### Running Tests

```bash
pytest
```

### Code Style

The project uses:
- Black for code formatting
- isort for import sorting
- mypy for type checking
- ruff for linting

Run all checks:
```bash
black .
isort .
mypy .
ruff .
```

## External APIs

The system integrates with:
- Finnhub for real-time market data
- Alpha Vantage for historical data
- yfinance for additional market data

Make sure to obtain API keys and configure them in your `.env` file.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License. 