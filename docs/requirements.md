# 📋 Project Requirements

This document outlines all dependencies, libraries, and APIs required for the Investment Recommendation System.

## Frontend Dependencies

### Core
```
react: ^19.0.0
react-dom: ^19.0.0
next: ^15.0.0
typescript: ^5.3.0
```

### UI Components and Styling
```
tailwindcss: ^4.0.0
@radix-ui/react-*: ^2.0.0
class-variance-authority: ^0.8.0
clsx: ^2.0.0
```

### State Management
```
@tanstack/react-query: ^5.0.0
zustand: ^5.0.0
```

### Forms and Validation
```
react-hook-form: ^8.0.0
zod: ^4.0.0
```

### Data Visualization
```
chart.js: ^5.0.0
react-chartjs-2: ^6.0.0
lightweight-charts: ^4.0.0
```

## Backend (Node.js) Dependencies

### Core
```
express: ^5.0.0
cors: ^3.0.0
helmet: ^8.0.0
morgan: ^2.0.0
dotenv: ^17.0.0
```

### Database
```
pg: ^9.0.0
sequelize: ^7.0.0
mongoose: ^8.0.0 (optional)
ioredis: ^6.0.0
```

### Authentication
```
jsonwebtoken: ^10.0.0
bcryptjs: ^3.0.0
```

### Utilities
```
axios: ^2.0.0
joi: ^18.0.0
winston: ^4.0.0
```

## Python Backend Dependencies

### Core
```
fastapi>=1.0.0
uvicorn>=0.25.0
python-dotenv>=1.1.0
httpx>=0.26.0
```

### Data Science and ML
```
numpy>=2.0.0
pandas>=3.0.0
scikit-learn>=1.5.0
tensorflow>=3.0.0 or pytorch>=2.5.0
statsmodels>=0.15.0
prophet>=1.2.0
transformers>=5.0.0
```

### Database
```
sqlalchemy>=2.1.0
psycopg2-binary>=3.0.0
alembic>=1.12.0
pymongo>=5.0.0 (optional)
redis>=5.0.0
```

## APIs and External Services

### Financial Data APIs
- Alpha Vantage API
- Yahoo Finance API
- FRED Economic Data API
- Finnhub API

### News and Sentiment
- NewsAPI.org
- Twitter API (optional)
- Reddit API (optional)

### Security and Authentication
- Auth0 / NextAuth.js
- reCAPTCHA v3

## Development and DevOps Tools

### Testing
```
# Frontend
jest: ^30.0.0
@testing-library/react: ^15.0.0
cypress: ^13.0.0

# Backend (Node.js)
jest: ^30.0.0
supertest: ^7.0.0

# Backend (Python)
pytest>=8.0.0
httpx>=0.26.0
```

### Linting and Formatting
```
eslint: ^9.0.0
prettier: ^3.0.0
pylint>=3.0.0
black>=24.0.0
```

### DevOps
```
docker: ^25.0.0
docker-compose: ^3.0.0
```

---

## 🔄 Version Control
**Version:** 2.0.0
**Last Updated:** 2025-05-28 