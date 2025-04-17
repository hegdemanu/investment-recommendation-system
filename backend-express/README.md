# Investment Recommendation System - Backend

This is the Express.js backend for the Investment Recommendation System, providing APIs for portfolio management, stock data, user authentication, and investment recommendations.

## Features

- User authentication (register, login, profile management)
- Portfolio management (create, update, delete portfolios and portfolio items)
- Stock data management (retrieve stock prices, histories, details)
- Investment recommendations based on user profiles and historical data
- News sentiment analysis for stocks
- Time-series data storage with TimescaleDB

## Tech Stack

- Node.js & Express.js
- PostgreSQL with TimescaleDB extension
- Sequelize ORM
- JWT Authentication
- RESTful API architecture

## Prerequisites

- Node.js (v18 or higher)
- PostgreSQL with TimescaleDB extension
- npm or yarn

## Installation & Setup

1. **Clone the repository**

```bash
git clone https://github.com/yourusername/investment-recommendation-system.git
cd investment-recommendation-system/backend-express
```

2. **Install dependencies**

```bash
npm install
```

3. **Set up environment variables**

Create a `.env` file in the root directory by copying the `.env.example` file:

```bash
cp .env.example .env
```

Update the values in the `.env` file with your configuration.

4. **Set up PostgreSQL with TimescaleDB**

Make sure you have PostgreSQL installed with TimescaleDB extension. Create a database for the application:

```sql
CREATE DATABASE investment_system;
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;
```

5. **Start the development server**

```bash
npm run dev
```

The server will start on the port specified in your `.env` file (default is 5000).

## API Documentation

### Authentication Endpoints

- `POST /api/users/register` - Register a new user
- `POST /api/users/login` - Login and get JWT token
- `GET /api/users/profile` - Get user profile
- `PUT /api/users/profile` - Update user profile

### Stock Endpoints

- `GET /api/stocks` - Get all stocks
- `GET /api/stocks/:symbol` - Get stock by symbol
- `POST /api/stocks` - Create new stock (admin)
- `PUT /api/stocks/:symbol` - Update stock data (admin)
- `DELETE /api/stocks/:symbol` - Delete a stock (admin)
- `GET /api/stocks/:symbol/history` - Get stock price history

### Portfolio Endpoints

- `GET /api/portfolios` - Get all portfolios for user
- `POST /api/portfolios` - Create a new portfolio
- `GET /api/portfolios/:id` - Get portfolio by ID
- `PUT /api/portfolios/:id` - Update portfolio
- `DELETE /api/portfolios/:id` - Delete portfolio
- `POST /api/portfolios/:id/stocks` - Add stock to portfolio
- `PUT /api/portfolios/:id/stocks/:stockId` - Update portfolio item
- `DELETE /api/portfolios/:id/stocks/:stockId` - Remove stock from portfolio

### Recommendation Endpoints

- `GET /api/recommendations/user` - Get recommendations for user
- `GET /api/recommendations/generate` - Generate new recommendations
- `GET /api/recommendations/stock/:symbol` - Get recommendations for stock

## Development

### Database Models

The system uses the following main models:

- User - User accounts and authentication
- Stock - Stock information
- StockPrice - Time-series price data (hypertable)
- Portfolio - User investment portfolios
- PortfolioItem - Stocks within portfolios
- Recommendation - Investment recommendations
- NewsSentiment - News sentiment analysis (hypertable)

### Folder Structure

```
/backend-express
├── /src
│   ├── /config        # Configuration files
│   ├── /controllers   # Route controllers
│   ├── /middleware    # Custom middleware
│   ├── /models        # Sequelize models
│   ├── /routes        # Express routes
│   ├── /utils         # Utility functions
│   └── server.js      # Entry point
├── .env.example       # Example environment variables
├── package.json       # Dependencies and scripts
└── README.md          # Project documentation
```

## License

MIT 