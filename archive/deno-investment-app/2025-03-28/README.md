# InvestSage - Investment Recommendation System

A modern investment recommendation platform built with Deno, React, and Tailwind CSS. This application provides stock recommendations, portfolio analysis, and investment insights.

## Features

- Dashboard with portfolio performance metrics
- Interactive stock portfolio viewer 
- AI-powered investment recommendations
- User profile and preference settings
- Dark/light mode support
- Responsive design for all devices

## Tech Stack

### Backend
- Deno
- Oak (Express-like framework for Deno)
- SQLite database

### Frontend
- React
- Tailwind CSS
- Radix UI (accessible components)
- Recharts (data visualization)
- Lucide React (icons)

## Getting Started

### Prerequisites

- [Deno](https://deno.land/) v1.38 or higher
- [Node.js](https://nodejs.org/) v18 or higher (for frontend development)
- npm v9 or higher

### Development Setup

#### Backend

1. Navigate to the server directory:
   ```
   cd deno-investment-app/server
   ```

2. Run the Deno server:
   ```
   deno run --allow-net --allow-read --allow-write server.ts
   ```

#### Frontend

1. Navigate to the client directory:
   ```
   cd deno-investment-app/client
   ```

2. Install dependencies:
   ```
   npm install
   ```

3. Start the development server:
   ```
   npm run dev
   ```

4. The application will be available at `http://localhost:5173` and will proxy API requests to the Deno server at `http://localhost:8000`.

### Building for Production

1. Build the frontend:
   ```
   cd deno-investment-app/client
   npm run build
   ```

2. The built files will be in the `dist` folder, which the Deno server is configured to serve.

3. Run the production server:
   ```
   cd deno-investment-app/server
   deno run --allow-net --allow-read --allow-write server.ts
   ```

4. The complete application will be served at `http://localhost:8000`.

## Project Structure

```
deno-investment-app/
├── client/               # Frontend React application
│   ├── public/           # Static assets
│   └── src/
│       ├── components/   # Reusable UI components
│       └── pages/        # Page components
├── server/               # Deno backend
│   ├── deps.ts           # Dependencies
│   └── server.ts         # Server implementation
└── README.md             # Documentation
```

## API Endpoints

- `GET /api/health` - Health check endpoint
- `GET /api/recommendations` - Get all stock recommendations
- `GET /api/recommendations/:symbol` - Get recommendation for a specific stock
- `GET /api/portfolio/summary` - Get portfolio summary statistics
- `GET /api/models/performance` - Get ML model performance analytics

## License

MIT 