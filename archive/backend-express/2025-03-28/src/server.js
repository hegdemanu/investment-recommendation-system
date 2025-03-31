import express from 'express';
import cors from 'cors';
import dotenv from 'dotenv';
import morgan from 'morgan';
import { runStartupScripts } from './utils/startupScripts.js';
import userRoutes from './routes/user.routes.js';
import stockRoutes from './routes/stock.routes.js';
import portfolioRoutes from './routes/portfolio.routes.js';
import recommendationRoutes from './routes/recommendation.routes.js';

// Load environment variables
dotenv.config();

const app = express();
const PORT = process.env.PORT || 5000;

// Middleware
app.use(cors());
app.use(express.json());

// Logging middleware
if (process.env.NODE_ENV === 'development') {
  app.use(morgan('dev'));
}

// Routes
app.use('/api/users', userRoutes);
app.use('/api/stocks', stockRoutes);
app.use('/api/portfolios', portfolioRoutes);
app.use('/api/recommendations', recommendationRoutes);

// Root route
app.get('/', (req, res) => {
  res.json({ message: 'Welcome to Investment Recommendation System API' });
});

// Error handling middleware
app.use((err, req, res, next) => {
  const statusCode = err.statusCode || 500;
  console.error(err.message, err.stack);
  res.status(statusCode).json({ 
    message: err.message,
    stack: process.env.NODE_ENV === 'production' ? '🥞' : err.stack
  });
});

// Start server
const startServer = async () => {
  try {
    // Run startup scripts (database connection, migrations, seed data)
    await runStartupScripts();
    
    // Start Express server
    app.listen(PORT, () => {
      console.log(`Server running in ${process.env.NODE_ENV} mode on port ${PORT}`);
    });
  } catch (error) {
    console.error('Failed to start server:', error);
    process.exit(1);
  }
};

// Initialize server
startServer(); 