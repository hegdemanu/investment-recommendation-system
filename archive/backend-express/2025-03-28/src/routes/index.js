import express from 'express';
import stockRoutes from './stock.routes.js';

const router = express.Router();

// Import route modules
// Example: const authRoutes = require('./auth.routes');

// Basic test route
router.get('/status', (req, res) => {
  res.status(200).json({
    status: 'API is operational',
    timestamp: new Date(),
    environment: process.env.NODE_ENV || 'development',
    version: '0.1.0',
  });
});

// Use route modules
router.use('/stocks', stockRoutes);
// Example: router.use('/auth', authRoutes);

// Placeholder for future route modules
// These will be implemented as we develop the system
router.use('/auth', (req, res) => {
  res.status(501).json({ message: 'Authentication endpoints coming soon' });
});

router.use('/recommendations', (req, res) => {
  res.status(501).json({ message: 'Investment recommendation endpoints coming soon' });
});

router.use('/portfolio', (req, res) => {
  res.status(501).json({ message: 'Portfolio management endpoints coming soon' });
});

router.use('/analytics', (req, res) => {
  res.status(501).json({ message: 'Analytics endpoints coming soon' });
});

export default router; 