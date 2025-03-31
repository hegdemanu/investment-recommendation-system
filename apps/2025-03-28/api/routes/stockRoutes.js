const express = require('express');
const router = express.Router();
const { getStocks, getStockById, createSampleStocks } = require('../controllers/stockController');
const auth = require('../middleware/auth');

// Get all stocks
router.get('/', auth, getStocks);

// Get single stock by ID
router.get('/:id', auth, getStockById);

// Create sample stocks for testing
router.post('/sample', auth, createSampleStocks);

module.exports = router; 