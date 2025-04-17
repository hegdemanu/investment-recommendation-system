import express from 'express';
import * as stockController from '../controllers/stock.controller.js';

const router = express.Router();

/**
 * @route   GET /api/stocks
 * @desc    Get all stocks
 * @access  Public
 */
router.get('/', stockController.getAllStocks);

/**
 * @route   POST /api/stocks
 * @desc    Create a new stock
 * @access  Private/Admin
 */
router.post('/', stockController.createStock);

/**
 * @route   GET /api/stocks/:symbol
 * @desc    Get stock by symbol
 * @access  Public
 */
router.get('/:symbol', stockController.getStockBySymbol);

/**
 * @route   PUT /api/stocks/:symbol
 * @desc    Update stock details
 * @access  Private/Admin
 */
router.put('/:symbol', stockController.updateStock);

/**
 * @route   DELETE /api/stocks/:symbol
 * @desc    Delete a stock
 * @access  Private/Admin
 */
router.delete('/:symbol', stockController.deleteStock);

/**
 * @route   GET /api/stocks/:symbol/history
 * @desc    Get stock price history
 * @access  Public
 */
router.get('/:symbol/history', stockController.getStockPriceHistory);

export default router; 