import express from 'express';
import { protect } from '../middleware/auth.middleware.js';
import { 
  createPortfolio,
  getPortfolios,
  getPortfolioById,
  updatePortfolio,
  deletePortfolio,
  addStockToPortfolio,
  updatePortfolioItem,
  removeStockFromPortfolio,
  getPortfolioItems,
  getPortfolioPerformance
} from '../controllers/portfolio.controller.js';

const router = express.Router();

// @route   POST /api/portfolios
// @desc    Create a new portfolio
// @access  Private
router.post('/', protect, createPortfolio);

// @route   GET /api/portfolios
// @desc    Get all portfolios for logged in user
// @access  Private
router.get('/', protect, getPortfolios);

// @route   GET /api/portfolios/:id
// @desc    Get portfolio by ID
// @access  Private
router.get('/:id', protect, getPortfolioById);

// @route   PUT /api/portfolios/:id
// @desc    Update a portfolio
// @access  Private
router.put('/:id', protect, updatePortfolio);

// @route   DELETE /api/portfolios/:id
// @desc    Delete a portfolio
// @access  Private
router.delete('/:id', protect, deletePortfolio);

// @route   POST /api/portfolios/:id/stocks
// @desc    Add a stock to portfolio
// @access  Private
router.post('/:id/stocks', protect, addStockToPortfolio);

// @route   PUT /api/portfolios/:id/stocks/:stockId
// @desc    Update a stock in portfolio
// @access  Private
router.put('/:id/stocks/:stockId', protect, updatePortfolioItem);

// @route   DELETE /api/portfolios/:id/stocks/:stockId
// @desc    Remove a stock from portfolio
// @access  Private
router.delete('/:id/stocks/:stockId', protect, removeStockFromPortfolio);

// @route   GET /api/portfolios/:id/stocks
// @desc    Get all stocks in a portfolio
// @access  Private
router.get('/:id/stocks', protect, getPortfolioItems);

// @route   GET /api/portfolios/:id/performance
// @desc    Get portfolio performance metrics
// @access  Private
router.get('/:id/performance', protect, getPortfolioPerformance);

export default router; 