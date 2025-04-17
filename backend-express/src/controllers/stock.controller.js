/**
 * Stock Controller
 * Handles stock-related operations and API endpoints
 */

import { Stock, StockPrice } from '../models/index.js';
import { Op } from 'sequelize';
import sequelize from '../config/database.js';

/**
 * Get all stocks
 * @param {Object} req - Express request object
 * @param {Object} res - Express response object
 */
export const getAllStocks = async (req, res) => {
  try {
    const { sector, limit = 50, offset = 0, search } = req.query;
    
    // Build query conditions
    const whereConditions = {};
    
    if (sector) {
      whereConditions.sector = sector;
    }
    
    if (search) {
      whereConditions[Op.or] = [
        { symbol: { [Op.iLike]: `%${search}%` } },
        { name: { [Op.iLike]: `%${search}%` } }
      ];
    }
    
    // Fetch stocks from database
    const stocks = await Stock.findAndCountAll({
      where: whereConditions,
      limit: parseInt(limit),
      offset: parseInt(offset),
      order: [['symbol', 'ASC']]
    });
    
    res.status(200).json({
      success: true,
      count: stocks.count,
      data: stocks.rows
    });
  } catch (error) {
    console.error('Error fetching stocks:', error);
    res.status(500).json({
      success: false,
      error: 'Server error',
      message: error.message
    });
  }
};

/**
 * Get stock by symbol
 * @param {Object} req - Express request object
 * @param {Object} res - Express response object
 */
export const getStockBySymbol = async (req, res) => {
  try {
    const { symbol } = req.params;
    
    // Find stock in database
    const stock = await Stock.findOne({
      where: { symbol: symbol.toUpperCase() }
    });
    
    if (!stock) {
      return res.status(404).json({
        success: false,
        error: 'Stock not found',
        message: `No stock found with symbol ${symbol}`
      });
    }
    
    res.status(200).json({
      success: true,
      data: stock
    });
  } catch (error) {
    console.error('Error fetching stock:', error);
    res.status(500).json({
      success: false,
      error: 'Server error',
      message: error.message
    });
  }
};

/**
 * Get stock price history
 * @param {Object} req - Express request object
 * @param {Object} res - Express response object
 */
export const getStockPriceHistory = async (req, res) => {
  try {
    const { symbol } = req.params;
    const { 
      period = '1m',
      interval = '1d',
      from,
      to = new Date().toISOString().split('T')[0]
    } = req.query;
    
    // Determine date range based on period
    let fromDate;
    if (from) {
      fromDate = new Date(from);
    } else {
      fromDate = new Date();
      
      switch(period) {
        case '1d': fromDate.setDate(fromDate.getDate() - 1); break;
        case '1w': fromDate.setDate(fromDate.getDate() - 7); break;
        case '1m': fromDate.setDate(fromDate.getDate() - 30); break;
        case '3m': fromDate.setDate(fromDate.getDate() - 90); break;
        case '6m': fromDate.setDate(fromDate.getDate() - 180); break;
        case '1y': fromDate.setDate(fromDate.getDate() - 365); break;
        case '5y': fromDate.setDate(fromDate.getDate() - 365 * 5); break;
        default: fromDate.setDate(fromDate.getDate() - 30);
      }
    }
    
    const toDate = new Date(to);
    
    // Get stock ID
    const stock = await Stock.findOne({
      where: { symbol: symbol.toUpperCase() },
      attributes: ['id']
    });
    
    if (!stock) {
      return res.status(404).json({
        success: false,
        error: 'Stock not found',
        message: `No stock found with symbol ${symbol}`
      });
    }
    
    // Find price history
    let priceHistory;
    
    if (process.env.USE_TIMESCALEDB === 'true') {
      // Use TimescaleDB time_bucket if available for more efficient time-series queries
      priceHistory = await sequelize.query(`
        SELECT 
          time_bucket($1, timestamp) AS time,
          first(open, timestamp) AS open,
          max(high) AS high,
          min(low) AS low,
          last(close, timestamp) AS close,
          sum(volume) AS volume
        FROM "StockPrices"
        WHERE 
          symbol = $2 AND 
          timestamp >= $3 AND 
          timestamp <= $4 AND
          interval = $5
        GROUP BY time
        ORDER BY time ASC
      `, {
        bind: [
          interval === '1d' ? '1 day' : '1 hour', // time bucket size
          symbol.toUpperCase(),
          fromDate,
          toDate,
          interval
        ],
        type: sequelize.QueryTypes.SELECT
      });
    } else {
      // Standard query if TimescaleDB is not available
      priceHistory = await StockPrice.findAll({
        where: {
          symbol: symbol.toUpperCase(),
          timestamp: {
            [Op.between]: [fromDate, toDate]
          },
          interval: interval
        },
        order: [['timestamp', 'ASC']]
      });
    }
    
    res.status(200).json({
      success: true,
      symbol: symbol.toUpperCase(),
      period,
      interval,
      from: fromDate.toISOString(),
      to: toDate.toISOString(),
      data: priceHistory
    });
  } catch (error) {
    console.error('Error fetching stock history:', error);
    res.status(500).json({
      success: false,
      error: 'Server error',
      message: error.message
    });
  }
};

/**
 * Create a new stock
 * @param {Object} req - Express request object
 * @param {Object} res - Express response object
 */
export const createStock = async (req, res) => {
  try {
    const { 
      symbol, 
      name, 
      sector, 
      industry, 
      exchange, 
      currentPrice
    } = req.body;
    
    // Validate required fields
    if (!symbol || !name) {
      return res.status(400).json({
        success: false,
        error: 'Missing required fields',
        message: 'Symbol and name are required'
      });
    }
    
    // Check if stock already exists
    const existingStock = await Stock.findOne({
      where: { symbol: symbol.toUpperCase() }
    });
    
    if (existingStock) {
      return res.status(409).json({
        success: false,
        error: 'Stock already exists',
        message: `Stock with symbol ${symbol} already exists`
      });
    }
    
    // Create stock
    const stock = await Stock.create({
      symbol: symbol.toUpperCase(),
      name,
      sector,
      industry,
      exchange,
      currentPrice,
      lastUpdated: new Date()
    });
    
    res.status(201).json({
      success: true,
      data: stock
    });
  } catch (error) {
    console.error('Error creating stock:', error);
    res.status(500).json({
      success: false,
      error: 'Server error',
      message: error.message
    });
  }
};

/**
 * Update stock details
 * @param {Object} req - Express request object
 * @param {Object} res - Express response object
 */
export const updateStock = async (req, res) => {
  try {
    const { symbol } = req.params;
    const updates = req.body;
    
    // Find stock
    const stock = await Stock.findOne({
      where: { symbol: symbol.toUpperCase() }
    });
    
    if (!stock) {
      return res.status(404).json({
        success: false,
        error: 'Stock not found',
        message: `No stock found with symbol ${symbol}`
      });
    }
    
    // Update stock
    await stock.update({
      ...updates,
      lastUpdated: new Date()
    });
    
    res.status(200).json({
      success: true,
      data: stock
    });
  } catch (error) {
    console.error('Error updating stock:', error);
    res.status(500).json({
      success: false,
      error: 'Server error',
      message: error.message
    });
  }
};

/**
 * Delete a stock
 * @param {Object} req - Express request object
 * @param {Object} res - Express response object
 */
export const deleteStock = async (req, res) => {
  try {
    const { symbol } = req.params;
    
    // Find stock
    const stock = await Stock.findOne({
      where: { symbol: symbol.toUpperCase() }
    });
    
    if (!stock) {
      return res.status(404).json({
        success: false,
        error: 'Stock not found',
        message: `No stock found with symbol ${symbol}`
      });
    }
    
    // Delete stock
    await stock.destroy();
    
    res.status(200).json({
      success: true,
      message: `Stock ${symbol} deleted successfully`
    });
  } catch (error) {
    console.error('Error deleting stock:', error);
    res.status(500).json({
      success: false,
      error: 'Server error',
      message: error.message
    });
  }
}; 