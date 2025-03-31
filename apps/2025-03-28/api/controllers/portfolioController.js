const Portfolio = require('../models/Portfolio');
const Stock = require('../models/Stock');

/**
 * Get all portfolios for the current user
 * @route   GET /api/portfolios
 * @access  Private
 */
exports.getPortfolios = async (req, res) => {
  try {
    const portfolios = await Portfolio.find({ user: req.user.id })
      .populate('holdings.stock', 'symbol name price.current price.change price.changePercent');
    
    res.json(portfolios);
  } catch (err) {
    console.error(err.message);
    res.status(500).json({ message: 'Server Error' });
  }
};

/**
 * Get a single portfolio by ID
 * @route   GET /api/portfolios/:id
 * @access  Private
 */
exports.getPortfolioById = async (req, res) => {
  try {
    const portfolio = await Portfolio.findById(req.params.id)
      .populate('holdings.stock', 'symbol name price sector industry metrics analysis');
    
    // Check if portfolio exists
    if (!portfolio) {
      return res.status(404).json({ message: 'Portfolio not found' });
    }
    
    // Check if user owns the portfolio
    if (portfolio.user.toString() !== req.user.id) {
      return res.status(401).json({ message: 'Not authorized to access this portfolio' });
    }
    
    res.json(portfolio);
  } catch (err) {
    console.error(err.message);
    
    // Check if error is because of invalid ObjectId
    if (err.kind === 'ObjectId') {
      return res.status(404).json({ message: 'Portfolio not found' });
    }
    
    res.status(500).json({ message: 'Server Error' });
  }
};

/**
 * Create a new portfolio
 * @route   POST /api/portfolios
 * @access  Private
 */
exports.createPortfolio = async (req, res) => {
  try {
    const { name, description, riskProfile, holdings } = req.body;
    
    // Create a new portfolio
    const portfolio = new Portfolio({
      user: req.user.id,
      name,
      description,
      riskProfile,
      holdings: []
    });
    
    // Add holdings if provided
    if (holdings && holdings.length > 0) {
      for (const holding of holdings) {
        // Validate that stock exists
        const stockExists = await Stock.findById(holding.stock);
        if (!stockExists) {
          return res.status(404).json({ message: `Stock with ID ${holding.stock} not found` });
        }
        
        portfolio.holdings.push({
          stock: holding.stock,
          shares: holding.shares,
          purchasePrice: holding.purchasePrice,
          purchaseDate: holding.purchaseDate
        });
      }
    }
    
    await portfolio.save();
    
    res.status(201).json(portfolio);
  } catch (err) {
    console.error(err.message);
    res.status(500).json({ message: 'Server Error' });
  }
};

/**
 * Update a portfolio
 * @route   PUT /api/portfolios/:id
 * @access  Private
 */
exports.updatePortfolio = async (req, res) => {
  try {
    const { name, description, riskProfile, holdings } = req.body;
    
    // Find the portfolio
    let portfolio = await Portfolio.findById(req.params.id);
    
    // Check if portfolio exists
    if (!portfolio) {
      return res.status(404).json({ message: 'Portfolio not found' });
    }
    
    // Check if user owns the portfolio
    if (portfolio.user.toString() !== req.user.id) {
      return res.status(401).json({ message: 'Not authorized to update this portfolio' });
    }
    
    // Update basic portfolio information
    if (name) portfolio.name = name;
    if (description) portfolio.description = description;
    if (riskProfile) portfolio.riskProfile = riskProfile;
    
    // Update holdings if provided
    if (holdings && holdings.length > 0) {
      portfolio.holdings = [];
      
      for (const holding of holdings) {
        // Validate that stock exists
        const stockExists = await Stock.findById(holding.stock);
        if (!stockExists) {
          return res.status(404).json({ message: `Stock with ID ${holding.stock} not found` });
        }
        
        portfolio.holdings.push({
          stock: holding.stock,
          shares: holding.shares,
          purchasePrice: holding.purchasePrice,
          purchaseDate: holding.purchaseDate
        });
      }
    }
    
    await portfolio.save();
    
    res.json(portfolio);
  } catch (err) {
    console.error(err.message);
    
    // Check if error is because of invalid ObjectId
    if (err.kind === 'ObjectId') {
      return res.status(404).json({ message: 'Portfolio not found' });
    }
    
    res.status(500).json({ message: 'Server Error' });
  }
};

/**
 * Delete a portfolio
 * @route   DELETE /api/portfolios/:id
 * @access  Private
 */
exports.deletePortfolio = async (req, res) => {
  try {
    // Find the portfolio
    const portfolio = await Portfolio.findById(req.params.id);
    
    // Check if portfolio exists
    if (!portfolio) {
      return res.status(404).json({ message: 'Portfolio not found' });
    }
    
    // Check if user owns the portfolio
    if (portfolio.user.toString() !== req.user.id) {
      return res.status(401).json({ message: 'Not authorized to delete this portfolio' });
    }
    
    await portfolio.remove();
    
    res.json({ message: 'Portfolio removed' });
  } catch (err) {
    console.error(err.message);
    
    // Check if error is because of invalid ObjectId
    if (err.kind === 'ObjectId') {
      return res.status(404).json({ message: 'Portfolio not found' });
    }
    
    res.status(500).json({ message: 'Server Error' });
  }
};

/**
 * Add a stock to a portfolio
 * @route   POST /api/portfolios/:id/stocks
 * @access  Private
 */
exports.addStockToPortfolio = async (req, res) => {
  try {
    const { stockId, shares, purchasePrice, purchaseDate } = req.body;
    
    // Validate input
    if (!stockId || !shares || !purchasePrice) {
      return res.status(400).json({ message: 'Please provide stock ID, shares, and purchase price' });
    }
    
    // Find the portfolio
    const portfolio = await Portfolio.findById(req.params.id);
    
    // Check if portfolio exists
    if (!portfolio) {
      return res.status(404).json({ message: 'Portfolio not found' });
    }
    
    // Check if user owns the portfolio
    if (portfolio.user.toString() !== req.user.id) {
      return res.status(401).json({ message: 'Not authorized to update this portfolio' });
    }
    
    // Validate that stock exists
    const stockExists = await Stock.findById(stockId);
    if (!stockExists) {
      return res.status(404).json({ message: `Stock with ID ${stockId} not found` });
    }
    
    // Check if stock already exists in portfolio
    const stockIndex = portfolio.holdings.findIndex(
      holding => holding.stock.toString() === stockId
    );
    
    if (stockIndex !== -1) {
      // Update existing holding
      portfolio.holdings[stockIndex].shares += Number(shares);
      // Recalculate average purchase price
      const totalShares = portfolio.holdings[stockIndex].shares;
      const existingValue = (totalShares - Number(shares)) * portfolio.holdings[stockIndex].purchasePrice;
      const newValue = Number(shares) * Number(purchasePrice);
      portfolio.holdings[stockIndex].purchasePrice = (existingValue + newValue) / totalShares;
    } else {
      // Add new holding
      portfolio.holdings.push({
        stock: stockId,
        shares,
        purchasePrice,
        purchaseDate: purchaseDate || Date.now()
      });
    }
    
    await portfolio.save();
    
    res.json(portfolio);
  } catch (err) {
    console.error(err.message);
    
    // Check if error is because of invalid ObjectId
    if (err.kind === 'ObjectId') {
      return res.status(404).json({ message: 'Portfolio or stock not found' });
    }
    
    res.status(500).json({ message: 'Server Error' });
  }
};

/**
 * Remove a stock from a portfolio
 * @route   DELETE /api/portfolios/:id/stocks/:stockId
 * @access  Private
 */
exports.removeStockFromPortfolio = async (req, res) => {
  try {
    // Find the portfolio
    const portfolio = await Portfolio.findById(req.params.id);
    
    // Check if portfolio exists
    if (!portfolio) {
      return res.status(404).json({ message: 'Portfolio not found' });
    }
    
    // Check if user owns the portfolio
    if (portfolio.user.toString() !== req.user.id) {
      return res.status(401).json({ message: 'Not authorized to update this portfolio' });
    }
    
    // Find the stock holding in the portfolio
    const stockIndex = portfolio.holdings.findIndex(
      holding => holding.stock.toString() === req.params.stockId
    );
    
    if (stockIndex === -1) {
      return res.status(404).json({ message: 'Stock not found in this portfolio' });
    }
    
    // Remove the stock
    portfolio.holdings.splice(stockIndex, 1);
    
    await portfolio.save();
    
    res.json(portfolio);
  } catch (err) {
    console.error(err.message);
    
    // Check if error is because of invalid ObjectId
    if (err.kind === 'ObjectId') {
      return res.status(404).json({ message: 'Portfolio or stock not found' });
    }
    
    res.status(500).json({ message: 'Server Error' });
  }
}; 