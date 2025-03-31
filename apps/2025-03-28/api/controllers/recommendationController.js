const Recommendation = require('../models/Recommendation');
const Stock = require('../models/Stock');
const Portfolio = require('../models/Portfolio');

/**
 * Get all recommendations for the current user
 * @route   GET /api/recommendations
 * @access  Private
 */
exports.getRecommendations = async (req, res) => {
  try {
    const recommendations = await Recommendation.find({ user: req.user.id })
      .populate('stock', 'symbol name price.current price.change price.changePercent')
      .sort({ createdAt: -1 });
    
    res.json(recommendations);
  } catch (err) {
    console.error(err.message);
    res.status(500).json({ message: 'Server Error' });
  }
};

/**
 * Get recommendations filtered by status
 * @route   GET /api/recommendations/status/:status
 * @access  Private
 */
exports.getRecommendationsByStatus = async (req, res) => {
  try {
    const { status } = req.params;
    
    // Validate status
    if (!['Active', 'Expired', 'Fulfilled'].includes(status)) {
      return res.status(400).json({ message: 'Invalid status parameter' });
    }
    
    const recommendations = await Recommendation.find({ 
      user: req.user.id,
      status
    })
      .populate('stock', 'symbol name price.current price.change price.changePercent')
      .sort({ createdAt: -1 });
    
    res.json(recommendations);
  } catch (err) {
    console.error(err.message);
    res.status(500).json({ message: 'Server Error' });
  }
};

/**
 * Get a single recommendation by ID
 * @route   GET /api/recommendations/:id
 * @access  Private
 */
exports.getRecommendationById = async (req, res) => {
  try {
    const recommendation = await Recommendation.findById(req.params.id)
      .populate('stock', 'symbol name price sector industry metrics analysis');
    
    // Check if recommendation exists
    if (!recommendation) {
      return res.status(404).json({ message: 'Recommendation not found' });
    }
    
    // Check if user owns the recommendation
    if (recommendation.user.toString() !== req.user.id) {
      return res.status(401).json({ message: 'Not authorized to access this recommendation' });
    }
    
    res.json(recommendation);
  } catch (err) {
    console.error(err.message);
    
    // Check if error is because of invalid ObjectId
    if (err.kind === 'ObjectId') {
      return res.status(404).json({ message: 'Recommendation not found' });
    }
    
    res.status(500).json({ message: 'Server Error' });
  }
};

/**
 * Get recommendations for a specific stock
 * @route   GET /api/recommendations/stock/:stockId
 * @access  Private
 */
exports.getRecommendationsByStock = async (req, res) => {
  try {
    const recommendations = await Recommendation.find({ 
      user: req.user.id,
      stock: req.params.stockId
    })
      .populate('stock', 'symbol name price.current price.change price.changePercent')
      .sort({ createdAt: -1 });
    
    res.json(recommendations);
  } catch (err) {
    console.error(err.message);
    
    // Check if error is because of invalid ObjectId
    if (err.kind === 'ObjectId') {
      return res.status(404).json({ message: 'Stock not found' });
    }
    
    res.status(500).json({ message: 'Server Error' });
  }
};

/**
 * Create a new recommendation
 * @route   POST /api/recommendations
 * @access  Private
 */
exports.createRecommendation = async (req, res) => {
  try {
    const { 
      stockId, 
      title, 
      description, 
      action, 
      targetPrice, 
      potentialReturn, 
      riskLevel, 
      timeHorizon, 
      rationale, 
      targetDate 
    } = req.body;
    
    // Validate input
    if (!stockId || !title || !description || !action || !riskLevel || !timeHorizon || !rationale) {
      return res.status(400).json({ message: 'Please provide all required fields' });
    }
    
    // Verify stock exists
    const stock = await Stock.findById(stockId);
    if (!stock) {
      return res.status(404).json({ message: 'Stock not found' });
    }
    
    // Create new recommendation
    const recommendation = new Recommendation({
      user: req.user.id,
      stock: stockId,
      title,
      description,
      action,
      targetPrice,
      potentialReturn,
      riskLevel,
      timeHorizon,
      rationale,
      targetDate
    });
    
    await recommendation.save();
    
    // Populate stock details for response
    await recommendation.populate('stock', 'symbol name price.current');
    
    res.status(201).json(recommendation);
  } catch (err) {
    console.error(err.message);
    res.status(500).json({ message: 'Server Error' });
  }
};

/**
 * Update a recommendation
 * @route   PUT /api/recommendations/:id
 * @access  Private
 */
exports.updateRecommendation = async (req, res) => {
  try {
    const { 
      title, 
      description, 
      action, 
      targetPrice, 
      potentialReturn, 
      riskLevel, 
      timeHorizon, 
      rationale, 
      status,
      targetDate,
      accuracy
    } = req.body;
    
    // Find the recommendation
    let recommendation = await Recommendation.findById(req.params.id);
    
    // Check if recommendation exists
    if (!recommendation) {
      return res.status(404).json({ message: 'Recommendation not found' });
    }
    
    // Check if user owns the recommendation
    if (recommendation.user.toString() !== req.user.id) {
      return res.status(401).json({ message: 'Not authorized to update this recommendation' });
    }
    
    // Update fields if provided
    if (title) recommendation.title = title;
    if (description) recommendation.description = description;
    if (action) recommendation.action = action;
    if (targetPrice !== undefined) recommendation.targetPrice = targetPrice;
    if (potentialReturn !== undefined) recommendation.potentialReturn = potentialReturn;
    if (riskLevel) recommendation.riskLevel = riskLevel;
    if (timeHorizon) recommendation.timeHorizon = timeHorizon;
    if (rationale) recommendation.rationale = rationale;
    if (status) recommendation.status = status;
    if (targetDate) recommendation.targetDate = targetDate;
    if (accuracy !== undefined) recommendation.accuracy = accuracy;
    
    await recommendation.save();
    
    // Populate stock details for response
    await recommendation.populate('stock', 'symbol name price.current');
    
    res.json(recommendation);
  } catch (err) {
    console.error(err.message);
    
    // Check if error is because of invalid ObjectId
    if (err.kind === 'ObjectId') {
      return res.status(404).json({ message: 'Recommendation not found' });
    }
    
    res.status(500).json({ message: 'Server Error' });
  }
};

/**
 * Delete a recommendation
 * @route   DELETE /api/recommendations/:id
 * @access  Private
 */
exports.deleteRecommendation = async (req, res) => {
  try {
    // Find the recommendation
    const recommendation = await Recommendation.findById(req.params.id);
    
    // Check if recommendation exists
    if (!recommendation) {
      return res.status(404).json({ message: 'Recommendation not found' });
    }
    
    // Check if user owns the recommendation
    if (recommendation.user.toString() !== req.user.id) {
      return res.status(401).json({ message: 'Not authorized to delete this recommendation' });
    }
    
    await recommendation.remove();
    
    res.json({ message: 'Recommendation removed' });
  } catch (err) {
    console.error(err.message);
    
    // Check if error is because of invalid ObjectId
    if (err.kind === 'ObjectId') {
      return res.status(404).json({ message: 'Recommendation not found' });
    }
    
    res.status(500).json({ message: 'Server Error' });
  }
};

/**
 * Generate personalized recommendations based on user portfolio
 * @route   GET /api/recommendations/generate
 * @access  Private
 */
exports.generateRecommendations = async (req, res) => {
  try {
    // Get user's portfolios
    const portfolios = await Portfolio.find({ user: req.user.id })
      .populate('holdings.stock');
    
    if (portfolios.length === 0) {
      return res.status(400).json({ 
        message: 'No portfolios found. Create a portfolio first to get personalized recommendations.' 
      });
    }
    
    // Get all stocks
    const allStocks = await Stock.find({});
    
    // Create a set of stocks already in user's portfolios
    const userStocks = new Set();
    portfolios.forEach(portfolio => {
      portfolio.holdings.forEach(holding => {
        if (holding.stock) {
          userStocks.add(holding.stock._id.toString());
        }
      });
    });
    
    // Filter stocks not in user's portfolios
    const candidateStocks = allStocks.filter(stock => !userStocks.has(stock._id.toString()));
    
    // If no candidate stocks, return message
    if (candidateStocks.length === 0) {
      return res.status(400).json({ 
        message: 'You already own all available stocks in our database.' 
      });
    }
    
    // Determine risk profile from portfolios
    let riskProfile = 'Moderate';
    if (portfolios.length > 0) {
      const riskScores = {
        'Conservative': 1,
        'Moderate': 2,
        'Aggressive': 3
      };
      
      let totalScore = 0;
      portfolios.forEach(portfolio => {
        totalScore += riskScores[portfolio.riskProfile] || 2;
      });
      
      const avgScore = totalScore / portfolios.length;
      
      if (avgScore <= 1.5) {
        riskProfile = 'Conservative';
      } else if (avgScore >= 2.5) {
        riskProfile = 'Aggressive';
      }
    }
    
    // Sort candidate stocks by recommendation strength
    candidateStocks.sort((a, b) => {
      // Sort logic based on analysis data and risk profile
      const aScore = getRecommendationScore(a, riskProfile);
      const bScore = getRecommendationScore(b, riskProfile);
      return bScore - aScore; // Higher score first
    });
    
    // Take top 5 stocks as recommendations
    const topStocks = candidateStocks.slice(0, 5);
    
    // Create recommendation objects (not saved to DB)
    const recommendations = topStocks.map(stock => {
      // Generate appropriate action, target price, and other details based on stock data
      const action = 'Buy';
      const targetPrice = Math.round((stock.price.current * 1.1) * 100) / 100; // 10% higher
      const potentialReturn = Math.round(((targetPrice / stock.price.current - 1) * 100) * 10) / 10;
      
      // Map the stock's recommendation to our action
      let stockRiskLevel = 'Medium';
      if (stock.analysis && stock.analysis.riskLevel) {
        stockRiskLevel = stock.analysis.riskLevel;
      }
      
      let timeHorizon = 'Medium-term';
      if (riskProfile === 'Conservative') {
        timeHorizon = 'Long-term';
      } else if (riskProfile === 'Aggressive') {
        timeHorizon = 'Short-term';
      }
      
      return {
        stock: {
          _id: stock._id,
          symbol: stock.symbol,
          name: stock.name,
          price: stock.price,
          sector: stock.sector,
          industry: stock.industry
        },
        title: `${action} ${stock.name} (${stock.symbol})`,
        description: `Recommended ${action.toLowerCase()} based on your portfolio profile and market analysis.`,
        action,
        targetPrice,
        potentialReturn,
        riskLevel: stockRiskLevel,
        timeHorizon,
        rationale: generateRationale(stock, action, riskProfile),
        status: 'Active',
        createdAt: new Date()
      };
    });
    
    res.json(recommendations);
  } catch (err) {
    console.error(err.message);
    res.status(500).json({ message: 'Server Error' });
  }
};

/**
 * Helper to calculate recommendation score for sorting
 * @param {Object} stock - Stock object
 * @param {String} riskProfile - User risk profile
 * @returns {Number} - Score for sorting
 */
function getRecommendationScore(stock, riskProfile) {
  let score = 0;
  
  // Base score from recommendation analysis
  if (stock.analysis && stock.analysis.recommendation) {
    const recMap = {
      'Strong Buy': 5,
      'Buy': 4,
      'Hold': 2,
      'Sell': 0,
      'Strong Sell': 0
    };
    score += recMap[stock.analysis.recommendation] || 0;
  }
  
  // Adjust score based on risk profile
  if (stock.analysis && stock.analysis.riskLevel && riskProfile) {
    // Risk profile match adds points
    const stockRisk = stock.analysis.riskLevel;
    
    // Align stock risk with portfolio risk preference
    if (riskProfile === 'Conservative' && stockRisk === 'Low') {
      score += 3;
    } else if (riskProfile === 'Moderate' && stockRisk === 'Medium') {
      score += 3;
    } else if (riskProfile === 'Aggressive' && stockRisk === 'High') {
      score += 3;
    }
  }
  
  // Additional scoring factors
  if (stock.metrics) {
    // Positive EPS adds points
    if (stock.metrics.eps > 0) {
      score += 1;
    }
    
    // Dividend paying stocks get a bonus
    if (stock.metrics.dividendYield > 0) {
      score += 1;
    }
  }
  
  // Price momentum factor
  if (stock.price && stock.price.change) {
    if (stock.price.change > 0) {
      score += 1;
    }
  }
  
  return score;
}

/**
 * Generate rationale text for recommendation
 * @param {Object} stock - Stock object
 * @param {String} action - Buy/Sell/Hold
 * @param {String} riskProfile - User risk profile
 * @returns {String} - Rationale text
 */
function generateRationale(stock, action, riskProfile) {
  let rationale = '';
  
  // Generate rationale based on action and stock attributes
  if (action === 'Buy') {
    // Base rationale on stock metrics
    rationale = `${stock.name} aligns with your ${riskProfile.toLowerCase()} risk profile.`;
    
    if (stock.metrics) {
      // Add metrics-based reasoning
      if (stock.metrics.pe && stock.metrics.pe < 20) {
        rationale += ` The company has an attractive P/E ratio of ${stock.metrics.pe}.`;
      }
      
      if (stock.metrics.dividendYield && stock.metrics.dividendYield > 0) {
        rationale += ` It offers a dividend yield of ${stock.metrics.dividendYield}%.`;
      }
      
      if (stock.metrics.beta) {
        if (stock.metrics.beta < 1) {
          rationale += ` With a beta of ${stock.metrics.beta}, the stock is less volatile than the market.`;
        } else if (stock.metrics.beta > 1.5) {
          rationale += ` With a beta of ${stock.metrics.beta}, the stock offers higher growth potential with increased volatility.`;
        }
      }
    }
    
    // Add sector-specific reasoning
    if (stock.sector) {
      rationale += ` ${stock.name} operates in the ${stock.sector} sector`;
      if (stock.industry) {
        rationale += ` within the ${stock.industry} industry`;
      }
      rationale += '.';
    }
    
    // Add recommendation consensus if available
    if (stock.analysis && stock.analysis.recommendation) {
      rationale += ` Market consensus is "${stock.analysis.recommendation}" for this stock.`;
    }
  }
  
  return rationale;
} 