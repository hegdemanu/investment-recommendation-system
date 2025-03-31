const sentimentService = require('../services/sentimentService');
const Stock = require('../models/Stock');

/**
 * Sentiment Controller - Handles API endpoints for sentiment analysis
 */

/**
 * Analyze sentiment for a stock
 * @route POST /api/sentiment/:stockId
 * @access Private
 */
exports.analyzeSentiment = async (req, res) => {
  try {
    const { stockId } = req.params;
    
    // Verify stock exists
    const stockExists = await Stock.exists({ _id: stockId });
    if (!stockExists) {
      return res.status(404).json({ message: 'Stock not found' });
    }
    
    const result = await sentimentService.analyzeSentiment(stockId, req.body);
    
    res.status(201).json(result);
  } catch (error) {
    console.error('Error in analyzeSentiment:', error);
    res.status(500).json({ message: error.message });
  }
};

/**
 * Analyze sentiment in batch for a stock
 * @route POST /api/sentiment/:stockId/batch
 * @access Private
 */
exports.analyzeBatch = async (req, res) => {
  try {
    const { stockId } = req.params;
    
    // Verify stock exists
    const stockExists = await Stock.exists({ _id: stockId });
    if (!stockExists) {
      return res.status(404).json({ message: 'Stock not found' });
    }
    
    if (!Array.isArray(req.body)) {
      return res.status(400).json({ message: 'Request body must be an array of texts to analyze' });
    }
    
    const results = await sentimentService.analyzeBatch(stockId, req.body);
    
    res.status(201).json(results);
  } catch (error) {
    console.error('Error in analyzeBatch:', error);
    res.status(500).json({ message: error.message });
  }
};

/**
 * Get sentiment analysis for a stock
 * @route GET /api/sentiment/:stockId
 * @access Private
 */
exports.getSentimentForStock = async (req, res) => {
  try {
    const { stockId } = req.params;
    
    // Verify stock exists
    const stockExists = await Stock.exists({ _id: stockId });
    if (!stockExists) {
      return res.status(404).json({ message: 'Stock not found' });
    }
    
    // Parse query parameters
    const options = {
      source: req.query.source,
      dateFrom: req.query.dateFrom,
      dateTo: req.query.dateTo,
      sentiment: req.query.sentiment,
      sortBy: req.query.sortBy || 'publishedAt',
      sortOrder: req.query.sortOrder || 'desc',
      page: parseInt(req.query.page) || 1,
      limit: parseInt(req.query.limit) || 20
    };
    
    const sentiments = await sentimentService.getSentimentForStock(stockId, options);
    
    res.json(sentiments);
  } catch (error) {
    console.error('Error in getSentimentForStock:', error);
    res.status(500).json({ message: error.message });
  }
};

/**
 * Get aggregate sentiment for a stock
 * @route GET /api/sentiment/:stockId/aggregate
 * @access Private
 */
exports.getAggregateSentiment = async (req, res) => {
  try {
    const { stockId } = req.params;
    
    // Verify stock exists
    const stockExists = await Stock.exists({ _id: stockId });
    if (!stockExists) {
      return res.status(404).json({ message: 'Stock not found' });
    }
    
    // Parse timeframe parameter
    const timeframe = parseInt(req.query.timeframe) || 7;
    
    const aggregateSentiment = await sentimentService.getAggregateSentiment(stockId, timeframe);
    
    res.json(aggregateSentiment);
  } catch (error) {
    console.error('Error in getAggregateSentiment:', error);
    res.status(500).json({ message: error.message });
  }
};

/**
 * Delete sentiment analysis
 * @route DELETE /api/sentiment/:id
 * @access Private
 */
exports.deleteSentiment = async (req, res) => {
  try {
    const success = await sentimentService.deleteSentiment(req.params.id);
    
    if (!success) {
      return res.status(404).json({ message: 'Sentiment analysis not found' });
    }
    
    res.json({ message: 'Sentiment analysis deleted successfully' });
  } catch (error) {
    console.error('Error in deleteSentiment:', error);
    res.status(500).json({ message: error.message });
  }
};

/**
 * Analyze news article for a stock
 * @route POST /api/sentiment/:stockId/news
 * @access Private
 */
exports.analyzeNewsArticle = async (req, res) => {
  try {
    const { stockId } = req.params;
    
    // Verify stock exists
    const stockExists = await Stock.exists({ _id: stockId });
    if (!stockExists) {
      return res.status(404).json({ message: 'Stock not found' });
    }
    
    // Validate required fields
    if (!req.body.title || !req.body.content) {
      return res.status(400).json({ message: 'Title and content are required' });
    }
    
    // Prepare the data
    const newsData = {
      source: 'News',
      sourceTitle: req.body.title,
      sourceUrl: req.body.url,
      content: req.body.content,
      publishedAt: req.body.publishedAt || new Date(),
      impactWeight: req.body.impactWeight || 1.0
    };
    
    const result = await sentimentService.analyzeSentiment(stockId, newsData);
    
    res.status(201).json(result);
  } catch (error) {
    console.error('Error in analyzeNewsArticle:', error);
    res.status(500).json({ message: error.message });
  }
};

/**
 * Analyze earnings report for a stock
 * @route POST /api/sentiment/:stockId/earnings
 * @access Private
 */
exports.analyzeEarningsReport = async (req, res) => {
  try {
    const { stockId } = req.params;
    
    // Verify stock exists
    const stockExists = await Stock.exists({ _id: stockId });
    if (!stockExists) {
      return res.status(404).json({ message: 'Stock not found' });
    }
    
    // Validate required fields
    if (!req.body.content) {
      return res.status(400).json({ message: 'Report content is required' });
    }
    
    // Prepare the data
    const reportData = {
      source: 'EarningsReport',
      sourceTitle: req.body.title || 'Earnings Report',
      sourceUrl: req.body.url,
      content: req.body.content,
      publishedAt: req.body.publishedAt || new Date(),
      impactWeight: req.body.impactWeight || 2.0 // Higher weight for earnings reports
    };
    
    const result = await sentimentService.analyzeSentiment(stockId, reportData);
    
    res.status(201).json(result);
  } catch (error) {
    console.error('Error in analyzeEarningsReport:', error);
    res.status(500).json({ message: error.message });
  }
};

/**
 * Get sentiment analysis counts by source
 * @route GET /api/sentiment/:stockId/sources
 * @access Private
 */
exports.getSentimentCountBySource = async (req, res) => {
  try {
    const { stockId } = req.params;
    
    // Verify stock exists
    const stockExists = await Stock.exists({ _id: stockId });
    if (!stockExists) {
      return res.status(404).json({ message: 'Stock not found' });
    }
    
    // Use MongoDB aggregation to count sentiments by source
    const SentimentAnalysis = require('../models/SentimentAnalysis');
    const results = await SentimentAnalysis.aggregate([
      { $match: { stock: require('mongoose').Types.ObjectId(stockId) } },
      { $group: { _id: '$source', count: { $sum: 1 } } },
      { $sort: { count: -1 } }
    ]);
    
    // Format results
    const formattedResults = results.map(result => ({
      source: result._id,
      count: result.count
    }));
    
    res.json(formattedResults);
  } catch (error) {
    console.error('Error in getSentimentCountBySource:', error);
    res.status(500).json({ message: error.message });
  }
}; 