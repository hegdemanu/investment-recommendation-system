const SentimentAnalysis = require('../models/SentimentAnalysis');
const Stock = require('../models/Stock');

/**
 * Sentiment Analysis Service - Handles sentiment analysis of financial texts
 * In a production environment, this would integrate with actual NLP models
 */
class SentimentService {
  /**
   * Analyze sentiment of a text for a specific stock
   * @param {String} stockId - Stock ID
   * @param {Object} data - Sentiment data
   * @returns {Promise<Object>} Sentiment analysis result
   */
  async analyzeSentiment(stockId, data) {
    try {
      // Verify stock exists
      const stock = await Stock.findById(stockId);
      if (!stock) {
        throw new Error('Stock not found');
      }
      
      // This would typically call an NLP service like FinBERT
      // For now, we'll simulate sentiment analysis results
      
      // Extract or create required fields
      const content = data.content || '';
      const source = data.source || 'Other';
      
      // Mock sentiment processing
      // In a real implementation, this would use an actual NLP model
      const sentiment = this._mockAnalyzeSentiment(content, stock.symbol);
      
      // Create sentiment analysis record
      const sentimentAnalysis = new SentimentAnalysis({
        stock: stockId,
        source: source,
        sourceUrl: data.sourceUrl || '',
        sourceTitle: data.sourceTitle || '',
        content: content,
        language: data.language || 'en',
        model: 'FinBERT', // Would be configurable in actual implementation
        sentiment: sentiment.dominant,
        score: {
          bullish: sentiment.scores.bullish,
          neutral: sentiment.scores.neutral,
          bearish: sentiment.scores.bearish
        },
        confidence: sentiment.confidence,
        keywords: sentiment.keywords,
        entities: sentiment.entities,
        impactWeight: data.impactWeight || 1.0,
        publishedAt: data.publishedAt || new Date(),
        validUntil: data.validUntil || null
      });
      
      await sentimentAnalysis.save();
      return sentimentAnalysis;
    } catch (error) {
      throw new Error(`Error analyzing sentiment: ${error.message}`);
    }
  }

  /**
   * Analyze sentiment in batch for multiple texts
   * @param {String} stockId - Stock ID
   * @param {Array} texts - Array of text objects
   * @returns {Promise<Array>} Sentiment analysis results
   */
  async analyzeBatch(stockId, texts) {
    try {
      const results = [];
      
      for (const text of texts) {
        try {
          const result = await this.analyzeSentiment(stockId, text);
          results.push(result);
        } catch (error) {
          console.error(`Error analyzing text: ${error.message}`);
          // Continue with next text
        }
      }
      
      return results;
    } catch (error) {
      throw new Error(`Error in batch sentiment analysis: ${error.message}`);
    }
  }

  /**
   * Get sentiment analysis for a stock
   * @param {String} stockId - Stock ID
   * @param {Object} options - Query options
   * @returns {Promise<Array>} Sentiment analysis results
   */
  async getSentimentForStock(stockId, options = {}) {
    try {
      const query = { stock: stockId };
      
      // Apply filters if provided
      if (options.source) {
        query.source = options.source;
      }
      
      if (options.dateFrom) {
        query.publishedAt = { ...query.publishedAt, $gte: new Date(options.dateFrom) };
      }
      
      if (options.dateTo) {
        query.publishedAt = { ...query.publishedAt, $lte: new Date(options.dateTo) };
      }
      
      if (options.sentiment) {
        query.sentiment = options.sentiment;
      }
      
      // Apply sorting
      const sortField = options.sortBy || 'publishedAt';
      const sortOrder = options.sortOrder === 'asc' ? 1 : -1;
      
      // Apply pagination
      const limit = options.limit || 20;
      const skip = options.page ? (options.page - 1) * limit : 0;
      
      const sentiments = await SentimentAnalysis.find(query)
        .sort({ [sortField]: sortOrder })
        .skip(skip)
        .limit(limit);
      
      return sentiments;
    } catch (error) {
      throw new Error(`Error fetching sentiment data: ${error.message}`);
    }
  }

  /**
   * Get aggregate sentiment for a stock
   * @param {String} stockId - Stock ID
   * @param {Number} timeframe - Timeframe in days
   * @returns {Promise<Object>} Aggregate sentiment
   */
  async getAggregateSentiment(stockId, timeframe = 7) {
    try {
      return await SentimentAnalysis.calculateAggregateSentiment(stockId, timeframe);
    } catch (error) {
      throw new Error(`Error calculating aggregate sentiment: ${error.message}`);
    }
  }

  /**
   * Delete sentiment analysis
   * @param {String} sentimentId - Sentiment analysis ID
   * @returns {Promise<Boolean>} Success status
   */
  async deleteSentiment(sentimentId) {
    try {
      const result = await SentimentAnalysis.findByIdAndDelete(sentimentId);
      return !!result;
    } catch (error) {
      throw new Error(`Error deleting sentiment analysis: ${error.message}`);
    }
  }

  /**
   * Mock sentiment analysis function
   * @param {String} text - Text to analyze
   * @param {String} symbol - Stock symbol
   * @returns {Object} Mocked sentiment analysis result
   * @private
   */
  _mockAnalyzeSentiment(text, symbol) {
    // This is a very basic sentiment mockup
    // In a real implementation, this would use FinBERT or similar NLP model
    
    // Lowercase for simple word matching
    const lowerText = text.toLowerCase();
    
    // Basic keyword detection
    const bullishWords = ['growth', 'profit', 'increase', 'positive', 'uptrend', 'bullish', 'outperform'];
    const bearishWords = ['decline', 'loss', 'decrease', 'negative', 'downtrend', 'bearish', 'underperform'];
    
    // Count occurrences of sentiment words
    let bullishCount = 0;
    let bearishCount = 0;
    
    bullishWords.forEach(word => {
      const regex = new RegExp(`\\b${word}\\b`, 'gi');
      const matches = lowerText.match(regex);
      if (matches) bullishCount += matches.length;
    });
    
    bearishWords.forEach(word => {
      const regex = new RegExp(`\\b${word}\\b`, 'gi');
      const matches = lowerText.match(regex);
      if (matches) bearishCount += matches.length;
    });
    
    // Calculate total sentiment words
    const totalWords = bullishCount + bearishCount;
    
    // Calculate scores
    let bullishScore = 0;
    let bearishScore = 0;
    let neutralScore = 0.5; // Default neutral
    
    if (totalWords > 0) {
      bullishScore = bullishCount / (totalWords * 2);
      bearishScore = bearishCount / (totalWords * 2);
      neutralScore = 1 - (bullishScore + bearishScore);
    }
    
    // Determine dominant sentiment
    let dominant = 'Neutral';
    let confidence = neutralScore;
    
    if (bullishScore > neutralScore && bullishScore > bearishScore) {
      dominant = 'Bullish';
      confidence = bullishScore;
    } else if (bearishScore > neutralScore && bearishScore > bullishScore) {
      dominant = 'Bearish';
      confidence = bearishScore;
    }
    
    // Extract simple keywords
    const words = lowerText.split(/\W+/).filter(word => word.length > 3);
    const keywords = [...new Set(words)].slice(0, 5); // Unique words, limit to 5
    
    // Mock entity extraction
    const entities = [];
    if (lowerText.includes(symbol.toLowerCase())) {
      entities.push({
        name: symbol,
        type: 'Company',
        sentiment: dominant
      });
    }
    
    // Detect potential people
    const potentialPeople = lowerText.match(/[A-Z][a-z]+ [A-Z][a-z]+/g);
    if (potentialPeople) {
      potentialPeople.slice(0, 2).forEach(person => {
        entities.push({
          name: person,
          type: 'Person',
          sentiment: 'Neutral'
        });
      });
    }
    
    return {
      dominant,
      confidence,
      scores: {
        bullish: bullishScore,
        neutral: neutralScore,
        bearish: bearishScore
      },
      keywords,
      entities
    };
  }
}

module.exports = new SentimentService(); 