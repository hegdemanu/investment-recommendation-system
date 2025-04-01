const mongoose = require('mongoose');

const SentimentAnalysisSchema = new mongoose.Schema({
  stock: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'Stock',
    required: true
  },
  source: {
    type: String,
    required: [true, 'Please provide a source for the sentiment data'],
    enum: ['News', 'TwitterSocialMedia', 'EarningsReport', 'SECFiling', 'AnalystReport', 'PressRelease', 'Other'],
    trim: true
  },
  sourceUrl: {
    type: String,
    trim: true
  },
  sourceTitle: {
    type: String,
    trim: true
  },
  content: {
    type: String,
    trim: true
  },
  language: {
    type: String,
    default: 'en',
    trim: true
  },
  model: {
    type: String,
    default: 'FinBERT',
    trim: true
  },
  sentiment: {
    type: String,
    required: true,
    enum: ['Bullish', 'Neutral', 'Bearish'],
    default: 'Neutral'
  },
  score: {
    bullish: {
      type: Number,
      default: 0,
      min: 0,
      max: 1
    },
    neutral: {
      type: Number,
      default: 0,
      min: 0,
      max: 1
    },
    bearish: {
      type: Number,
      default: 0,
      min: 0,
      max: 1
    }
  },
  confidence: {
    type: Number,
    default: 0,
    min: 0,
    max: 1
  },
  keywords: [String],
  entities: [{
    name: String,
    type: {
      type: String,
      enum: ['Company', 'Person', 'Product', 'Event', 'Other']
    },
    sentiment: {
      type: String,
      enum: ['Bullish', 'Neutral', 'Bearish'],
      default: 'Neutral'
    }
  }],
  impactWeight: {
    type: Number,
    default: 1.0,
    min: 0,
    max: 5.0
  },
  publishedAt: {
    type: Date
  },
  analyzedAt: {
    type: Date,
    default: Date.now
  },
  validUntil: {
    type: Date
  },
  createdAt: {
    type: Date,
    default: Date.now
  },
  updatedAt: {
    type: Date,
    default: Date.now
  }
});

// Update the updatedAt field before saving
SentimentAnalysisSchema.pre('save', function(next) {
  this.updatedAt = Date.now();
  next();
});

// Set the dominant sentiment based on scores
SentimentAnalysisSchema.pre('save', function(next) {
  const { bullish, neutral, bearish } = this.score;
  
  if (bullish > neutral && bullish > bearish) {
    this.sentiment = 'Bullish';
    this.confidence = bullish;
  } else if (bearish > neutral && bearish > bullish) {
    this.sentiment = 'Bearish';
    this.confidence = bearish;
  } else {
    this.sentiment = 'Neutral';
    this.confidence = neutral;
  }
  
  next();
});

// Static method to find latest sentiment for a stock
SentimentAnalysisSchema.statics.findLatestForStock = function(stockId, limit = 10) {
  return this.find({ 
    stock: stockId 
  })
  .sort({ analyzedAt: -1 })
  .limit(limit);
};

// Static method to calculate aggregate sentiment for a stock
SentimentAnalysisSchema.statics.calculateAggregateSentiment = async function(stockId, timeframe = 7) {
  const cutoffDate = new Date();
  cutoffDate.setDate(cutoffDate.getDate() - timeframe);
  
  const sentiments = await this.find({
    stock: stockId,
    analyzedAt: { $gte: cutoffDate }
  });
  
  if (sentiments.length === 0) return { sentiment: 'Neutral', confidence: 0 };
  
  let totalBullish = 0;
  let totalBearish = 0;
  let totalNeutral = 0;
  
  sentiments.forEach(item => {
    totalBullish += item.score.bullish * item.impactWeight;
    totalBearish += item.score.bearish * item.impactWeight;
    totalNeutral += item.score.neutral * item.impactWeight;
  });
  
  const total = totalBullish + totalBearish + totalNeutral;
  
  if (total === 0) return { sentiment: 'Neutral', confidence: 0 };
  
  const bullishRatio = totalBullish / total;
  const bearishRatio = totalBearish / total;
  const neutralRatio = totalNeutral / total;
  
  let dominantSentiment = 'Neutral';
  let confidence = neutralRatio;
  
  if (bullishRatio > neutralRatio && bullishRatio > bearishRatio) {
    dominantSentiment = 'Bullish';
    confidence = bullishRatio;
  } else if (bearishRatio > neutralRatio && bearishRatio > bullishRatio) {
    dominantSentiment = 'Bearish';
    confidence = bearishRatio;
  }
  
  return {
    sentiment: dominantSentiment,
    confidence,
    distribution: {
      bullish: bullishRatio,
      neutral: neutralRatio,
      bearish: bearishRatio
    },
    sampleSize: sentiments.length
  };
};

// Indexes for efficient querying
SentimentAnalysisSchema.index({ stock: 1, analyzedAt: -1 });
SentimentAnalysisSchema.index({ sentiment: 1, confidence: -1 });

module.exports = mongoose.model('SentimentAnalysis', SentimentAnalysisSchema); 