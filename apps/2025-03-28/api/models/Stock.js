const mongoose = require('mongoose');

const StockSchema = new mongoose.Schema({
  symbol: {
    type: String,
    required: [true, 'Please provide a stock symbol'],
    unique: true,
    trim: true,
    uppercase: true
  },
  name: {
    type: String,
    required: [true, 'Please provide a company name'],
    trim: true
  },
  sector: {
    type: String,
    trim: true
  },
  industry: {
    type: String,
    trim: true
  },
  price: {
    current: {
      type: Number,
      required: [true, 'Please provide current price']
    },
    change: {
      type: Number,
      default: 0
    },
    changePercent: {
      type: Number,
      default: 0
    },
    history: [
      {
        date: {
          type: Date,
          required: true
        },
        price: {
          type: Number,
          required: true
        }
      }
    ]
  },
  marketCap: {
    type: Number
  },
  metrics: {
    pe: Number,
    eps: Number,
    dividend: Number,
    dividendYield: Number,
    beta: Number,
    dayRange: {
      low: Number,
      high: Number
    },
    yearRange: {
      low: Number,
      high: Number
    },
    volume: Number,
    avgVolume: Number
  },
  analysis: {
    recommendation: {
      type: String,
      enum: ['Strong Buy', 'Buy', 'Hold', 'Sell', 'Strong Sell'],
      default: 'Hold'
    },
    targetPrice: Number,
    riskLevel: {
      type: String,
      enum: ['Low', 'Medium', 'High'],
      default: 'Medium'
    },
    potentialReturn: Number
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
StockSchema.pre('save', function(next) {
  this.updatedAt = Date.now();
  next();
});

module.exports = mongoose.model('Stock', StockSchema); 