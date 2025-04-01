const mongoose = require('mongoose');

const PortfolioSchema = new mongoose.Schema({
  user: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'User',
    required: true
  },
  name: {
    type: String,
    required: [true, 'Please provide a portfolio name'],
    trim: true,
    maxlength: [50, 'Name cannot be more than 50 characters']
  },
  description: {
    type: String,
    trim: true,
    maxlength: [200, 'Description cannot be more than 200 characters']
  },
  holdings: [
    {
      stock: {
        type: mongoose.Schema.Types.ObjectId,
        ref: 'Stock',
        required: true
      },
      shares: {
        type: Number,
        required: [true, 'Please provide number of shares'],
        min: [0.001, 'Shares must be a positive number']
      },
      purchasePrice: {
        type: Number,
        required: [true, 'Please provide purchase price']
      },
      purchaseDate: {
        type: Date,
        default: Date.now
      }
    }
  ],
  totalValue: {
    type: Number,
    default: 0
  },
  performance: {
    day: {
      change: Number,
      changePercent: Number
    },
    week: {
      change: Number,
      changePercent: Number
    },
    month: {
      change: Number,
      changePercent: Number
    },
    year: {
      change: Number,
      changePercent: Number
    },
    allTime: {
      change: Number,
      changePercent: Number
    }
  },
  riskProfile: {
    type: String,
    enum: ['Conservative', 'Moderate', 'Aggressive'],
    default: 'Moderate'
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
PortfolioSchema.pre('save', function(next) {
  this.updatedAt = Date.now();
  next();
});

// Calculate total value before saving
PortfolioSchema.pre('save', async function(next) {
  try {
    let totalValue = 0;
    
    // Populate the stock references to get current prices
    await this.populate('holdings.stock');
    
    // Calculate total value based on current stock prices
    for (const holding of this.holdings) {
      if (holding.stock && holding.stock.price && holding.stock.price.current) {
        totalValue += holding.shares * holding.stock.price.current;
      }
    }
    
    this.totalValue = totalValue;
    next();
  } catch (error) {
    next(error);
  }
});

module.exports = mongoose.model('Portfolio', PortfolioSchema); 