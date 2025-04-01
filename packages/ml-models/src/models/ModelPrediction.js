const mongoose = require('mongoose');

const ModelPredictionSchema = new mongoose.Schema({
  stock: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'Stock',
    required: true
  },
  model: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'AIModel',
    required: true
  },
  modelType: {
    type: String,
    enum: ['LSTM', 'GRU', 'ARIMA', 'GARCH', 'Prophet', 'XGBoost', 'Ensemble', 'RL', 'SARIMA', 'Custom'],
    required: true
  },
  modelVersion: {
    type: String,
    required: true
  },
  timeHorizon: {
    type: String,
    enum: ['Short-term', 'Medium-term', 'Long-term'],
    required: true
  },
  predictionType: {
    type: String,
    enum: ['Price', 'Direction', 'Volatility', 'TrendChange'],
    default: 'Price'
  },
  sentimentWeighted: {
    type: Boolean,
    default: false
  },
  confidenceScore: {
    type: Number,
    min: 0,
    max: 1,
    default: 0.5
  },
  dateGenerated: {
    type: Date,
    default: Date.now
  },
  predictionDates: [{
    date: {
      type: Date,
      required: true
    },
    predictedValue: {
      type: Number,
      required: true
    },
    predictedHigh: Number,
    predictedLow: Number,
    confidenceInterval: {
      lower: Number,
      upper: Number
    }
  }],
  actualValues: [{
    date: {
      type: Date,
      required: true
    },
    actualValue: {
      type: Number,
      required: true
    }
  }],
  accuracy: {
    rmse: Number,
    mape: Number,
    rSquared: Number,
    directionAccuracy: Number
  },
  // RL-specific fields
  reinforcementLearning: {
    isRLStrategy: {
      type: Boolean,
      default: false
    },
    actions: [{
      date: {
        type: Date,
        required: true
      },
      action: {
        type: String,
        enum: ['Buy', 'Sell', 'Hold'],
        required: true
      },
      confidence: {
        type: Number,
        min: 0,
        max: 1,
        default: 0.5
      },
      state: {
        type: mongoose.Schema.Types.Mixed
      },
      reward: Number,
      cumulativeReward: Number
    }],
    performance: {
      totalReward: Number,
      finalPortfolioValue: Number,
      totalTrades: {
        type: Number,
        default: 0
      },
      profitableTrades: {
        type: Number,
        default: 0
      },
      roi: Number,
      sharpeRatio: Number
    }
  },
  // Backtracking analysis
  backtracking: {
    isBacktested: {
      type: Boolean,
      default: false
    },
    backtestPeriod: {
      startDate: Date,
      endDate: Date
    },
    performanceMetrics: {
      profitLoss: Number,
      winRate: Number,
      maxDrawdown: Number,
      averageGain: Number,
      averageLoss: Number,
      expectancy: Number
    },
    // Store parameter sets tested during backtracking
    parameterSets: [{
      parameters: {
        type: mongoose.Schema.Types.Mixed
      },
      performance: {
        rmse: Number,
        profitLoss: Number,
        winRate: Number
      }
    }],
    // Best parameters found
    optimizedParameters: {
      type: mongoose.Schema.Types.Mixed
    }
  },
  metadata: {
    type: mongoose.Schema.Types.Mixed,
    default: {}
  },
  status: {
    type: String,
    enum: ['Pending', 'Active', 'Expired', 'Invalidated', 'Backtested'],
    default: 'Pending'
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
ModelPredictionSchema.pre('save', function(next) {
  this.updatedAt = Date.now();
  next();
});

// Method to update prediction with actual values
ModelPredictionSchema.methods.updateWithActual = function(date, actualValue) {
  // Find if this date already exists in actualValues
  const existingIndex = this.actualValues.findIndex(
    item => item.date.toISOString().split('T')[0] === new Date(date).toISOString().split('T')[0]
  );
  
  // Update or add the actual value
  if (existingIndex >= 0) {
    this.actualValues[existingIndex].actualValue = actualValue;
  } else {
    this.actualValues.push({ date, actualValue });
  }
  
  // Calculate accuracy metrics
  this.calculateAccuracy();
  
  return this.save();
};

// Calculate accuracy metrics
ModelPredictionSchema.methods.calculateAccuracy = function() {
  // Match predictions with actuals based on date
  const matchedData = this.predictionDates.map(prediction => {
    const predDate = new Date(prediction.date).toISOString().split('T')[0];
    const actual = this.actualValues.find(
      a => new Date(a.date).toISOString().split('T')[0] === predDate
    );
    
    return {
      predicted: prediction.predictedValue,
      actual: actual ? actual.actualValue : null,
      date: prediction.date
    };
  }).filter(item => item.actual !== null);
  
  // Calculate RMSE
  if (matchedData.length > 0) {
    // Root Mean Squared Error
    const squaredErrors = matchedData.map(item => 
      Math.pow(item.predicted - item.actual, 2)
    );
    const mse = squaredErrors.reduce((sum, error) => sum + error, 0) / matchedData.length;
    this.accuracy.rmse = Math.sqrt(mse);
    
    // Mean Absolute Percentage Error
    const absolutePercentageErrors = matchedData.map(item => 
      Math.abs((item.actual - item.predicted) / item.actual)
    );
    this.accuracy.mape = (absolutePercentageErrors.reduce((sum, error) => sum + error, 0) / matchedData.length) * 100;
    
    // Direction Accuracy (up/down prediction)
    let correctDirections = 0;
    for (let i = 1; i < matchedData.length; i++) {
      const actualChange = matchedData[i].actual - matchedData[i-1].actual;
      const predictedChange = matchedData[i].predicted - matchedData[i-1].predicted;
      
      if ((actualChange >= 0 && predictedChange >= 0) || (actualChange < 0 && predictedChange < 0)) {
        correctDirections++;
      }
    }
    
    this.accuracy.directionAccuracy = matchedData.length > 1 
      ? (correctDirections / (matchedData.length - 1)) * 100 
      : 0;
  }
  
  // Update status if all predictions have actuals
  if (this.actualValues.length >= this.predictionDates.length) {
    this.status = 'Expired';
  }
};

// Method to add RL action
ModelPredictionSchema.methods.addRLAction = function(actionData) {
  if (!this.reinforcementLearning.isRLStrategy) {
    this.reinforcementLearning.isRLStrategy = true;
  }
  
  if (!this.reinforcementLearning.actions) {
    this.reinforcementLearning.actions = [];
  }
  
  this.reinforcementLearning.actions.push(actionData);
  
  // Update total trades count
  if (actionData.action !== 'Hold') {
    this.reinforcementLearning.performance.totalTrades = (this.reinforcementLearning.performance.totalTrades || 0) + 1;
    
    // Update profitable trades if reward is positive
    if (actionData.reward > 0) {
      this.reinforcementLearning.performance.profitableTrades = (this.reinforcementLearning.performance.profitableTrades || 0) + 1;
    }
  }
  
  return this.save();
};

// Method to update RL performance
ModelPredictionSchema.methods.updateRLPerformance = function(performanceData) {
  this.reinforcementLearning.performance = {
    ...this.reinforcementLearning.performance,
    ...performanceData
  };
  
  return this.save();
};

// Method to run backtracking analysis
ModelPredictionSchema.methods.runBacktest = function(backtestData) {
  this.backtracking.isBacktested = true;
  this.backtracking.backtestPeriod = backtestData.backtestPeriod;
  this.backtracking.performanceMetrics = backtestData.performanceMetrics;
  
  if (backtestData.parameterSets) {
    this.backtracking.parameterSets = backtestData.parameterSets;
  }
  
  if (backtestData.optimizedParameters) {
    this.backtracking.optimizedParameters = backtestData.optimizedParameters;
  }
  
  this.status = 'Backtested';
  
  return this.save();
};

// Static method to find latest predictions for a stock
ModelPredictionSchema.statics.findLatestForStock = function(stockId, limit = 10) {
  return this.find({ 
    stock: stockId,
    status: 'Active'
  })
  .populate('model', 'name type weights')
  .sort({ dateGenerated: -1 })
  .limit(limit);
};

// Static method to find best performing model predictions
ModelPredictionSchema.statics.findBestPerforming = function(stockId, timeHorizon, metric = 'rmse') {
  const sortField = `accuracy.${metric}`;
  const sortDirection = metric === 'directionAccuracy' ? -1 : 1; // Higher is better for direction accuracy
  
  return this.find({
    stock: stockId,
    timeHorizon,
    status: { $in: ['Active', 'Expired', 'Backtested'] },
    [`accuracy.${metric}`]: { $exists: true, $ne: null }
  })
  .sort({ [sortField]: sortDirection })
  .limit(1);
};

// Static method to find best RL strategies
ModelPredictionSchema.statics.findBestRLStrategy = function(stockId, metric = 'totalReward') {
  const sortField = `reinforcementLearning.performance.${metric}`;
  
  return this.find({
    stock: stockId,
    'reinforcementLearning.isRLStrategy': true,
    [`reinforcementLearning.performance.${metric}`]: { $exists: true, $ne: null }
  })
  .sort({ [sortField]: -1 }) // Higher rewards are better
  .limit(1);
};

// Static method to find backtested predictions
ModelPredictionSchema.statics.findBacktested = function(stockId, limit = 10) {
  return this.find({
    stock: stockId,
    'backtracking.isBacktested': true
  })
  .populate('model', 'name type timeHorizon')
  .sort({ updatedAt: -1 })
  .limit(limit);
};

// Indexes for efficient querying
ModelPredictionSchema.index({ stock: 1, status: 1, dateGenerated: -1 });
ModelPredictionSchema.index({ model: 1, status: 1 });
ModelPredictionSchema.index({ timeHorizon: 1, predictionType: 1 });
ModelPredictionSchema.index({ 'reinforcementLearning.isRLStrategy': 1, stock: 1 });
ModelPredictionSchema.index({ 'backtracking.isBacktested': 1, stock: 1 });

module.exports = mongoose.model('ModelPrediction', ModelPredictionSchema); 