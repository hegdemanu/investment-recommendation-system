const mongoose = require('mongoose');

const AIModelSchema = new mongoose.Schema({
  name: {
    type: String,
    required: [true, 'Please provide a model name'],
    trim: true,
    unique: true
  },
  type: {
    type: String,
    required: [true, 'Please provide a model type'],
    enum: ['LSTM', 'GRU', 'ARIMA', 'GARCH', 'Prophet', 'XGBoost', 'Ensemble', 'FinBERT', 'RL', 'SARIMA', 'Custom'],
    trim: true
  },
  description: {
    type: String,
    required: [true, 'Please provide a model description'],
    trim: true
  },
  parameters: {
    type: mongoose.Schema.Types.Mixed,
    default: {}
  },
  weights: {
    type: Number,
    default: 1.0,
    min: 0,
    max: 1.0
  },
  timeHorizon: {
    type: String,
    enum: ['Short-term', 'Medium-term', 'Long-term'],
    required: true
  },
  accuracy: {
    rmse: {
      type: Number,
      default: 0
    },
    mape: {
      type: Number,
      default: 0
    },
    rSquared: {
      type: Number,
      default: 0
    },
    // Track backtracking-specific accuracy metrics
    backtrackingAccuracy: {
      type: Number,
      default: 0
    }
  },
  // RL-specific properties
  reinforcementLearning: {
    enabled: {
      type: Boolean,
      default: false
    },
    rewardFunction: {
      type: String,
      enum: ['ProfitMaximization', 'RiskAdjustedReturn', 'SharpeRatio', 'Custom'],
      default: 'ProfitMaximization'
    },
    explorationRate: {
      type: Number,
      default: 0.1,
      min: 0,
      max: 1
    },
    learningRate: {
      type: Number,
      default: 0.01,
      min: 0,
      max: 1
    },
    discountFactor: {
      type: Number,
      default: 0.95,
      min: 0,
      max: 1
    },
    stateRepresentation: {
      type: [String],
      default: ['Price', 'Volume', 'Indicators']
    },
    actionSpace: {
      type: [String],
      default: ['Buy', 'Sell', 'Hold']
    },
    episodes: {
      type: Number,
      default: 1000
    },
    totalReward: {
      type: Number,
      default: 0
    }
  },
  // Backtracking-specific properties
  backtracking: {
    enabled: {
      type: Boolean,
      default: false
    },
    windowSize: {
      type: Number,
      default: 30,  // Number of days to look back
      min: 1
    },
    stepSize: {
      type: Number,
      default: 1,   // Number of days to step back each time
      min: 1
    },
    optimizedParameters: {
      type: mongoose.Schema.Types.Mixed,
      default: {}
    },
    lastBacktestResults: {
      startDate: Date,
      endDate: Date,
      profitLoss: Number,
      successRate: Number,
      trades: Number,
      avgHoldingPeriod: Number
    }
  },
  sentimentWeighted: {
    type: Boolean,
    default: false
  },
  sentimentImpact: {
    type: Number,
    default: 0,
    min: 0,
    max: 1
  },
  status: {
    type: String,
    enum: ['Active', 'Inactive', 'Training', 'Evaluating', 'Backtracking'],
    default: 'Inactive'
  },
  lastTrained: {
    type: Date,
    default: null
  },
  lastEvaluated: {
    type: Date,
    default: null
  },
  lastBacktracked: {
    type: Date,
    default: null
  },
  version: {
    type: String,
    default: '1.0.0'
  },
  modelPath: {
    type: String,
    trim: true
  },
  tags: [String],
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
AIModelSchema.pre('save', function(next) {
  this.updatedAt = Date.now();
  next();
});

// Method to evaluate model performance
AIModelSchema.methods.updateAccuracy = function(metrics) {
  this.accuracy = {
    ...this.accuracy,
    ...metrics
  };
  this.lastEvaluated = Date.now();
  return this.save();
};

// Method to update model weights for ensemble
AIModelSchema.methods.updateWeight = function(weight) {
  this.weights = weight;
  return this.save();
};

// Method to update RL model parameters based on rewards
AIModelSchema.methods.updateRLParameters = function(params, totalReward) {
  if (this.reinforcementLearning && this.reinforcementLearning.enabled) {
    this.reinforcementLearning = {
      ...this.reinforcementLearning,
      ...params,
      totalReward
    };
    return this.save();
  }
  throw new Error('RL is not enabled for this model');
};

// Method to update backtracking results
AIModelSchema.methods.updateBacktrackingResults = function(results) {
  if (this.backtracking && this.backtracking.enabled) {
    this.backtracking.lastBacktestResults = results;
    this.backtracking.optimizedParameters = {
      ...this.backtracking.optimizedParameters,
      ...results.optimizedParameters
    };
    this.lastBacktracked = Date.now();
    return this.save();
  }
  throw new Error('Backtracking is not enabled for this model');
};

// Static method to find best model for a specific time horizon
AIModelSchema.statics.findBestModel = function(timeHorizon) {
  return this.find({ 
    timeHorizon,
    status: 'Active'
  }).sort({ 'accuracy.rmse': 1 }).limit(1);
};

// Static method to find best RL model
AIModelSchema.statics.findBestRLModel = function() {
  return this.find({
    type: 'RL',
    status: 'Active',
    'reinforcementLearning.enabled': true
  }).sort({ 'reinforcementLearning.totalReward': -1 }).limit(1);
};

module.exports = mongoose.model('AIModel', AIModelSchema); 