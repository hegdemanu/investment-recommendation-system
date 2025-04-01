const mongoose = require('mongoose');

const RecommendationSchema = new mongoose.Schema({
  user: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'User',
    required: true
  },
  stock: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'Stock',
    required: true
  },
  title: {
    type: String,
    required: [true, 'Please provide a recommendation title'],
    trim: true
  },
  description: {
    type: String,
    required: [true, 'Please provide a recommendation description'],
    trim: true
  },
  action: {
    type: String,
    required: true,
    enum: ['Buy', 'Sell', 'Hold']
  },
  targetPrice: {
    type: Number
  },
  potentialReturn: {
    type: Number
  },
  riskLevel: {
    type: String,
    enum: ['Low', 'Medium', 'High'],
    required: true
  },
  timeHorizon: {
    type: String,
    enum: ['Short-term', 'Medium-term', 'Long-term'],
    required: true
  },
  rationale: {
    type: String,
    required: [true, 'Please provide a rationale for the recommendation'],
    trim: true
  },
  status: {
    type: String,
    enum: ['Active', 'Expired', 'Fulfilled'],
    default: 'Active'
  },
  targetDate: {
    type: Date
  },
  accuracy: {
    type: Number,
    min: 0,
    max: 100
  },
  relevanceScore: {
    type: Number,
    default: 0
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
RecommendationSchema.pre('save', function(next) {
  this.updatedAt = Date.now();
  next();
});

// Index for efficient querying
RecommendationSchema.index({ user: 1, stock: 1, createdAt: -1 });

module.exports = mongoose.model('Recommendation', RecommendationSchema); 