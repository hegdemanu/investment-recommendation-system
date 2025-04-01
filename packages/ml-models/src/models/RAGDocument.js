const mongoose = require('mongoose');

const RAGDocumentSchema = new mongoose.Schema({
  title: {
    type: String,
    required: [true, 'Please provide a document title'],
    trim: true
  },
  content: {
    type: String,
    required: [true, 'Please provide document content'],
    trim: true
  },
  contentType: {
    type: String,
    enum: ['Report', 'News', 'SECFiling', 'EarningsCall', 'AnalystReport', 'ResearchPaper', 'MarketAnalysis', 'Other'],
    default: 'Other'
  },
  metadata: {
    type: mongoose.Schema.Types.Mixed,
    default: {}
  },
  source: {
    type: String,
    trim: true
  },
  sourceUrl: {
    type: String,
    trim: true
  },
  publishedDate: {
    type: Date
  },
  relatesToStocks: [{
    type: mongoose.Schema.Types.ObjectId,
    ref: 'Stock'
  }],
  relatesToIndustries: [String],
  vectorEmbedding: {
    type: [Number],
    select: false // Don't return vector by default to save bandwidth
  },
  embeddingModel: {
    type: String,
    default: 'sentence-transformers/all-MiniLM-L6-v2',
    trim: true
  },
  chunks: [{
    chunkId: String,
    content: String,
    vectorEmbedding: {
      type: [Number],
      select: false
    },
    metadata: {
      type: mongoose.Schema.Types.Mixed,
      default: {}
    }
  }],
  status: {
    type: String,
    enum: ['Pending', 'Processed', 'Failed'],
    default: 'Pending'
  },
  priority: {
    type: Number,
    default: 1,
    min: 1,
    max: 10
  },
  retrievalScore: {
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
  },
  lastRetrieved: {
    type: Date
  }
});

// Update the updatedAt field before saving
RAGDocumentSchema.pre('save', function(next) {
  this.updatedAt = Date.now();
  next();
});

// Method to update retrieval score (tracks how often the document has been useful)
RAGDocumentSchema.methods.updateRetrievalScore = function(score) {
  this.retrievalScore = this.retrievalScore + score;
  this.lastRetrieved = Date.now();
  return this.save();
};

// Method to mark document as processed after chunking and embedding
RAGDocumentSchema.methods.markAsProcessed = function() {
  this.status = 'Processed';
  return this.save();
};

// Static method to find relevant documents for RAG
RAGDocumentSchema.statics.findSimilar = function(embeddingVector, limit = 5) {
  // This is a placeholder for vector similarity search
  // In a real implementation, you would use a vector database or approximate nearest neighbor search
  // For now, we'll return documents sorted by priority and recency
  return this.find({ 
    status: 'Processed' 
  })
  .sort({ priority: -1, updatedAt: -1 })
  .limit(limit);
};

// Static method to find documents related to specific stocks
RAGDocumentSchema.statics.findForStocks = function(stockIds, limit = 10) {
  return this.find({
    relatesToStocks: { $in: stockIds },
    status: 'Processed'
  })
  .sort({ publishedDate: -1 })
  .limit(limit);
};

// Indexes for efficient querying
RAGDocumentSchema.index({ status: 1, priority: -1, updatedAt: -1 });
RAGDocumentSchema.index({ relatesToStocks: 1, publishedDate: -1 });
RAGDocumentSchema.index({ contentType: 1 });

module.exports = mongoose.model('RAGDocument', RAGDocumentSchema); 