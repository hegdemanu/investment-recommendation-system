const aiModelService = require('../services/aiModelService');
const Stock = require('../models/Stock');
const AIModel = require('../models/AIModel');
const ModelPrediction = require('../models/ModelPrediction');
const mongoose = require('mongoose');

/**
 * Model Controller - Handles API endpoints for AI models and predictions
 */

/**
 * Get all models
 * @route GET /api/models
 * @access Private
 */
exports.getModels = async (req, res) => {
  try {
    const models = await AIModel.find();
    res.json(models);
  } catch (error) {
    console.error('Error fetching models:', error.message);
    res.status(500).json({ message: 'Server error' });
  }
};

/**
 * Get a specific model by ID
 * @route GET /api/models/:id
 * @access Private
 */
exports.getModelById = async (req, res) => {
  try {
    const model = await aiModelService.getModelById(req.params.id);
    res.json(model);
  } catch (error) {
    console.error('Error in getModelById:', error);
    
    if (error.message === 'Model not found') {
      return res.status(404).json({ message: error.message });
    }
    
    res.status(500).json({ message: error.message });
  }
};

/**
 * Create a new model
 * @route POST /api/models
 * @access Private
 */
exports.createModel = async (req, res) => {
  try {
    const model = await aiModelService.createModel(req.body);
    res.status(201).json(model);
  } catch (error) {
    console.error('Error in createModel:', error);
    res.status(400).json({ message: error.message });
  }
};

/**
 * Update a model
 * @route PUT /api/models/:id
 * @access Private
 */
exports.updateModel = async (req, res) => {
  try {
    const model = await aiModelService.updateModel(req.params.id, req.body);
    res.json(model);
  } catch (error) {
    console.error('Error in updateModel:', error);
    
    if (error.message === 'Model not found') {
      return res.status(404).json({ message: error.message });
    }
    
    res.status(400).json({ message: error.message });
  }
};

/**
 * Delete a model
 * @route DELETE /api/models/:id
 * @access Private
 */
exports.deleteModel = async (req, res) => {
  try {
    const success = await aiModelService.deleteModel(req.params.id);
    
    if (!success) {
      return res.status(404).json({ message: 'Model not found' });
    }
    
    res.json({ message: 'Model deleted successfully' });
  } catch (error) {
    console.error('Error in deleteModel:', error);
    res.status(500).json({ message: error.message });
  }
};

/**
 * Generate predictions for a stock
 * @route POST /api/models/predictions/:stockId
 * @access Private
 */
exports.generatePredictions = async (req, res) => {
  try {
    const { stockId } = req.params;
    const { modelIds, options } = req.body;
    
    // Verify stock exists
    const stockExists = await Stock.exists({ _id: stockId });
    if (!stockExists) {
      return res.status(404).json({ message: 'Stock not found' });
    }
    
    const predictions = await aiModelService.generatePredictions(stockId, modelIds, options);
    
    res.json(predictions);
  } catch (error) {
    console.error('Error in generatePredictions:', error);
    res.status(500).json({ message: error.message });
  }
};

/**
 * Generate ensemble prediction for a stock
 * @route POST /api/models/predictions/:stockId/ensemble
 * @access Private
 */
exports.generateEnsemblePrediction = async (req, res) => {
  try {
    const { stockId } = req.params;
    const options = req.body;
    
    // Verify stock exists
    const stockExists = await Stock.exists({ _id: stockId });
    if (!stockExists) {
      return res.status(404).json({ message: 'Stock not found' });
    }
    
    const prediction = await aiModelService.generateEnsemblePrediction(stockId, options);
    
    res.json(prediction);
  } catch (error) {
    console.error('Error in generateEnsemblePrediction:', error);
    res.status(500).json({ message: error.message });
  }
};

/**
 * Get predictions for a stock
 * @route GET /api/models/predictions/:stockId
 * @access Private
 */
exports.getPredictionsForStock = async (req, res) => {
  try {
    const { stockId } = req.params;
    
    // Parse query parameters
    const modelId = req.query.modelId;
    const timeHorizon = req.query.timeHorizon;
    const status = req.query.status || 'Active';
    const limit = parseInt(req.query.limit) || 10;
    
    // Verify stock exists
    const stockExists = await Stock.exists({ _id: stockId });
    if (!stockExists) {
      return res.status(404).json({ message: 'Stock not found' });
    }
    
    // Build query
    const query = { stock: stockId, status };
    
    if (modelId) {
      query.model = modelId;
    }
    
    if (timeHorizon) {
      query.timeHorizon = timeHorizon;
    }
    
    // Get predictions from database
    const ModelPrediction = require('../models/ModelPrediction');
    const predictions = await ModelPrediction.find(query)
      .populate('model', 'name type timeHorizon sentimentWeighted')
      .sort({ dateGenerated: -1 })
      .limit(limit);
    
    res.json(predictions);
  } catch (error) {
    console.error('Error in getPredictionsForStock:', error);
    res.status(500).json({ message: error.message });
  }
};

/**
 * Update model weights based on prediction accuracy
 * @route PUT /api/models/:id/weights
 * @access Private
 */
exports.updateModelWeights = async (req, res) => {
  try {
    const { id } = req.params;
    
    const model = await aiModelService.updateModelWeights(id);
    
    res.json(model);
  } catch (error) {
    console.error('Error in updateModelWeights:', error);
    
    if (error.message.includes('Model not found') || error.message.includes('No completed predictions')) {
      return res.status(404).json({ message: error.message });
    }
    
    res.status(500).json({ message: error.message });
  }
};

/**
 * Create default models
 * @route POST /api/models/defaults
 * @access Private
 */
exports.createDefaultModels = async (req, res) => {
  try {
    // Check if models already exist
    const existingModels = await aiModelService.getModels();
    
    if (existingModels.length > 0) {
      return res.status(400).json({ 
        message: 'Default models already exist', 
        count: existingModels.length 
      });
    }
    
    // Create default models
    const defaultModels = [
      {
        name: 'LSTM Short-Term',
        type: 'LSTM',
        description: 'LSTM model for short-term predictions (1-7 days)',
        timeHorizon: 'Short-term',
        parameters: {
          sequence_length: 60,
          batch_size: 32,
          epochs: 100,
          learning_rate: 0.001
        },
        sentimentWeighted: false,
        status: 'Active',
        version: '1.0.0'
      },
      {
        name: 'GRU Medium-Term',
        type: 'GRU',
        description: 'GRU model for medium-term predictions (8-30 days)',
        timeHorizon: 'Medium-term',
        parameters: {
          sequence_length: 90,
          batch_size: 32,
          epochs: 100,
          learning_rate: 0.001
        },
        sentimentWeighted: false,
        status: 'Active',
        version: '1.0.0'
      },
      {
        name: 'LSTM with Sentiment',
        type: 'LSTM',
        description: 'LSTM model with sentiment weighting for short-term predictions',
        timeHorizon: 'Short-term',
        parameters: {
          sequence_length: 60,
          batch_size: 32,
          epochs: 100,
          learning_rate: 0.001
        },
        sentimentWeighted: true,
        sentimentImpact: 0.3,
        status: 'Active',
        version: '1.0.0'
      },
      {
        name: 'ARIMA Medium-Term',
        type: 'ARIMA',
        description: 'ARIMA model for medium-term predictions',
        timeHorizon: 'Medium-term',
        parameters: {
          p: 5,
          d: 1,
          q: 0
        },
        sentimentWeighted: false,
        status: 'Active',
        version: '1.0.0'
      },
      {
        name: 'Ensemble Long-Term',
        type: 'Ensemble',
        description: 'Ensemble model for long-term predictions (30+ days)',
        timeHorizon: 'Long-term',
        sentimentWeighted: true,
        sentimentImpact: 0.3,
        status: 'Active',
        version: '1.0.0'
      }
    ];
    
    const createdModels = [];
    
    for (const modelData of defaultModels) {
      const model = await aiModelService.createModel(modelData);
      createdModels.push(model);
    }
    
    res.status(201).json({ 
      message: 'Default models created successfully', 
      models: createdModels 
    });
  } catch (error) {
    console.error('Error in createDefaultModels:', error);
    res.status(500).json({ message: error.message });
  }
};

/**
 * Train a Reinforcement Learning model
 * @route POST /api/models/:id/train-rl
 * @access Private
 */
exports.trainRLModel = async (req, res) => {
  try {
    const { id } = req.params;
    const { stockId, trainingConfig } = req.body;

    if (!mongoose.Types.ObjectId.isValid(id)) {
      return res.status(400).json({ message: 'Invalid model ID' });
    }

    if (!mongoose.Types.ObjectId.isValid(stockId)) {
      return res.status(400).json({ message: 'Invalid stock ID' });
    }

    const result = await aiModelService.trainRLModel(id, stockId, trainingConfig);
    res.json(result);
  } catch (error) {
    console.error('Error training RL model:', error.message);
    res.status(500).json({ message: error.message });
  }
};

/**
 * Generate prediction using a Reinforcement Learning model
 * @route POST /api/models/:id/predict-rl/:stockId
 * @access Private
 */
exports.generateRLPrediction = async (req, res) => {
  try {
    const { id, stockId } = req.params;
    const options = req.body;

    if (!mongoose.Types.ObjectId.isValid(id)) {
      return res.status(400).json({ message: 'Invalid model ID' });
    }

    if (!mongoose.Types.ObjectId.isValid(stockId)) {
      return res.status(400).json({ message: 'Invalid stock ID' });
    }

    const prediction = await aiModelService.generateRLPrediction(stockId, id, options);
    res.json(prediction);
  } catch (error) {
    console.error('Error generating RL prediction:', error.message);
    res.status(500).json({ message: error.message });
  }
};

/**
 * Perform backtracking analysis on a model
 * @route POST /api/models/:id/backtrack/:stockId
 * @access Private
 */
exports.performBacktracking = async (req, res) => {
  try {
    const { id, stockId } = req.params;
    const backtrackConfig = req.body;

    if (!mongoose.Types.ObjectId.isValid(id)) {
      return res.status(400).json({ message: 'Invalid model ID' });
    }

    if (!mongoose.Types.ObjectId.isValid(stockId)) {
      return res.status(400).json({ message: 'Invalid stock ID' });
    }

    const result = await aiModelService.performBacktracking(id, stockId, backtrackConfig);
    res.json(result);
  } catch (error) {
    console.error('Error performing backtracking analysis:', error.message);
    res.status(500).json({ message: error.message });
  }
};

/**
 * Get backtracking results for a model
 * @route GET /api/models/:id/backtracking-results
 * @access Private
 */
exports.getBacktrackingResults = async (req, res) => {
  try {
    const { id } = req.params;

    if (!mongoose.Types.ObjectId.isValid(id)) {
      return res.status(400).json({ message: 'Invalid model ID' });
    }

    const model = await AIModel.findById(id);
    if (!model) {
      return res.status(404).json({ message: 'Model not found' });
    }

    if (!model.backtracking || !model.backtracking.enabled) {
      return res.status(404).json({ message: 'No backtracking results available for this model' });
    }

    // Get the latest predictions with backtracking results
    const predictions = await ModelPrediction.find({
      model: id,
      'backtracking.isBacktested': true
    }).sort({ dateGenerated: -1 }).limit(5);

    res.json({
      modelId: id,
      backtrackingConfig: model.backtracking,
      results: model.backtracking.lastBacktestResults || {},
      recentBacktests: predictions
    });
  } catch (error) {
    console.error('Error fetching backtracking results:', error.message);
    res.status(500).json({ message: error.message });
  }
};

/**
 * Get reinforcement learning results for a model
 * @route GET /api/models/:id/rl-performance/:stockId?
 * @access Private
 */
exports.getRLPerformance = async (req, res) => {
  try {
    const { id, stockId } = req.params;

    if (!mongoose.Types.ObjectId.isValid(id)) {
      return res.status(400).json({ message: 'Invalid model ID' });
    }

    const model = await AIModel.findById(id);
    if (!model) {
      return res.status(404).json({ message: 'Model not found' });
    }

    if (model.type !== 'RL' || !model.reinforcementLearning || !model.reinforcementLearning.enabled) {
      return res.status(404).json({ message: 'No RL data available for this model' });
    }

    // Find predictions with RL data
    const query = { model: id, 'reinforcementLearning.isRLStrategy': true };
    
    if (stockId && mongoose.Types.ObjectId.isValid(stockId)) {
      query.stock = stockId;
    }

    const predictions = await ModelPrediction.find(query)
      .sort({ dateGenerated: -1 })
      .limit(20);

    // Aggregate RL actions and performance
    const allActions = [];
    let totalReward = 0;
    let totalTrades = 0;
    let profitableTrades = 0;
    
    predictions.forEach(prediction => {
      if (prediction.reinforcementLearning && prediction.reinforcementLearning.actions) {
        allActions.push(...prediction.reinforcementLearning.actions);
        
        if (prediction.reinforcementLearning.performance) {
          totalReward += prediction.reinforcementLearning.performance.totalReward || 0;
          totalTrades += prediction.reinforcementLearning.performance.totalTrades || 0;
          profitableTrades += prediction.reinforcementLearning.performance.profitableTrades || 0;
        }
      }
    });

    // Sort actions by date
    allActions.sort((a, b) => new Date(a.date) - new Date(b.date));

    res.json({
      modelId: id,
      reinforcementLearning: model.reinforcementLearning,
      performance: {
        totalReward,
        totalTrades,
        profitableTrades,
        winRate: totalTrades > 0 ? (profitableTrades / totalTrades) * 100 : 0
      },
      recentActions: allActions.slice(-50), // Just get the most recent 50 actions
      recentPredictions: predictions
    });
  } catch (error) {
    console.error('Error fetching RL performance:', error.message);
    res.status(500).json({ message: error.message });
  }
};

/**
 * Update reinforcement learning parameters
 * @route PATCH /api/models/:id/rl-parameters
 * @access Private
 */
exports.updateRLParameters = async (req, res) => {
  try {
    const { id } = req.params;
    const updatedParameters = req.body;

    if (!mongoose.Types.ObjectId.isValid(id)) {
      return res.status(400).json({ message: 'Invalid model ID' });
    }

    const model = await AIModel.findById(id);
    if (!model) {
      return res.status(404).json({ message: 'Model not found' });
    }

    if (model.type !== 'RL') {
      return res.status(400).json({ message: 'Model is not a Reinforcement Learning type' });
    }

    // Update RL parameters
    if (!model.reinforcementLearning) {
      model.reinforcementLearning = { enabled: true };
    }

    // Update each parameter if provided
    if (updatedParameters.learningRate !== undefined) {
      model.reinforcementLearning.learningRate = updatedParameters.learningRate;
    }
    
    if (updatedParameters.explorationRate !== undefined) {
      model.reinforcementLearning.explorationRate = updatedParameters.explorationRate;
    }
    
    if (updatedParameters.discountFactor !== undefined) {
      model.reinforcementLearning.discountFactor = updatedParameters.discountFactor;
    }
    
    if (updatedParameters.rewardFunction !== undefined) {
      model.reinforcementLearning.rewardFunction = updatedParameters.rewardFunction;
    }
    
    if (updatedParameters.stateRepresentation !== undefined) {
      model.reinforcementLearning.stateRepresentation = updatedParameters.stateRepresentation;
    }
    
    if (updatedParameters.actionSpace !== undefined) {
      model.reinforcementLearning.actionSpace = updatedParameters.actionSpace;
    }

    await model.save();
    res.json(model);
  } catch (error) {
    console.error('Error updating RL parameters:', error.message);
    res.status(500).json({ message: error.message });
  }
}; 