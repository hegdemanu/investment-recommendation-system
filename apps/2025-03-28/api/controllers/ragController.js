const ragService = require('../services/ragService');
const Stock = require('../models/Stock');

/**
 * RAG (Retrieval-Augmented Generation) Controller
 * Handles API endpoints for RAG functionality
 */

/**
 * Add a document to the RAG system
 * @route POST /api/rag/documents
 * @access Private
 */
exports.addDocument = async (req, res) => {
  try {
    const document = await ragService.addDocument(req.body);
    res.status(201).json(document);
  } catch (error) {
    console.error('Error in addDocument:', error);
    res.status(400).json({ message: error.message });
  }
};

/**
 * Search documents using RAG
 * @route GET /api/rag/search
 * @access Private
 */
exports.searchDocuments = async (req, res) => {
  try {
    const { query } = req.query;
    
    if (!query) {
      return res.status(400).json({ message: 'Search query is required' });
    }
    
    // Parse options
    const options = {
      contentType: req.query.contentType,
      stockIds: req.query.stockIds ? req.query.stockIds.split(',') : undefined,
      industries: req.query.industries ? req.query.industries.split(',') : undefined,
      dateFrom: req.query.dateFrom,
      dateTo: req.query.dateTo,
      limit: parseInt(req.query.limit) || 5
    };
    
    const results = await ragService.searchDocuments(query, options);
    
    res.json(results);
  } catch (error) {
    console.error('Error in searchDocuments:', error);
    res.status(500).json({ message: error.message });
  }
};

/**
 * Generate a response using RAG
 * @route POST /api/rag/generate
 * @access Private
 */
exports.generateResponse = async (req, res) => {
  try {
    const { query, options } = req.body;
    
    if (!query) {
      return res.status(400).json({ message: 'Query is required' });
    }
    
    const response = await ragService.generateResponse(query, options || {});
    
    res.json(response);
  } catch (error) {
    console.error('Error in generateResponse:', error);
    res.status(500).json({ message: error.message });
  }
};

/**
 * Delete a document from the RAG system
 * @route DELETE /api/rag/documents/:id
 * @access Private
 */
exports.deleteDocument = async (req, res) => {
  try {
    const success = await ragService.deleteDocument(req.params.id);
    
    if (!success) {
      return res.status(404).json({ message: 'Document not found' });
    }
    
    res.json({ message: 'Document deleted successfully' });
  } catch (error) {
    console.error('Error in deleteDocument:', error);
    res.status(500).json({ message: error.message });
  }
};

/**
 * Add a financial document related to a stock
 * @route POST /api/rag/stocks/:stockId/documents
 * @access Private
 */
exports.addStockDocument = async (req, res) => {
  try {
    const { stockId } = req.params;
    
    // Verify stock exists
    const stock = await Stock.findById(stockId);
    if (!stock) {
      return res.status(404).json({ message: 'Stock not found' });
    }
    
    // Prepare document data
    const documentData = {
      ...req.body,
      relatesToStocks: [stockId]
    };
    
    // Add metadata about the stock if not provided
    if (!documentData.metadata) {
      documentData.metadata = {};
    }
    
    if (!documentData.metadata.stockSymbol) {
      documentData.metadata.stockSymbol = stock.symbol;
    }
    
    if (!documentData.metadata.stockName) {
      documentData.metadata.stockName = stock.name;
    }
    
    const document = await ragService.addDocument(documentData);
    
    res.status(201).json(document);
  } catch (error) {
    console.error('Error in addStockDocument:', error);
    res.status(400).json({ message: error.message });
  }
};

/**
 * Get documents related to a stock
 * @route GET /api/rag/stocks/:stockId/documents
 * @access Private
 */
exports.getStockDocuments = async (req, res) => {
  try {
    const { stockId } = req.params;
    
    // Verify stock exists
    const stockExists = await Stock.exists({ _id: stockId });
    if (!stockExists) {
      return res.status(404).json({ message: 'Stock not found' });
    }
    
    // Parse options
    const options = {
      contentType: req.query.contentType,
      dateFrom: req.query.dateFrom,
      dateTo: req.query.dateTo,
      sortBy: req.query.sortBy || 'publishedDate',
      sortOrder: req.query.sortOrder || 'desc',
      page: parseInt(req.query.page) || 1,
      limit: parseInt(req.query.limit) || 10
    };
    
    const documents = await ragService.getDocumentsForStocks([stockId], options);
    
    res.json(documents);
  } catch (error) {
    console.error('Error in getStockDocuments:', error);
    res.status(500).json({ message: error.message });
  }
};

/**
 * Ask a question about a stock using RAG
 * @route POST /api/rag/stocks/:stockId/ask
 * @access Private
 */
exports.askStockQuestion = async (req, res) => {
  try {
    const { stockId } = req.params;
    const { query } = req.body;
    
    if (!query) {
      return res.status(400).json({ message: 'Query is required' });
    }
    
    // Verify stock exists
    const stock = await Stock.findById(stockId);
    if (!stock) {
      return res.status(404).json({ message: 'Stock not found' });
    }
    
    // Prepare options
    const options = {
      stockIds: [stockId],
      ...req.body.options
    };
    
    // Enhance query with stock context
    const enhancedQuery = `${query} (regarding ${stock.name}, symbol: ${stock.symbol})`;
    
    const response = await ragService.generateResponse(enhancedQuery, options);
    
    res.json(response);
  } catch (error) {
    console.error('Error in askStockQuestion:', error);
    res.status(500).json({ message: error.message });
  }
}; 