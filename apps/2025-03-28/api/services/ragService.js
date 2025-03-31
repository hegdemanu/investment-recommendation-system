const RAGDocument = require('../models/RAGDocument');
const Stock = require('../models/Stock');

/**
 * RAG (Retrieval-Augmented Generation) Service
 * Handles storage, retrieval, and processing of RAG documents
 * In a production environment, this would integrate with vector databases and LLMs
 */
class RAGService {
  /**
   * Add a document to the RAG system
   * @param {Object} data - Document data
   * @returns {Promise<Object>} Newly created document
   */
  async addDocument(data) {
    try {
      // Validate related stocks if provided
      if (data.relatesToStocks && data.relatesToStocks.length > 0) {
        for (const stockId of data.relatesToStocks) {
          const stockExists = await Stock.exists({ _id: stockId });
          if (!stockExists) {
            throw new Error(`Stock with ID ${stockId} not found`);
          }
        }
      }
      
      // In a real implementation, this would calculate embeddings
      // using a model like sentence-transformers
      
      // Create document
      const document = new RAGDocument({
        title: data.title,
        content: data.content,
        contentType: data.contentType || 'Other',
        metadata: data.metadata || {},
        source: data.source || '',
        sourceUrl: data.sourceUrl || '',
        publishedDate: data.publishedDate || new Date(),
        relatesToStocks: data.relatesToStocks || [],
        relatesToIndustries: data.relatesToIndustries || [],
        status: 'Pending', // Initial status
        priority: data.priority || 1
      });
      
      await document.save();
      
      // Process document (chunking and embedding)
      // This would typically be an async operation in production
      await this._processDocument(document);
      
      return document;
    } catch (error) {
      throw new Error(`Error adding document to RAG system: ${error.message}`);
    }
  }

  /**
   * Process a document (chunking and embedding)
   * @param {Object} document - RAG document
   * @returns {Promise<Object>} Processed document
   * @private
   */
  async _processDocument(document) {
    try {
      // In a real implementation, this would:
      // 1. Split the document into chunks
      // 2. Generate embeddings for each chunk
      // 3. Store the chunks and embeddings
      
      // Mock implementation - create simple chunks by paragraphs
      const paragraphs = document.content.split(/\n\n+/);
      const chunks = [];
      
      for (let i = 0; i < paragraphs.length; i++) {
        const paragraph = paragraphs[i].trim();
        if (paragraph.length > 10) { // Skip very short paragraphs
          chunks.push({
            chunkId: `${document._id}-chunk-${i}`,
            content: paragraph,
            vectorEmbedding: this._mockGenerateEmbedding(paragraph), // Mock embedding
            metadata: {
              position: i,
              charLength: paragraph.length
            }
          });
        }
      }
      
      // Update document with chunks
      document.chunks = chunks;
      document.vectorEmbedding = this._mockGenerateEmbedding(document.title + ' ' + document.content);
      document.status = 'Processed';
      
      await document.save();
      return document;
    } catch (error) {
      console.error(`Error processing document: ${error}`);
      
      // Update document status to Failed
      document.status = 'Failed';
      await document.save();
      
      throw error;
    }
  }

  /**
   * Search for relevant documents using RAG
   * @param {String} query - Search query
   * @param {Object} options - Search options
   * @returns {Promise<Array>} Relevant documents
   */
  async searchDocuments(query, options = {}) {
    try {
      // In a real implementation, this would:
      // 1. Generate embedding for the query
      // 2. Perform vector similarity search
      // 3. Return relevant documents/chunks
      
      // Mock implementation - simple keyword search
      const searchRegex = new RegExp(query.split(/\s+/).join('|'), 'i');
      
      const filter = { status: 'Processed' };
      
      // Apply filters if provided
      if (options.contentType) {
        filter.contentType = options.contentType;
      }
      
      if (options.stockIds && options.stockIds.length > 0) {
        filter.relatesToStocks = { $in: options.stockIds };
      }
      
      if (options.industries && options.industries.length > 0) {
        filter.relatesToIndustries = { $in: options.industries };
      }
      
      if (options.dateFrom) {
        filter.publishedDate = { ...filter.publishedDate, $gte: new Date(options.dateFrom) };
      }
      
      if (options.dateTo) {
        filter.publishedDate = { ...filter.publishedDate, $lte: new Date(options.dateTo) };
      }
      
      // Perform search on title and content
      const documents = await RAGDocument.find({
        ...filter,
        $or: [
          { title: searchRegex },
          { content: searchRegex }
        ]
      })
      .sort({ priority: -1, publishedDate: -1 })
      .limit(options.limit || 5);
      
      // Calculate a mock relevance score
      const results = documents.map(doc => {
        // Count matches in title and content
        const titleMatches = (doc.title.match(searchRegex) || []).length;
        const contentMatches = (doc.content.match(searchRegex) || []).length;
        
        // Calculate a simple relevance score
        const relevanceScore = (titleMatches * 3 + contentMatches) / (doc.content.length / 100);
        
        return {
          document: doc,
          relevanceScore,
          chunks: this._findRelevantChunks(doc, query, 2) // Find up to 2 relevant chunks
        };
      });
      
      // Sort by relevance
      results.sort((a, b) => b.relevanceScore - a.relevanceScore);
      
      return results;
    } catch (error) {
      throw new Error(`Error searching RAG documents: ${error.message}`);
    }
  }

  /**
   * Find relevant chunks within a document
   * @param {Object} document - RAG document
   * @param {String} query - Search query
   * @param {Number} limit - Maximum number of chunks to return
   * @returns {Array} Relevant chunks
   * @private
   */
  _findRelevantChunks(document, query, limit = 2) {
    // In a real implementation, this would use vector similarity
    // Here we use simple keyword matching
    const searchRegex = new RegExp(query.split(/\s+/).join('|'), 'i');
    
    const relevantChunks = document.chunks
      .map(chunk => {
        const matches = (chunk.content.match(searchRegex) || []).length;
        return {
          chunk,
          relevance: matches / (chunk.content.length / 100)
        };
      })
      .filter(item => item.relevance > 0)
      .sort((a, b) => b.relevance - a.relevance)
      .slice(0, limit)
      .map(item => item.chunk);
    
    return relevantChunks;
  }

  /**
   * Generate a response using RAG
   * @param {String} query - User query
   * @param {Object} options - Options for RAG
   * @returns {Promise<Object>} RAG response
   */
  async generateResponse(query, options = {}) {
    try {
      // Search for relevant documents
      const searchResults = await this.searchDocuments(query, options);
      
      if (searchResults.length === 0) {
        return {
          response: "I couldn't find any relevant information to answer your query.",
          sources: []
        };
      }
      
      // Extract relevant chunks
      const relevantChunks = [];
      searchResults.forEach(result => {
        if (result.chunks && result.chunks.length > 0) {
          result.chunks.forEach(chunk => {
            relevantChunks.push({
              content: chunk.content,
              source: {
                title: result.document.title,
                url: result.document.sourceUrl,
                date: result.document.publishedDate
              }
            });
          });
        } else {
          // If no specific chunks, use a snippet from the document
          const snippet = result.document.content.substring(0, 200) + '...';
          relevantChunks.push({
            content: snippet,
            source: {
              title: result.document.title,
              url: result.document.sourceUrl,
              date: result.document.publishedDate
            }
          });
        }
      });
      
      // In a real implementation, this would call an LLM API
      // to generate a response based on the retrieved chunks
      const response = this._mockGenerateResponse(query, relevantChunks);
      
      // Update retrieval scores for the documents
      searchResults.forEach(async result => {
        await RAGDocument.findByIdAndUpdate(
          result.document._id,
          { 
            $inc: { retrievalScore: 1 },
            lastRetrieved: new Date()
          }
        );
      });
      
      return {
        response: response,
        sources: relevantChunks.map(chunk => chunk.source)
      };
    } catch (error) {
      throw new Error(`Error generating RAG response: ${error.message}`);
    }
  }

  /**
   * Mock function to generate embeddings
   * @param {String} text - Text to embed
   * @returns {Array} Mock embedding vector
   * @private
   */
  _mockGenerateEmbedding(text) {
    // In a real implementation, this would use a real embedding model
    // Here we generate a random vector of fixed size
    return Array.from({ length: 10 }, () => Math.random());
  }

  /**
   * Mock function to generate a response from chunks
   * @param {String} query - User query
   * @param {Array} chunks - Relevant chunks
   * @returns {String} Generated response
   * @private
   */
  _mockGenerateResponse(query, chunks) {
    // In a real implementation, this would use an LLM API
    // Here we generate a simple response based on the chunks
    
    if (chunks.length === 0) {
      return "I don't have enough information to answer your question.";
    }
    
    // Extract some sentences from the chunks
    const sentences = chunks.flatMap(chunk => 
      chunk.content.split(/[.!?]+/)
        .map(s => s.trim())
        .filter(s => s.length > 0)
    );
    
    // Simple response generation - concatenate a few sentences
    let response = `Based on the information I have, `;
    
    // Add 2-3 sentences from the chunks
    const numSentences = Math.min(3, sentences.length);
    for (let i = 0; i < numSentences; i++) {
      response += sentences[i] + '. ';
    }
    
    return response;
  }

  /**
   * Delete a document from the RAG system
   * @param {String} documentId - Document ID
   * @returns {Promise<Boolean>} Success status
   */
  async deleteDocument(documentId) {
    try {
      const result = await RAGDocument.findByIdAndDelete(documentId);
      return !!result;
    } catch (error) {
      throw new Error(`Error deleting RAG document: ${error.message}`);
    }
  }

  /**
   * Get documents related to specific stocks
   * @param {Array} stockIds - Array of stock IDs
   * @param {Object} options - Query options
   * @returns {Promise<Array>} Related documents
   */
  async getDocumentsForStocks(stockIds, options = {}) {
    try {
      if (!stockIds || stockIds.length === 0) {
        throw new Error('No stock IDs provided');
      }
      
      const query = {
        relatesToStocks: { $in: stockIds },
        status: 'Processed'
      };
      
      // Apply filters if provided
      if (options.contentType) {
        query.contentType = options.contentType;
      }
      
      if (options.dateFrom) {
        query.publishedDate = { ...query.publishedDate, $gte: new Date(options.dateFrom) };
      }
      
      if (options.dateTo) {
        query.publishedDate = { ...query.publishedDate, $lte: new Date(options.dateTo) };
      }
      
      // Apply sorting
      const sortField = options.sortBy || 'publishedDate';
      const sortOrder = options.sortOrder === 'asc' ? 1 : -1;
      
      // Apply pagination
      const limit = options.limit || 10;
      const skip = options.page ? (options.page - 1) * limit : 0;
      
      const documents = await RAGDocument.find(query)
        .sort({ [sortField]: sortOrder })
        .skip(skip)
        .limit(limit);
      
      return documents;
    } catch (error) {
      throw new Error(`Error fetching documents for stocks: ${error.message}`);
    }
  }
}

module.exports = new RAGService(); 