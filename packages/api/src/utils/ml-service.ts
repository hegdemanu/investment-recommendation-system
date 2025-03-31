import { apiClient } from './api-client';
import { ENDPOINTS } from '../constants/endpoints';
import type {
  ModelMetadata,
  PredictionRequest,
  PredictionResponse,
  SentimentResponse,
  ModelComparison,
  RAGResponse,
  ModelType,
} from '../types/ml-models';

export const mlService = {
  // Get predictions using ML models
  async getPrediction(request: PredictionRequest): Promise<PredictionResponse> {
    const { data } = await apiClient.post(ENDPOINTS.ML.PREDICT, request);
    return data;
  },

  // Get all available models
  async getModels(): Promise<ModelMetadata[]> {
    const { data } = await apiClient.get(ENDPOINTS.ML.GET_MODELS);
    return data;
  },

  // Get model by ID
  async getModelById(modelId: string): Promise<ModelMetadata> {
    const { data } = await apiClient.get(ENDPOINTS.ML.GET_MODEL_BY_ID(modelId));
    return data;
  },

  // Get models by symbol
  async getModelsBySymbol(symbol: string): Promise<ModelMetadata[]> {
    const { data } = await apiClient.get(ENDPOINTS.ML.GET_MODELS_BY_SYMBOL(symbol));
    return data;
  },

  // Get models by type
  async getModelsByType(type: ModelType): Promise<ModelMetadata[]> {
    const { data } = await apiClient.get(ENDPOINTS.ML.GET_MODELS_BY_TYPE(type));
    return data;
  },

  // Compare models for a symbol
  async compareModels(symbol: string): Promise<ModelComparison> {
    const { data } = await apiClient.get(ENDPOINTS.ML.COMPARE_MODELS(symbol));
    return data;
  },

  // Delete a model
  async deleteModel(modelId: string): Promise<void> {
    await apiClient.delete(ENDPOINTS.ML.DELETE_MODEL(modelId));
  },

  // Analyze sentiment for a symbol
  async analyzeSentiment(symbol: string): Promise<SentimentResponse> {
    const { data } = await apiClient.get(ENDPOINTS.ML.SENTIMENT_ANALYZE(symbol));
    return data;
  },

  // Query the RAG system
  async queryRAG(query: string, context?: string): Promise<RAGResponse> {
    const { data } = await apiClient.post(ENDPOINTS.ML.RAG_QUERY, { query, context });
    return data;
  },

  // Get model recommendation for a symbol
  async getRecommendedModel(symbol: string): Promise<ModelType> {
    const { data } = await apiClient.get(ENDPOINTS.ML.MODEL_SELECTOR(symbol));
    return data.recommendedModel;
  },

  // Retrain a model
  async retrainModel(
    symbol: string,
    modelType: ModelType,
    options?: Record<string, any>
  ): Promise<ModelMetadata> {
    const { data } = await apiClient.post(ENDPOINTS.ML.RETRAIN, {
      symbol,
      modelType,
      options,
    });
    return data;
  },
}; 