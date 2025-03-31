import { z } from 'zod';

// Model types
export const ModelTypeEnum = z.enum([
  'LSTM', 
  'ARIMA', 
  'PROPHET', 
  'SENTIMENT', 
  'ENSEMBLE'
]);

export type ModelType = z.infer<typeof ModelTypeEnum>;

// Model metadata schema
export const ModelMetadataSchema = z.object({
  modelId: z.string(),
  modelType: ModelTypeEnum,
  symbol: z.string(),
  createdAt: z.string(),
  updatedAt: z.string(),
  version: z.string(),
  accuracy: z.number(),
  mse: z.number().optional(),
  mae: z.number().optional(),
  rmse: z.number().optional(),
  trainHistory: z.record(z.array(z.number())).optional(),
  features: z.array(z.string()).optional(),
  hyperparameters: z.record(z.any()).optional(),
});

export type ModelMetadata = z.infer<typeof ModelMetadataSchema>;

// Prediction request schema
export const PredictionRequestSchema = z.object({
  symbol: z.string(),
  modelType: ModelTypeEnum.optional(),
  horizon: z.number().optional().default(5),
  includeMetadata: z.boolean().optional().default(false),
});

export type PredictionRequest = z.infer<typeof PredictionRequestSchema>;

// Prediction response schema
export const PredictionResponseSchema = z.object({
  symbol: z.string(),
  modelType: ModelTypeEnum,
  predictions: z.array(
    z.object({
      date: z.string(),
      price: z.number(),
      upperBound: z.number().optional(),
      lowerBound: z.number().optional(),
    })
  ),
  confidence: z.number(),
  metadata: ModelMetadataSchema.optional(),
});

export type PredictionResponse = z.infer<typeof PredictionResponseSchema>;

// Sentiment analysis response schema
export const SentimentResponseSchema = z.object({
  symbol: z.string(),
  overallSentiment: z.number(), // -1 to 1 scale
  sentimentLabel: z.enum(['BEARISH', 'NEUTRAL', 'BULLISH']),
  confidence: z.number(),
  newsItems: z.array(
    z.object({
      title: z.string(),
      url: z.string().optional(),
      source: z.string(),
      date: z.string(),
      sentiment: z.number(),
      relevance: z.number().optional(),
    })
  ).optional(),
});

export type SentimentResponse = z.infer<typeof SentimentResponseSchema>;

// Model comparison schema
export const ModelComparisonSchema = z.object({
  symbol: z.string(),
  metrics: z.array(
    z.object({
      modelType: ModelTypeEnum,
      accuracy: z.number(),
      mse: z.number().optional(),
      mae: z.number().optional(),
      rmse: z.number().optional(),
      trainingTime: z.number().optional(),
      lastUpdated: z.string().optional(),
    })
  ),
  recommendedModel: ModelTypeEnum,
});

export type ModelComparison = z.infer<typeof ModelComparisonSchema>;

// RAG response schema
export const RAGResponseSchema = z.object({
  query: z.string(),
  response: z.string(),
  sources: z.array(
    z.object({
      title: z.string(),
      url: z.string().optional(),
      relevance: z.number().optional(),
      snippet: z.string().optional(),
    })
  ).optional(),
  relatedSymbols: z.array(z.string()).optional(),
});

export type RAGResponse = z.infer<typeof RAGResponseSchema>; 