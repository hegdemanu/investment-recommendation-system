export const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:5000';

export const ENDPOINTS = {
  // Stock data endpoints
  STOCK: {
    GET_DATA: (symbol: string) => `/api/stocks/${symbol}`,
    GET_HISTORY: (symbol: string) => `/api/stocks/${symbol}/history`,
    SEARCH: '/api/stocks/search',
  },

  // Investment recommendations
  RECOMMENDATIONS: {
    GET_ALL: '/api/recommendations',
    GET_BY_SYMBOL: (symbol: string) => `/api/recommendations/${symbol}`,
    GET_PORTFOLIO: '/api/recommendations/portfolio',
  },

  // Portfolio management
  PORTFOLIO: {
    GET: '/api/portfolio',
    UPDATE: '/api/portfolio/update',
    ADD_HOLDING: '/api/portfolio/holdings/add',
    REMOVE_HOLDING: '/api/portfolio/holdings/remove',
  },

  // Market sentiment
  SENTIMENT: {
    GET_BY_SYMBOL: (symbol: string) => `/api/sentiment/${symbol}`,
    GET_MARKET: '/api/sentiment/market',
  },

  // ML Model endpoints
  ML: {
    PREDICT: '/api/ml/predict',
    ANALYZE: '/api/ml/analyze',
    RETRAIN: '/api/ml/retrain',
    GET_MODELS: '/api/ml/models',
    GET_MODEL_BY_ID: (modelId: string) => `/api/ml/models/${modelId}`,
    GET_MODELS_BY_SYMBOL: (symbol: string) => `/api/ml/models/symbol/${symbol}`,
    GET_MODELS_BY_TYPE: (type: string) => `/api/ml/models/type/${type}`,
    COMPARE_MODELS: (symbol: string) => `/api/ml/models/compare/${symbol}`,
    DELETE_MODEL: (modelId: string) => `/api/ml/models/${modelId}`,
    SENTIMENT_ANALYZE: (symbol: string) => `/api/ml/sentiment/${symbol}`,
    RAG_QUERY: '/api/ml/rag/query',
    MODEL_SELECTOR: (symbol: string) => `/api/ml/selector/${symbol}`,
  },

  // User management
  USER: {
    PROFILE: '/api/user/profile',
    PREFERENCES: '/api/user/preferences',
    WATCHLIST: '/api/user/watchlist',
  },
} as const; 