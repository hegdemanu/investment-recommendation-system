import api from '../utils/api';

export interface StockData {
  symbol: string;
  price: number;
  change: number;
  volume: number;
  timestamp: string;
}

export interface RecommendationData {
  symbol: string;
  confidence: number;
  type: 'buy' | 'sell' | 'hold';
  reason: string;
  timestamp: string;
}

export const investmentService = {
  // Stock related endpoints
  getStockData: async (symbol: string): Promise<StockData> => {
    const response = await api.get(`/stocks/${symbol}`);
    return response.data;
  },

  // Portfolio related endpoints
  getPortfolio: async () => {
    const response = await api.get('/portfolios');
    return response.data;
  },

  updatePortfolio: async (data: any) => {
    const response = await api.put('/portfolios', data);
    return response.data;
  },

  // Recommendation related endpoints
  getRecommendations: async () => {
    const response = await api.get('/recommendations');
    return response.data;
  },

  getStockRecommendation: async (symbol: string): Promise<RecommendationData> => {
    const response = await api.get(`/recommendations/${symbol}`);
    return response.data;
  },

  // Health check
  checkHealth: async () => {
    const response = await api.get('/health');
    return response.data;
  }
}; 