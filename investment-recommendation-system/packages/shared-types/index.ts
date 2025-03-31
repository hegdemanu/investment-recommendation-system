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

export interface UserPortfolio {
  id: string;
  userId: string;
  stocks: {
    symbol: string;
    shares: number;
    averagePrice: number;
  }[];
  createdAt: string;
  updatedAt: string;
}

export interface ApiResponse<T> {
  success: boolean;
  data?: T;
  error?: string;
  timestamp: string;
} 