import { apiClient } from './api-client';
import { ENDPOINTS } from '../constants/endpoints';
import type {
  StockData,
  Recommendation,
  Portfolio,
  MarketSentiment,
} from '../types/investment';

export const stockService = {
  async getStockData(symbol: string): Promise<StockData> {
    const { data } = await apiClient.get(ENDPOINTS.STOCK.GET_DATA(symbol));
    return data;
  },

  async getStockHistory(symbol: string, period = '1y'): Promise<StockData[]> {
    const { data } = await apiClient.get(ENDPOINTS.STOCK.GET_HISTORY(symbol), {
      params: { period },
    });
    return data;
  },

  async searchStocks(query: string): Promise<StockData[]> {
    const { data } = await apiClient.get(ENDPOINTS.STOCK.SEARCH, {
      params: { query },
    });
    return data;
  },
};

export const recommendationService = {
  async getAllRecommendations(): Promise<Recommendation[]> {
    const { data } = await apiClient.get(ENDPOINTS.RECOMMENDATIONS.GET_ALL);
    return data;
  },

  async getRecommendationBySymbol(symbol: string): Promise<Recommendation> {
    const { data } = await apiClient.get(ENDPOINTS.RECOMMENDATIONS.GET_BY_SYMBOL(symbol));
    return data;
  },

  async getPortfolioRecommendations(): Promise<Recommendation[]> {
    const { data } = await apiClient.get(ENDPOINTS.RECOMMENDATIONS.GET_PORTFOLIO);
    return data;
  },
};

export const portfolioService = {
  async getPortfolio(): Promise<Portfolio> {
    const { data } = await apiClient.get(ENDPOINTS.PORTFOLIO.GET);
    return data;
  },

  async updatePortfolio(portfolio: Partial<Portfolio>): Promise<Portfolio> {
    const { data } = await apiClient.put(ENDPOINTS.PORTFOLIO.UPDATE, portfolio);
    return data;
  },

  async addHolding(symbol: string, shares: number, price: number): Promise<Portfolio> {
    const { data } = await apiClient.post(ENDPOINTS.PORTFOLIO.ADD_HOLDING, {
      symbol,
      shares,
      price,
    });
    return data;
  },

  async removeHolding(symbol: string): Promise<Portfolio> {
    const { data } = await apiClient.post(ENDPOINTS.PORTFOLIO.REMOVE_HOLDING, { symbol });
    return data;
  },
};

export const sentimentService = {
  async getSentimentBySymbol(symbol: string): Promise<MarketSentiment> {
    const { data } = await apiClient.get(ENDPOINTS.SENTIMENT.GET_BY_SYMBOL(symbol));
    return data;
  },

  async getMarketSentiment(): Promise<MarketSentiment[]> {
    const { data } = await apiClient.get(ENDPOINTS.SENTIMENT.GET_MARKET);
    return data;
  },
}; 