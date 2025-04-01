/**
 * Common types for the investment application
 */

/**
 * Stock data interface
 */
export interface StockData {
  symbol: string;
  name?: string;
  price: number;
  change: number;
  changePercent: number;
  volume?: number;
  marketCap?: number;
  lastUpdated: Date | string;
}

/**
 * User profile interface
 */
export interface UserProfile {
  id: string;
  name: string;
  email: string;
  preferences?: UserPreferences;
  createdAt: Date | string;
}

/**
 * User preferences interface
 */
export interface UserPreferences {
  theme?: 'light' | 'dark' | 'system';
  currency?: string;
  riskTolerance?: 'low' | 'medium' | 'high';
  notifications?: boolean;
}

/**
 * Time periods for financial data
 */
export type TimePeriod = '1d' | '1w' | '1m' | '3m' | '6m' | '1y' | '5y' | 'max'; 