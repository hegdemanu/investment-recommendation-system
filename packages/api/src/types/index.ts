/**
 * API-specific types for the investment application
 */

/**
 * API response wrapper interface
 */
export interface ApiResponse<T> {
  success: boolean;
  data?: T;
  error?: string;
  timestamp: string;
}

/**
 * Pagination parameters
 */
export interface PaginationParams {
  page: number;
  limit: number;
  totalPages?: number;
  totalItems?: number;
}

/**
 * Paginated response
 */
export interface PaginatedResponse<T> extends ApiResponse<T[]> {
  pagination: PaginationParams;
}

/**
 * Stock prediction request
 */
export interface PredictionRequest {
  symbol: string;
  modelType?: 'lstm' | 'arima' | 'prophet' | 'auto';
  horizon?: number;
  includeConfidenceIntervals?: boolean;
}

/**
 * RAG query request
 */
export interface RAGQueryRequest {
  query: string;
  context?: string;
  timeFrame?: '1d' | '7d' | '30d' | '90d';
} 