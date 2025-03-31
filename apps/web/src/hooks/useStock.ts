import { useQuery, useMutation, useQueryClient } from 'react-query';
import { stockService, type StockData } from '@repo/api';

export function useStock(symbol: string) {
  const queryClient = useQueryClient();

  // Fetch stock data
  const {
    data: stockData,
    isLoading: isLoadingStock,
    error: stockError,
  } = useQuery<StockData>(['stock', symbol], () => stockService.getStockData(symbol), {
    enabled: !!symbol,
    refetchInterval: 60000, // Refetch every minute
  });

  // Fetch stock history
  const {
    data: stockHistory,
    isLoading: isLoadingHistory,
    error: historyError,
  } = useQuery<StockData[]>(
    ['stockHistory', symbol],
    () => stockService.getStockHistory(symbol),
    {
      enabled: !!symbol,
      refetchInterval: 300000, // Refetch every 5 minutes
    }
  );

  // Search stocks
  const searchStocks = async (query: string) => {
    if (!query) return [];
    return stockService.searchStocks(query);
  };

  return {
    stockData,
    stockHistory,
    isLoading: isLoadingStock || isLoadingHistory,
    error: stockError || historyError,
    searchStocks,
  };
} 