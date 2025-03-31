import { useQuery } from 'react-query';
import {
  mlService,
  type SentimentResponse,
} from '@repo/api';

export function useSentimentAnalysis(symbol?: string) {
  // Fetch sentiment analysis for a symbol
  const {
    data: sentiment,
    isLoading,
    error,
    refetch,
  } = useQuery<SentimentResponse>(
    ['sentiment', symbol],
    () => mlService.analyzeSentiment(symbol || ''),
    {
      enabled: !!symbol,
      staleTime: 15 * 60 * 1000, // 15 minutes
      refetchOnWindowFocus: false,
    }
  );

  // Get sentiment color based on sentiment value
  const getSentimentColor = (sentimentValue: number) => {
    if (sentimentValue > 0.2) return 'text-green-500';
    if (sentimentValue < -0.2) return 'text-red-500';
    return 'text-amber-500';
  };

  // Get sentiment label based on sentiment value
  const getSentimentLabel = (sentimentValue: number) => {
    if (sentimentValue > 0.2) return 'Bullish';
    if (sentimentValue < -0.2) return 'Bearish';
    return 'Neutral';
  };

  // Get sentiment icon based on sentiment value
  const getSentimentIcon = (sentimentValue: number) => {
    if (sentimentValue > 0.2) return '📈';
    if (sentimentValue < -0.2) return '📉';
    return '⚖️';
  };

  return {
    sentiment,
    isLoading,
    error,
    refetch,
    getSentimentColor,
    getSentimentLabel,
    getSentimentIcon,
  };
} 