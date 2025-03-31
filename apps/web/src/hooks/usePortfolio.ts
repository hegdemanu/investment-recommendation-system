import { useQuery, useMutation, useQueryClient } from 'react-query';
import { portfolioService, recommendationService, type Portfolio } from '@repo/api';

export function usePortfolio() {
  const queryClient = useQueryClient();

  // Fetch portfolio data
  const {
    data: portfolio,
    isLoading: isLoadingPortfolio,
    error: portfolioError,
  } = useQuery<Portfolio>('portfolio', portfolioService.getPortfolio);

  // Fetch portfolio recommendations
  const {
    data: recommendations,
    isLoading: isLoadingRecommendations,
    error: recommendationsError,
  } = useQuery('portfolioRecommendations', recommendationService.getPortfolioRecommendations);

  // Add holding mutation
  const addHolding = useMutation(
    ({ symbol, shares, price }: { symbol: string; shares: number; price: number }) =>
      portfolioService.addHolding(symbol, shares, price),
    {
      onSuccess: () => {
        queryClient.invalidateQueries('portfolio');
        queryClient.invalidateQueries('portfolioRecommendations');
      },
    }
  );

  // Remove holding mutation
  const removeHolding = useMutation(
    (symbol: string) => portfolioService.removeHolding(symbol),
    {
      onSuccess: () => {
        queryClient.invalidateQueries('portfolio');
        queryClient.invalidateQueries('portfolioRecommendations');
      },
    }
  );

  // Update portfolio mutation
  const updatePortfolio = useMutation(
    (updatedPortfolio: Partial<Portfolio>) => portfolioService.updatePortfolio(updatedPortfolio),
    {
      onSuccess: () => {
        queryClient.invalidateQueries('portfolio');
      },
    }
  );

  return {
    portfolio,
    recommendations,
    isLoading: isLoadingPortfolio || isLoadingRecommendations,
    error: portfolioError || recommendationsError,
    addHolding: addHolding.mutate,
    removeHolding: removeHolding.mutate,
    updatePortfolio: updatePortfolio.mutate,
    isUpdating:
      addHolding.isLoading || removeHolding.isLoading || updatePortfolio.isLoading,
  };
} 