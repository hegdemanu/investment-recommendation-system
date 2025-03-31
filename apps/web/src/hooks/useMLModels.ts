import { useQuery, useMutation, useQueryClient } from 'react-query';
import {
  mlService,
  type ModelType,
  type PredictionRequest,
  type ModelMetadata,
} from '@repo/api';

export function useMLModels(symbol?: string) {
  const queryClient = useQueryClient();

  // Get model prediction
  const getPrediction = async (request: PredictionRequest) => {
    return mlService.getPrediction(request);
  };

  // Get prediction with default params
  const predictWithSymbol = async (
    symbolToUse: string = symbol || '', 
    modelType?: ModelType,
    horizon: number = 5
  ) => {
    return mlService.getPrediction({
      symbol: symbolToUse,
      modelType,
      horizon,
    });
  };

  // Fetch available models for a symbol
  const { 
    data: availableModels,
    isLoading: isLoadingModels,
    error: modelsError,
  } = useQuery(
    ['availableModels', symbol],
    () => mlService.getModelsBySymbol(symbol || ''),
    {
      enabled: !!symbol,
      staleTime: 5 * 60 * 1000, // 5 minutes
    }
  );

  // Get model comparison for a symbol
  const {
    data: modelComparison,
    isLoading: isLoadingComparison,
    error: comparisonError,
  } = useQuery(
    ['modelComparison', symbol],
    () => mlService.compareModels(symbol || ''),
    {
      enabled: !!symbol,
      staleTime: 5 * 60 * 1000, // 5 minutes
    }
  );

  // Get recommended model for a symbol
  const {
    data: recommendedModel,
    isLoading: isLoadingRecommendation,
    error: recommendationError,
  } = useQuery(
    ['recommendedModel', symbol],
    () => mlService.getRecommendedModel(symbol || ''),
    {
      enabled: !!symbol,
      staleTime: 5 * 60 * 1000, // 5 minutes
    }
  );

  // Retrain model mutation
  const retrainModel = useMutation(
    ({ symbolToUse, modelType, options }: { symbolToUse?: string, modelType: ModelType, options?: Record<string, any> }) =>
      mlService.retrainModel(symbolToUse || symbol || '', modelType, options),
    {
      onSuccess: () => {
        queryClient.invalidateQueries(['availableModels', symbol]);
        queryClient.invalidateQueries(['modelComparison', symbol]);
        queryClient.invalidateQueries(['recommendedModel', symbol]);
      },
    }
  );

  // Delete model mutation
  const deleteModel = useMutation(
    (modelId: string) => mlService.deleteModel(modelId),
    {
      onSuccess: () => {
        queryClient.invalidateQueries(['availableModels', symbol]);
        queryClient.invalidateQueries(['modelComparison', symbol]);
      },
    }
  );

  return {
    // Queries
    availableModels,
    modelComparison,
    recommendedModel,
    isLoading: isLoadingModels || isLoadingComparison || isLoadingRecommendation,
    error: modelsError || comparisonError || recommendationError,
    
    // Methods
    getPrediction,
    predictWithSymbol,
    retrainModel: retrainModel.mutate,
    deleteModel: deleteModel.mutate,
    
    // Mutation states
    isRetraining: retrainModel.isLoading,
    isDeleting: deleteModel.isLoading,
  };
} 