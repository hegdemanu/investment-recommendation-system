'use client';

import { useState } from 'react';
import { useMLModels } from '../hooks/useMLModels';
import { type ModelType, type ModelMetadata } from '@repo/api';

interface ModelSelectorProps {
  symbol: string;
  onSelectModel: (modelType: ModelType) => void;
  selectedModel?: ModelType;
}

export function ModelSelector({ symbol, onSelectModel, selectedModel }: ModelSelectorProps) {
  const [showModelStats, setShowModelStats] = useState(false);

  const {
    availableModels,
    modelComparison,
    recommendedModel,
    isLoading,
    error,
  } = useMLModels(symbol);

  if (isLoading) {
    return (
      <div className="animate-pulse rounded-lg bg-white p-4 shadow-sm">
        <div className="h-4 w-1/2 bg-gray-200 rounded mb-4"></div>
        <div className="h-8 bg-gray-200 rounded mb-4"></div>
        <div className="h-24 bg-gray-200 rounded"></div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="rounded-lg bg-white p-4 shadow-sm">
        <h3 className="text-lg font-semibold mb-2">Model Selection</h3>
        <p className="text-red-500">Error loading models: {error.message}</p>
      </div>
    );
  }

  if (!availableModels || availableModels.length === 0) {
    return (
      <div className="rounded-lg bg-white p-4 shadow-sm">
        <h3 className="text-lg font-semibold mb-2">Model Selection</h3>
        <p className="text-gray-500">No models available for {symbol}</p>
      </div>
    );
  }

  // Group models by type
  const modelsByType: Record<ModelType, ModelMetadata[]> = {} as Record<ModelType, ModelMetadata[]>;
  
  availableModels.forEach(model => {
    if (!modelsByType[model.modelType]) {
      modelsByType[model.modelType] = [];
    }
    modelsByType[model.modelType].push(model);
  });

  return (
    <div className="rounded-lg bg-white p-4 shadow-sm">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold">Model Selection</h3>
        {recommendedModel && (
          <div className="text-sm">
            <span className="text-gray-500">Recommended:</span>
            <span className="ml-1 text-blue-600 font-medium">{recommendedModel}</span>
          </div>
        )}
      </div>

      <div className="space-y-2 mb-4">
        {Object.entries(modelsByType).map(([type, models]) => (
          <button
            key={type}
            className={`w-full py-2 px-4 text-left rounded-md transition-colors ${
              selectedModel === type
                ? 'bg-blue-100 text-blue-700 border border-blue-300'
                : 'bg-gray-50 hover:bg-gray-100'
            }`}
            onClick={() => onSelectModel(type as ModelType)}
          >
            <div className="flex items-center justify-between">
              <div>
                <span className="font-medium">{getModelLabel(type as ModelType)}</span>
                <span className="text-xs text-gray-500 ml-2">
                  ({models.length} {models.length === 1 ? 'model' : 'models'})
                </span>
              </div>
              {modelComparison && (
                <span className="text-sm">
                  {getAccuracyFromComparison(modelComparison, type as ModelType)}
                </span>
              )}
            </div>
          </button>
        ))}
      </div>

      <button
        className="text-sm text-blue-600 hover:text-blue-800 flex items-center mt-1"
        onClick={() => setShowModelStats(!showModelStats)}
      >
        {showModelStats ? 'Hide Details' : 'Show Model Details'}
        <svg
          className={`ml-1 w-4 h-4 transition-transform ${showModelStats ? 'rotate-180' : ''}`}
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
          xmlns="http://www.w3.org/2000/svg"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M19 9l-7 7-7-7"
          />
        </svg>
      </button>

      {showModelStats && modelComparison && (
        <div className="mt-4 border-t pt-4">
          <h4 className="text-sm font-semibold mb-2">Performance Comparison</h4>
          <div className="overflow-x-auto">
            <table className="min-w-full text-sm">
              <thead>
                <tr className="border-b">
                  <th className="text-left py-2">Model</th>
                  <th className="text-left py-2">Accuracy</th>
                  <th className="text-left py-2">MSE</th>
                  <th className="text-left py-2">Last Updated</th>
                </tr>
              </thead>
              <tbody>
                {modelComparison.metrics.map((metric, index) => (
                  <tr key={index} className="border-b">
                    <td className="py-2">{getModelLabel(metric.modelType)}</td>
                    <td className="py-2">{(metric.accuracy * 100).toFixed(1)}%</td>
                    <td className="py-2">{metric.mse?.toFixed(4) || 'N/A'}</td>
                    <td className="py-2">{metric.lastUpdated ? formatDate(metric.lastUpdated) : 'N/A'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}

// Helper functions
function getModelLabel(modelType: ModelType): string {
  switch (modelType) {
    case 'LSTM':
      return 'LSTM';
    case 'ARIMA':
      return 'ARIMA-GARCH';
    case 'PROPHET':
      return 'Prophet';
    case 'ENSEMBLE':
      return 'Ensemble';
    case 'SENTIMENT':
      return 'Sentiment';
    default:
      return modelType;
  }
}

function getAccuracyFromComparison(
  comparison: { metrics: { modelType: ModelType; accuracy: number }[] },
  modelType: ModelType
): string {
  const metric = comparison.metrics.find(m => m.modelType === modelType);
  return metric ? `${(metric.accuracy * 100).toFixed(1)}%` : 'N/A';
}

function formatDate(dateString: string): string {
  const date = new Date(dateString);
  return new Intl.DateTimeFormat('en-US', {
    month: 'short',
    day: 'numeric',
    year: 'numeric',
  }).format(date);
} 