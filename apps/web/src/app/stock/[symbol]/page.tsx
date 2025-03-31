'use client';

import { useState, useEffect } from 'react';
import { useStock } from '../../../hooks/useStock';
import { useMLModels } from '../../../hooks/useMLModels';
import { PredictionChart } from '../../../components/PredictionChart';
import { ModelSelector } from '../../../components/ModelSelector';
import { SentimentAnalysis } from '../../../components/SentimentAnalysis';
import { RAGChat } from '../../../components/RAGChat';
import { type ModelType } from '@repo/api';

interface PageProps {
  params: {
    symbol: string;
  };
}

export default function StockDetailPage({ params }: PageProps) {
  const symbol = params.symbol.toUpperCase();
  const [selectedModelType, setSelectedModelType] = useState<ModelType | undefined>();
  const [predictionHorizon, setPredictionHorizon] = useState(5);
  const [prediction, setPrediction] = useState<any>(null);
  const [isLoadingPrediction, setIsLoadingPrediction] = useState(false);
  const [predictionError, setPredictionError] = useState<Error | null>(null);

  const { stockData, stockHistory, isLoading: isLoadingStock, error: stockError } = useStock(symbol);
  const { predictWithSymbol, recommendedModel } = useMLModels(symbol);

  // Load initial prediction using recommended model when available
  useEffect(() => {
    if (recommendedModel && !selectedModelType) {
      setSelectedModelType(recommendedModel);
    }
  }, [recommendedModel, selectedModelType]);

  // Get prediction when model type changes
  useEffect(() => {
    const fetchPrediction = async () => {
      if (!selectedModelType) return;
      
      try {
        setIsLoadingPrediction(true);
        setPredictionError(null);
        const result = await predictWithSymbol(symbol, selectedModelType, predictionHorizon);
        setPrediction(result);
      } catch (error) {
        console.error('Error fetching prediction:', error);
        setPredictionError(error as Error);
      } finally {
        setIsLoadingPrediction(false);
      }
    };

    fetchPrediction();
  }, [symbol, selectedModelType, predictionHorizon, predictWithSymbol]);

  // Handle model selection
  const handleModelSelect = (modelType: ModelType) => {
    setSelectedModelType(modelType);
  };

  // Handle prediction horizon change
  const handleHorizonChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    setPredictionHorizon(Number(e.target.value));
  };

  if (isLoadingStock && !stockData) {
    return (
      <div className="container mx-auto p-4">
        <h1 className="text-2xl font-bold mb-6">Loading {symbol} Data...</h1>
        <div className="animate-pulse">
          <div className="h-64 bg-gray-200 rounded-lg mb-4"></div>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="h-64 bg-gray-200 rounded-lg"></div>
            <div className="h-64 bg-gray-200 rounded-lg"></div>
          </div>
        </div>
      </div>
    );
  }

  if (stockError) {
    return (
      <div className="container mx-auto p-4">
        <h1 className="text-2xl font-bold mb-6">Error Loading {symbol}</h1>
        <div className="bg-red-100 text-red-800 p-4 rounded-lg">
          <p>{stockError.message}</p>
        </div>
      </div>
    );
  }

  // Format historical data for chart
  const historicalChartData = stockHistory
    ? stockHistory.map((item) => ({
        date: item.date,
        price: item.price,
      }))
    : [];

  return (
    <div className="container mx-auto p-4">
      <div className="mb-6">
        <h1 className="text-2xl font-bold">{symbol}</h1>
        {stockData && (
          <div className="flex items-center mt-2">
            <span className="text-3xl font-semibold">${stockData.price.toFixed(2)}</span>
            <span
              className={`ml-2 text-lg ${
                stockData.change >= 0 ? 'text-green-500' : 'text-red-500'
              }`}
            >
              {stockData.change >= 0 ? '+' : ''}
              {stockData.change.toFixed(2)} ({stockData.changePercent.toFixed(2)}%)
            </span>
          </div>
        )}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Main content - 8 columns */}
        <div className="lg:col-span-8 space-y-6">
          {/* Prediction Section */}
          <div className="bg-white rounded-lg p-4 shadow-sm">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-xl font-semibold">Price Prediction</h2>
              <div className="flex items-center">
                <label htmlFor="horizon" className="text-sm mr-2">
                  Horizon:
                </label>
                <select
                  id="horizon"
                  value={predictionHorizon}
                  onChange={handleHorizonChange}
                  className="text-sm border rounded p-1"
                >
                  <option value="3">3 days</option>
                  <option value="5">5 days</option>
                  <option value="7">7 days</option>
                  <option value="14">14 days</option>
                  <option value="30">30 days</option>
                </select>
              </div>
            </div>

            {predictionError && (
              <div className="bg-red-100 text-red-800 p-3 rounded mb-4">
                <p className="text-sm">Error loading prediction: {predictionError.message}</p>
              </div>
            )}

            {isLoadingPrediction && !prediction ? (
              <div className="h-64 flex items-center justify-center">
                <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-blue-500"></div>
              </div>
            ) : prediction ? (
              <PredictionChart
                prediction={prediction}
                historicalData={historicalChartData.slice(-30)}
                height={350}
              />
            ) : (
              <div className="h-64 flex items-center justify-center">
                <p className="text-gray-500">Select a model to view predictions</p>
              </div>
            )}
          </div>

          {/* RAG Chat */}
          <div className="bg-white rounded-lg shadow-sm" style={{ height: '500px' }}>
            <RAGChat symbol={symbol} />
          </div>
        </div>

        {/* Sidebar - 4 columns */}
        <div className="lg:col-span-4 space-y-6">
          {/* Model Selector */}
          <ModelSelector
            symbol={symbol}
            onSelectModel={handleModelSelect}
            selectedModel={selectedModelType}
          />

          {/* Sentiment Analysis */}
          <SentimentAnalysis symbol={symbol} />
        </div>
      </div>
    </div>
  );
} 