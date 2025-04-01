'use client';

import React, { useState } from 'react';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Tooltip } from '@/components/ui/tooltip';
import { InfoIcon, TrendingUp, TrendingDown, BarChart2 } from 'lucide-react';

const SentimentAnalysis = () => {
  const [symbol, setSymbol] = useState('');
  const [timeframe, setTimeframe] = useState('7d');
  const [sentiment, setSentiment] = useState<any>(null);
  const [loading, setLoading] = useState(false);

  const analyzeSentiment = async () => {
    setLoading(true);
    try {
      const response = await fetch(`/api/sentiment?symbol=${symbol}&timeframe=${timeframe}`);
      const data = await response.json();
      setSentiment(data);
    } catch (error) {
      console.error('Error analyzing sentiment:', error);
    }
    setLoading(false);
  };

  const getSentimentColor = (score: number) => {
    if (score >= 75) return 'text-success-500';
    if (score >= 50) return 'text-primary-500';
    if (score >= 25) return 'text-warning-500';
    return 'text-danger-500';
  };

  const getSentimentIcon = (score: number) => {
    if (score >= 60) return <TrendingUp className="ml-2 h-4 w-4 text-success-500" />;
    if (score <= 40) return <TrendingDown className="ml-2 h-4 w-4 text-danger-500" />;
    return <BarChart2 className="ml-2 h-4 w-4 text-warning-500" />;
  };

  return (
    <Card className="p-6 border border-gray-200 dark:border-gray-800 shadow-md rounded-xl">
      <h2 className="text-2xl font-bold text-primary-700 dark:text-primary-400 mb-6 flex items-center">
        Market Sentiment Analysis
        <Tooltip content="Real-time sentiment analysis using fine-tuned FinBERT models">
          <InfoIcon className="ml-2 h-4 w-4 text-gray-400 cursor-help" />
        </Tooltip>
      </h2>
      
      <div className="space-y-4">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
              Stock Symbol
            </label>
            <Input
              type="text"
              value={symbol}
              onChange={(e) => setSymbol(e.target.value.toUpperCase())}
              placeholder="e.g., AAPL, MSFT, TSLA"
              className="w-full bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100"
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
              Timeframe
            </label>
            <select
              value={timeframe}
              onChange={(e) => setTimeframe(e.target.value)}
              className="w-full p-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100"
            >
              <option value="1d">1 Day</option>
              <option value="7d">1 Week</option>
              <option value="30d">1 Month</option>
              <option value="90d">3 Months</option>
            </select>
          </div>
        </div>
        
        <Button 
          onClick={analyzeSentiment} 
          className="w-full bg-primary-600 hover:bg-primary-700 text-white"
          disabled={loading || !symbol}
        >
          {loading ? 'Analyzing...' : 'Analyze Sentiment'}
        </Button>
        
        {sentiment && (
          <div className="mt-6 space-y-5">
            <div className="p-5 bg-gray-50 dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700">
              <h3 className="text-lg font-semibold mb-3 text-gray-800 dark:text-gray-100">Overall Market Sentiment</h3>
              
              <div className="flex items-center justify-between mb-3">
                <span className="font-medium">Bullish</span>
                <div className="w-2/3 h-3 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
                  <div 
                    className="h-full bg-gradient-to-r from-primary-400 to-success-500 rounded-full transition-all duration-500"
                    style={{ width: `${sentiment.bullish}%` }}
                  />
                </div>
                <span className={`font-bold ${getSentimentColor(sentiment.bullish)}`}>
                  {sentiment.bullish}%
                  {getSentimentIcon(sentiment.bullish)}
                </span>
              </div>
              
              <div className="flex items-center justify-between">
                <span className="font-medium">Bearish</span>
                <div className="w-2/3 h-3 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
                  <div 
                    className="h-full bg-gradient-to-r from-warning-400 to-danger-500 rounded-full transition-all duration-500"
                    style={{ width: `${sentiment.bearish}%` }}
                  />
                </div>
                <span className={`font-bold ${getSentimentColor(100 - sentiment.bearish)}`}>
                  {sentiment.bearish}%
                  {getSentimentIcon(100 - sentiment.bearish)}
                </span>
              </div>
            </div>
            
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="p-4 bg-gray-50 dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700">
                <h3 className="text-md font-semibold mb-2 text-gray-800 dark:text-gray-100">News Sentiment</h3>
                <div className="flex items-center space-x-2 mb-2">
                  <div className={`text-xl font-bold ${getSentimentColor(sentiment.newsScore * 100)}`}>
                    {(sentiment.newsScore * 100).toFixed(1)}%
                  </div>
                  {getSentimentIcon(sentiment.newsScore * 100)}
                </div>
                <p className="text-sm text-gray-600 dark:text-gray-300">{sentiment.newsSummary}</p>
              </div>
              
              <div className="p-4 bg-gray-50 dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700">
                <h3 className="text-md font-semibold mb-2 text-gray-800 dark:text-gray-100">Social Media</h3>
                <div className="flex items-center space-x-2 mb-2">
                  <div className={`text-xl font-bold ${getSentimentColor(sentiment.socialScore * 100)}`}>
                    {(sentiment.socialScore * 100).toFixed(1)}%
                  </div>
                  {getSentimentIcon(sentiment.socialScore * 100)}
                </div>
                <p className="text-sm text-gray-600 dark:text-gray-300">{sentiment.socialSentiment}</p>
              </div>
              
              <div className="p-4 bg-gray-50 dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700">
                <h3 className="text-md font-semibold mb-2 text-gray-800 dark:text-gray-100">Analyst Sentiment</h3>
                <div className="flex items-center space-x-2 mb-2">
                  <div className={`text-xl font-bold ${getSentimentColor(sentiment.analystScore * 100)}`}>
                    {(sentiment.analystScore * 100).toFixed(1)}%
                  </div>
                  {getSentimentIcon(sentiment.analystScore * 100)}
                </div>
                <p className="text-sm text-gray-600 dark:text-gray-300">{sentiment.analystInsights}</p>
              </div>
            </div>
            
            <div className="p-4 bg-gray-50 dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700">
              <h3 className="text-lg font-semibold mb-2 text-gray-800 dark:text-gray-100">AI-Generated Market Insights</h3>
              <p className="text-sm text-gray-600 dark:text-gray-300">{sentiment.aiInsights}</p>
            </div>
          </div>
        )}
      </div>
    </Card>
  );
};

export default SentimentAnalysis; 