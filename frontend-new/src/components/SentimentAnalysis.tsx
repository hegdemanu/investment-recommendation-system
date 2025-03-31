import React, { useState } from 'react';
import { Card } from './ui/card';
import { Button } from './ui/button';

const SentimentAnalysis = () => {
  const [symbol, setSymbol] = useState('');
  const [sentiment, setSentiment] = useState<any>(null);
  const [loading, setLoading] = useState(false);

  const analyzeSentiment = async () => {
    setLoading(true);
    try {
      const response = await fetch(`/api/sentiment?symbol=${symbol}`);
      const data = await response.json();
      setSentiment(data);
    } catch (error) {
      console.error('Error analyzing sentiment:', error);
    }
    setLoading(false);
  };

  return (
    <Card className="p-6">
      <h2 className="text-2xl font-bold bg-gradient-to-r from-purple-600 to-blue-500 bg-clip-text text-transparent mb-6">
        Market Sentiment Analysis
      </h2>
      <div className="space-y-4">
        <div>
          <label className="block text-sm font-medium mb-1">Stock Symbol</label>
          <input
            type="text"
            value={symbol}
            onChange={(e) => setSymbol(e.target.value.toUpperCase())}
            placeholder="e.g., AAPL"
            className="w-full p-2 border rounded-md"
          />
        </div>
        <Button 
          onClick={analyzeSentiment} 
          className="w-full"
          disabled={loading || !symbol}
        >
          {loading ? 'Analyzing...' : 'Analyze Sentiment'}
        </Button>
        {sentiment && (
          <div className="mt-4 space-y-4">
            <div className="p-4 bg-gray-50 dark:bg-gray-800 rounded-md">
              <h3 className="text-lg font-semibold mb-2">Overall Sentiment</h3>
              <div className="flex items-center justify-between">
                <span className="text-sm">Bullish</span>
                <div className="w-2/3 h-2 bg-gray-200 rounded-full">
                  <div 
                    className="h-full bg-gradient-to-r from-purple-600 to-blue-500 rounded-full"
                    style={{ width: `${sentiment.bullish}%` }}
                  />
                </div>
                <span className="text-sm">{sentiment.bullish}%</span>
              </div>
              <div className="flex items-center justify-between mt-2">
                <span className="text-sm">Bearish</span>
                <div className="w-2/3 h-2 bg-gray-200 rounded-full">
                  <div 
                    className="h-full bg-gradient-to-r from-red-500 to-orange-500 rounded-full"
                    style={{ width: `${sentiment.bearish}%` }}
                  />
                </div>
                <span className="text-sm">{sentiment.bearish}%</span>
              </div>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="p-4 bg-gray-50 dark:bg-gray-800 rounded-md">
                <h3 className="text-lg font-semibold mb-2">News Sentiment</h3>
                <p className="text-sm">{sentiment.newsSummary}</p>
              </div>
              <div className="p-4 bg-gray-50 dark:bg-gray-800 rounded-md">
                <h3 className="text-lg font-semibold mb-2">Social Media Sentiment</h3>
                <p className="text-sm">{sentiment.socialSentiment}</p>
              </div>
            </div>
          </div>
        )}
      </div>
    </Card>
  );
};

export default SentimentAnalysis; 