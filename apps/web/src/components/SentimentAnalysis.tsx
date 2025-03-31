'use client';

import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Input } from './ui/input';
import { Button } from './ui/button';
import { PieChart, Pie, Cell, Tooltip, Legend } from 'recharts';
import { useSentimentAnalysis } from '../hooks/useSentimentAnalysis';
import { formatTimeAgo } from '../utils/dateUtils';

interface SentimentData {
  symbol: string;
  sentiment: {
    positive: number;
    negative: number;
    neutral: number;
  };
  recentNews: Array<{
    title: string;
    sentiment: 'positive' | 'negative' | 'neutral';
    score: number;
    date: string;
  }>;
}

const COLORS = ['#4ade80', '#f87171', '#94a3b8'];

const SentimentAnalysis: React.FC = () => {
  const [symbol, setSymbol] = useState<string>('');
  const [loading, setLoading] = useState<boolean>(false);
  const [sentimentData, setSentimentData] = useState<SentimentData | null>(null);

  const fetchSentimentData = async () => {
    if (!symbol) return;
    
    setLoading(true);
    try {
      const response = await fetch(`/api/v1/sentiment/${symbol}`);
      const data = await response.json();
      setSentimentData(data);
    } catch (error) {
      console.error('Error fetching sentiment data:', error);
    } finally {
      setLoading(false);
    }
  };

  const getSentimentColor = (sentiment: string) => {
    switch (sentiment) {
      case 'positive':
        return 'text-green-500';
      case 'negative':
        return 'text-red-500';
      default:
        return 'text-gray-500';
    }
  };

  const formatSentimentData = (sentiment: SentimentData['sentiment']) => {
    return [
      { name: 'Positive', value: sentiment.positive },
      { name: 'Negative', value: sentiment.negative },
      { name: 'Neutral', value: sentiment.neutral },
    ];
  };

  return (
    <Card className="w-full max-w-4xl mx-auto">
      <CardHeader>
        <CardTitle className="text-2xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
          Market Sentiment Analysis
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="flex gap-4 mb-6">
          <Input
            type="text"
            placeholder="Enter stock symbol (e.g., AAPL)"
            value={symbol}
            onChange={(e) => setSymbol(e.target.value.toUpperCase())}
            className="flex-1"
          />
          <Button
            onClick={fetchSentimentData}
            disabled={loading || !symbol}
            className="bg-gradient-to-r from-blue-600 to-purple-600 text-white"
          >
            Analyze
          </Button>
        </div>

        {sentimentData && (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <h3 className="text-lg font-semibold mb-4">Sentiment Distribution</h3>
              <PieChart width={300} height={300}>
                <Pie
                  data={formatSentimentData(sentimentData.sentiment)}
                  cx={150}
                  cy={150}
                  innerRadius={60}
                  outerRadius={80}
                  fill="#8884d8"
                  paddingAngle={5}
                  dataKey="value"
                >
                  {formatSentimentData(sentimentData.sentiment).map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip />
                <Legend />
              </PieChart>
            </div>
            
            <div>
              <h3 className="text-lg font-semibold mb-4">Recent News Sentiment</h3>
              <div className="space-y-4">
                {sentimentData.recentNews.map((news, index) => (
                  <div key={index} className="border rounded p-3">
                    <p className="font-medium">{news.title}</p>
                    <div className="flex justify-between mt-2 text-sm">
                      <span className={getSentimentColor(news.sentiment)}>
                        {news.sentiment.charAt(0).toUpperCase() + news.sentiment.slice(1)}
                        {' '}({(news.score * 100).toFixed(1)}%)
                      </span>
                      <span className="text-gray-500">{new Date(news.date).toLocaleDateString()}</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
};

export default SentimentAnalysis; 