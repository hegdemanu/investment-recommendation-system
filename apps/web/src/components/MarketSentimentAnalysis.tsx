import React from 'react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';

interface SentimentData {
  sector: string;
  sentiment: number;
  newsCount: number;
  trendDirection: 'up' | 'down' | 'neutral';
  topKeywords: string[];
}

interface MarketSentimentAnalysisProps {
  data: SentimentData[];
  overallSentiment: {
    score: number;
    trend: 'bullish' | 'bearish' | 'neutral';
    confidence: number;
  };
}

const MarketSentimentAnalysis: React.FC<MarketSentimentAnalysisProps> = ({
  data = [
    {
      sector: 'Technology',
      sentiment: 0.75,
      newsCount: 120,
      trendDirection: 'up',
      topKeywords: ['AI', 'Cloud', 'Cybersecurity'],
    },
    {
      sector: 'Finance',
      sentiment: 0.45,
      newsCount: 85,
      trendDirection: 'neutral',
      topKeywords: ['Banking', 'Fintech', 'Crypto'],
    },
    {
      sector: 'Healthcare',
      sentiment: 0.65,
      newsCount: 95,
      trendDirection: 'up',
      topKeywords: ['Biotech', 'Digital Health', 'Medicare'],
    },
  ],
  overallSentiment = {
    score: 0.62,
    trend: 'bullish',
    confidence: 0.85,
  },
}) => {
  const getSentimentColor = (sentiment: number) => {
    if (sentiment >= 0.6) return '#10B981';
    if (sentiment >= 0.4) return '#F59E0B';
    return '#EF4444';
  };

  const getTrendIcon = (direction: 'up' | 'down' | 'neutral') => {
    switch (direction) {
      case 'up':
        return '↑';
      case 'down':
        return '↓';
      default:
        return '→';
    }
  };

  const getTrendColor = (direction: 'up' | 'down' | 'neutral') => {
    switch (direction) {
      case 'up':
        return 'text-green-500';
      case 'down':
        return 'text-red-500';
      default:
        return 'text-yellow-500';
    }
  };

  return (
    <div className="space-y-6 rounded-lg border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-800 dark:bg-gray-900">
      <div className="grid grid-cols-1 gap-6 md:grid-cols-3">
        <div className="rounded-lg bg-gradient-to-br from-green-50 to-blue-50 p-6 dark:from-green-900/20 dark:to-blue-900/20">
          <h3 className="mb-2 text-lg font-semibold">Overall Market Sentiment</h3>
          <div className="flex items-baseline justify-between">
            <p className="text-3xl font-bold text-green-600 dark:text-green-400">
              {(overallSentiment.score * 100).toFixed(1)}%
            </p>
            <span className="text-sm text-gray-600 dark:text-gray-400">
              Confidence: {(overallSentiment.confidence * 100).toFixed(1)}%
            </span>
          </div>
          <p className="mt-2 text-sm font-medium capitalize text-gray-600 dark:text-gray-400">
            Trend: {overallSentiment.trend}
          </p>
        </div>
      </div>

      <div className="h-80">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="sector" />
            <YAxis domain={[0, 1]} tickFormatter={(value) => `${(value * 100).toFixed(0)}%`} />
            <Tooltip
              formatter={(value: number) => [`${(value * 100).toFixed(1)}%`, 'Sentiment Score']}
            />
            <Legend />
            <Bar
              dataKey="sentiment"
              name="Sentiment Score"
              fill="#10B981"
              radius={[4, 4, 0, 0]}
            />
          </BarChart>
        </ResponsiveContainer>
      </div>

      <div className="mt-6">
        <h3 className="mb-4 text-lg font-semibold">Sector Analysis</h3>
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
          {data.map((item) => (
            <div
              key={item.sector}
              className="rounded-lg border border-gray-200 p-4 dark:border-gray-700"
            >
              <div className="flex items-center justify-between">
                <h4 className="font-medium">{item.sector}</h4>
                <span
                  className={`text-lg font-bold ${getTrendColor(item.trendDirection)}`}
                >
                  {getTrendIcon(item.trendDirection)}
                </span>
              </div>
              <div className="mt-2 flex items-center justify-between">
                <span className="text-sm text-gray-600 dark:text-gray-400">
                  News Count: {item.newsCount}
                </span>
                <span
                  className="font-medium"
                  style={{ color: getSentimentColor(item.sentiment) }}
                >
                  {(item.sentiment * 100).toFixed(1)}%
                </span>
              </div>
              <div className="mt-2">
                <p className="text-xs text-gray-600 dark:text-gray-400">Top Keywords:</p>
                <div className="mt-1 flex flex-wrap gap-2">
                  {item.topKeywords.map((keyword) => (
                    <span
                      key={keyword}
                      className="rounded-full bg-gray-100 px-2 py-1 text-xs font-medium dark:bg-gray-800"
                    >
                      {keyword}
                    </span>
                  ))}
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default MarketSentimentAnalysis; 