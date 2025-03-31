import React from 'react';

type SentimentData = {
  sector: string;
  sentiment: 'Strongly Bullish' | 'Bullish' | 'Neutral' | 'Bearish' | 'Strongly Bearish';
};

interface MarketSentimentProps {
  overallMood: string;
  sectorSentiments: SentimentData[];
}

const getSentimentColorClass = (sentiment: string) => {
  switch (sentiment) {
    case 'Strongly Bullish':
      return 'text-green-600 dark:text-green-400';
    case 'Bullish':
      return 'text-green-500 dark:text-green-300';
    case 'Neutral':
      return 'text-gray-600 dark:text-gray-400';
    case 'Bearish':
      return 'text-red-500 dark:text-red-300';
    case 'Strongly Bearish':
      return 'text-red-600 dark:text-red-400';
    default:
      return 'text-gray-600 dark:text-gray-400';
  }
};

const getMoodBadgeClass = (mood: string) => {
  switch (mood.toLowerCase()) {
    case 'bullish':
      return 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200';
    case 'bearish':
      return 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200';
    case 'neutral':
      return 'bg-gray-100 text-gray-800 dark:bg-gray-800 dark:text-gray-200';
    default:
      return 'bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200';
  }
};

const MarketSentiment: React.FC<MarketSentimentProps> = ({ 
  overallMood = 'Bullish',
  sectorSentiments = [
    { sector: 'Technology', sentiment: 'Strongly Bullish' },
    { sector: 'Finance', sentiment: 'Neutral' },
    { sector: 'Healthcare', sentiment: 'Bullish' },
    { sector: 'Energy', sentiment: 'Bearish' },
  ]
}) => {
  return (
    <div className="rounded-lg border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-800 dark:bg-gray-900">
      <div className="mb-4 flex items-center justify-between">
        <span className="text-lg font-medium">Overall Market Mood</span>
        <span className={`rounded-full px-3 py-1 text-sm font-medium ${getMoodBadgeClass(overallMood)}`}>
          {overallMood}
        </span>
      </div>
      <div className="space-y-4">
        {sectorSentiments.map((item, i) => (
          <div key={i} className="flex items-center justify-between">
            <span className="text-gray-600 dark:text-gray-400">{item.sector}</span>
            <span className={getSentimentColorClass(item.sentiment)}>
              {item.sentiment}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
};

export default MarketSentiment; 