import React, { useState } from 'react';
import InteractiveChart from '@/components/InteractiveChart';

// Sample data generator
const generateSampleData = (days: number) => {
  const data = [];
  const startDate = new Date();
  startDate.setDate(startDate.getDate() - days);

  let price = 150;
  let volume = 1000000;

  for (let i = 0; i < days; i++) {
    const date = new Date(startDate);
    date.setDate(date.getDate() + i);

    // Simulate price movement
    const change = (Math.random() - 0.5) * 5;
    price += change;
    volume += (Math.random() - 0.5) * 200000;

    // Calculate technical indicators
    const ma20 = price + (Math.random() - 0.5) * 2;
    const ma50 = price + (Math.random() - 0.5) * 3;
    const ma200 = price + (Math.random() - 0.5) * 4;
    const rsi = Math.random() * 100;
    const macd = (Math.random() - 0.5) * 2;
    const signal = macd + (Math.random() - 0.5);
    const histogram = macd - signal;
    const bollingerBand = 10;

    // Generate random patterns
    const patterns = Math.random() > 0.8 ? [
      {
        type: Math.random() > 0.5 ? 'bullish' : 'bearish' as const,
        name: Math.random() > 0.5 ? 'Double Bottom' : 'Head and Shoulders',
        description: 'A technical analysis pattern indicating potential reversal',
      },
    ] : [];

    data.push({
      date: date.toISOString().split('T')[0],
      actual: price,
      predicted: price * (1 + (Math.random() - 0.5) * 0.1),
      volume: Math.max(100000, volume),
      ma20,
      ma50,
      ma200,
      rsi,
      macd,
      signal,
      histogram,
      bollingerUpper: price + bollingerBand,
      bollingerLower: price - bollingerBand,
      sentiment: Math.random(),
      patterns,
    });
  }

  return data;
};

const ChartShowcase = () => {
  const [data, setData] = useState(generateSampleData(90));
  const [showPrediction, setShowPrediction] = useState(true);
  const [showIndicators, setShowIndicators] = useState(true);
  const [showVolume, setShowVolume] = useState(true);
  const [showSentiment, setShowSentiment] = useState(true);
  const [showPatterns, setShowPatterns] = useState(true);
  const [refreshInterval, setRefreshInterval] = useState(60000);

  const handleDataUpdate = (newData: any[]) => {
    setData(newData);
  };

  return (
    <div className="min-h-screen bg-gray-50 p-8 dark:bg-gray-900">
      <div className="mx-auto max-w-7xl">
        <div className="mb-8">
          <h1 className="mb-4 text-3xl font-bold">Interactive Chart Showcase</h1>
          <p className="text-gray-600 dark:text-gray-400">
            Explore the capabilities of our advanced technical analysis chart
          </p>
        </div>

        <div className="mb-6 flex flex-wrap gap-4">
          <button
            onClick={() => setShowPrediction(!showPrediction)}
            className={`rounded-md px-4 py-2 text-sm font-medium ${
              showPrediction
                ? 'bg-blue-100 text-blue-700 dark:bg-blue-900 dark:text-blue-300'
                : 'bg-gray-100 text-gray-700 dark:bg-gray-800 dark:text-gray-300'
            }`}
          >
            Predictions
          </button>
          <button
            onClick={() => setShowIndicators(!showIndicators)}
            className={`rounded-md px-4 py-2 text-sm font-medium ${
              showIndicators
                ? 'bg-green-100 text-green-700 dark:bg-green-900 dark:text-green-300'
                : 'bg-gray-100 text-gray-700 dark:bg-gray-800 dark:text-gray-300'
            }`}
          >
            Technical Indicators
          </button>
          <button
            onClick={() => setShowVolume(!showVolume)}
            className={`rounded-md px-4 py-2 text-sm font-medium ${
              showVolume
                ? 'bg-purple-100 text-purple-700 dark:bg-purple-900 dark:text-purple-300'
                : 'bg-gray-100 text-gray-700 dark:bg-gray-800 dark:text-gray-300'
            }`}
          >
            Volume
          </button>
          <button
            onClick={() => setShowSentiment(!showSentiment)}
            className={`rounded-md px-4 py-2 text-sm font-medium ${
              showSentiment
                ? 'bg-cyan-100 text-cyan-700 dark:bg-cyan-900 dark:text-cyan-300'
                : 'bg-gray-100 text-gray-700 dark:bg-gray-800 dark:text-gray-300'
            }`}
          >
            Sentiment
          </button>
          <button
            onClick={() => setShowPatterns(!showPatterns)}
            className={`rounded-md px-4 py-2 text-sm font-medium ${
              showPatterns
                ? 'bg-pink-100 text-pink-700 dark:bg-pink-900 dark:text-pink-300'
                : 'bg-gray-100 text-gray-700 dark:bg-gray-800 dark:text-gray-300'
            }`}
          >
            Patterns
          </button>
        </div>

        <div className="mb-6">
          <label className="mb-2 block text-sm font-medium text-gray-700 dark:text-gray-300">
            Refresh Interval (seconds)
          </label>
          <select
            value={refreshInterval}
            onChange={(e) => setRefreshInterval(Number(e.target.value))}
            className="rounded-md border border-gray-300 bg-white px-3 py-2 text-sm dark:border-gray-700 dark:bg-gray-800"
          >
            <option value={1000}>1</option>
            <option value={5000}>5</option>
            <option value={15000}>15</option>
            <option value={30000}>30</option>
            <option value={60000}>60</option>
          </select>
        </div>

        <InteractiveChart
          data={data}
          title="DEMO Stock Analysis"
          showPrediction={showPrediction}
          showIndicators={showIndicators}
          showVolume={showVolume}
          showSentiment={showSentiment}
          showPatterns={showPatterns}
          height={600}
          onDataUpdate={handleDataUpdate}
          symbol="DEMO"
          refreshInterval={refreshInterval}
        />

        <div className="mt-8 rounded-lg border border-gray-200 bg-white p-6 dark:border-gray-800 dark:bg-gray-900">
          <h2 className="mb-4 text-xl font-semibold">Features</h2>
          <ul className="list-inside list-disc space-y-2 text-gray-600 dark:text-gray-400">
            <li>Interactive zoom and pan</li>
            <li>Real-time data updates</li>
            <li>Multiple technical indicators (MA20, MA50, MA200, RSI, MACD)</li>
            <li>Bollinger Bands</li>
            <li>Pattern recognition with visual markers</li>
            <li>Sentiment analysis overlay</li>
            <li>Volume analysis</li>
            <li>Price predictions</li>
            <li>Dark mode support</li>
            <li>Responsive design</li>
          </ul>
        </div>
      </div>
    </div>
  );
};

export default ChartShowcase; 