import React from 'react';
import Link from 'next/link';
import Navbar from '@/components/Navbar';
import PortfolioStats from '@/components/PortfolioStats';
import StockRecommendations from '@/components/StockRecommendations';
import MarketSentiment from '@/components/MarketSentiment';
import PortfolioDashboard from '@/components/PortfolioDashboard';

export const metadata = {
  title: 'Dashboard',
  description: 'View your investment portfolio and recommendations',
};

// Sample data - replace with actual data fetching logic
const sampleData = {
  portfolioValue: 380.88,
  portfolio: [
    {
      symbol: 'AAPL',
      price: 384.85,
      change: -1.15,
      changePercent: -0.30,
      value: 3848.52
    },
    {
      symbol: 'MSFT',
      price: 390.58,
      change: 0.61,
      changePercent: 0.16,
      value: 3905.80
    },
    {
      symbol: 'GOOGL',
      price: 467.63,
      change: -8.37,
      changePercent: -1.76,
      value: 4676.34
    }
  ],
  watchlist: [
    {
      symbol: 'AMZN',
      price: 418.51,
      change: 8.51,
      changePercent: 2.08
    },
    {
      symbol: 'TSLA',
      price: 409.24,
      change: 1.24,
      changePercent: 0.30
    },
    {
      symbol: 'META',
      price: 391.95,
      change: -3.05,
      changePercent: -0.77
    }
  ],
  recentTrades: [
    {
      date: '2025-03-27',
      symbol: 'AAPL',
      action: 'BUY' as const,
      shares: 10,
      price: 185.92,
      total: 1859.20
    },
    {
      date: '2025-03-25',
      symbol: 'TSLA',
      action: 'SELL' as const,
      shares: 5,
      price: 175.34,
      total: 876.70
    },
    {
      date: '2025-03-22',
      symbol: 'MSFT',
      action: 'BUY' as const,
      shares: 8,
      price: 416.42,
      total: 3331.36
    }
  ]
};

export default function DashboardPage() {
  return (
    <>
      <Navbar />
      <div className="container mx-auto p-6">
        <header className="mb-8 flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold">Dashboard</h1>
            <p className="text-gray-600 dark:text-gray-400">
              View your portfolio performance and investment recommendations
            </p>
          </div>
          <Link
            href="/recommendations"
            className="rounded-md bg-primary px-4 py-2 text-primary-foreground shadow-sm hover:bg-primary/90 focus:outline-none focus:ring-2 focus:ring-primary focus:ring-offset-2"
          >
            View Recommendations
          </Link>
        </header>

        <PortfolioStats stats={[
          { title: 'Portfolio Value', value: '$125,430.00', change: '+2.3%' },
          { title: 'Total Return', value: '+$12,430.00', change: '+10.8%' },
          { title: 'Assets', value: '15', change: '' },
          { title: 'Risk Score', value: '68/100', change: 'Moderate' },
        ]} />

        <div className="mb-8">
          <h2 className="mb-4 text-xl font-semibold">Portfolio Performance</h2>
          <div className="h-80 rounded-lg border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-800 dark:bg-gray-900">
            <div className="flex items-center justify-center h-full">
              <p className="text-gray-500 dark:text-gray-400">Chart will be displayed here</p>
            </div>
          </div>
        </div>

        <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
          <StockRecommendations recommendations={[
            { asset: 'AAPL', action: 'Buy', price: '$184.25', confidence: '92%' },
            { asset: 'MSFT', action: 'Hold', price: '$405.68', confidence: '87%' },
            { asset: 'AMZN', action: 'Buy', price: '$175.35', confidence: '89%' },
          ]} />
          
          <div>
            <h2 className="mb-4 text-xl font-semibold">Market Sentiment</h2>
            <MarketSentiment 
              overallMood="Bullish"
              sectorSentiments={[
                { sector: 'Technology', sentiment: 'Strongly Bullish' },
                { sector: 'Finance', sentiment: 'Neutral' },
                { sector: 'Healthcare', sentiment: 'Bullish' },
                { sector: 'Energy', sentiment: 'Bearish' },
              ]}
            />
          </div>
        </div>

        <PortfolioDashboard 
          portfolioValue={sampleData.portfolioValue}
          portfolio={sampleData.portfolio}
          watchlist={sampleData.watchlist}
          recentTrades={sampleData.recentTrades}
        />
      </div>
    </>
  );
} 