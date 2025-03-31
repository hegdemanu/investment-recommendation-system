import React from 'react';
import Navbar from '@/components/Navbar';
import MarketSentimentAnalysis from '@/components/MarketSentimentAnalysis';

export const metadata = {
  title: 'Market Analysis',
  description: 'Real-time market sentiment analysis and sector insights',
};

export default function AnalysisPage() {
  return (
    <>
      <Navbar />
      <div className="container mx-auto p-6 pt-20">
        <header className="mb-8">
          <h1 className="text-3xl font-bold">Market Analysis</h1>
          <p className="text-gray-600 dark:text-gray-400">
            AI-powered market sentiment analysis and sector-wise insights
          </p>
        </header>

        <div className="grid gap-6">
          <MarketSentimentAnalysis
            data={[
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
              {
                sector: 'Energy',
                sentiment: 0.35,
                newsCount: 75,
                trendDirection: 'down',
                topKeywords: ['Renewable', 'Oil', 'Climate'],
              },
              {
                sector: 'Consumer',
                sentiment: 0.55,
                newsCount: 110,
                trendDirection: 'neutral',
                topKeywords: ['Retail', 'E-commerce', 'Supply Chain'],
              },
            ]}
            overallSentiment={{
              score: 0.62,
              trend: 'bullish',
              confidence: 0.85,
            }}
          />

          <div className="grid gap-6 md:grid-cols-2">
            <div className="rounded-lg border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-800 dark:bg-gray-900">
              <h2 className="mb-4 text-xl font-semibold">Market Insights</h2>
              <div className="space-y-4">
                <div className="rounded-lg bg-green-50 p-4 dark:bg-green-900/20">
                  <h3 className="mb-2 font-medium text-green-700 dark:text-green-400">
                    Bullish Signals
                  </h3>
                  <ul className="list-inside list-disc space-y-2 text-gray-600 dark:text-gray-400">
                    <li>Strong technology sector performance</li>
                    <li>Positive healthcare sector momentum</li>
                    <li>Increasing institutional investment</li>
                  </ul>
                </div>
                <div className="rounded-lg bg-red-50 p-4 dark:bg-red-900/20">
                  <h3 className="mb-2 font-medium text-red-700 dark:text-red-400">
                    Risk Factors
                  </h3>
                  <ul className="list-inside list-disc space-y-2 text-gray-600 dark:text-gray-400">
                    <li>Energy sector volatility</li>
                    <li>Financial sector uncertainty</li>
                    <li>Global supply chain concerns</li>
                  </ul>
                </div>
              </div>
            </div>

            <div className="rounded-lg border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-800 dark:bg-gray-900">
              <h2 className="mb-4 text-xl font-semibold">News Impact</h2>
              <div className="space-y-4">
                <div className="border-b border-gray-200 pb-4 dark:border-gray-700">
                  <h3 className="mb-2 font-medium">Technology Sector</h3>
                  <p className="text-sm text-gray-600 dark:text-gray-400">
                    AI and cloud computing continue to drive growth, with major tech companies 
                    reporting strong earnings. Cybersecurity concerns remain a key focus.
                  </p>
                </div>
                <div className="border-b border-gray-200 pb-4 dark:border-gray-700">
                  <h3 className="mb-2 font-medium">Healthcare Innovation</h3>
                  <p className="text-sm text-gray-600 dark:text-gray-400">
                    Biotech breakthroughs and digital health adoption are creating new 
                    opportunities. Medicare policy changes could impact the sector.
                  </p>
                </div>
                <div>
                  <h3 className="mb-2 font-medium">Financial Markets</h3>
                  <p className="text-sm text-gray-600 dark:text-gray-400">
                    Fintech disruption continues while traditional banking faces challenges. 
                    Cryptocurrency markets show increased institutional interest.
                  </p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </>
  );
} 