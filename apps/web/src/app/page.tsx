'use client';

import { useState } from 'react';
import Link from 'next/link';
import Navbar from '@/components/Navbar';
import { Navigation } from '@/components/Navigation';
import CompoundCalculator from '@/components/CompoundCalculator';
import SentimentAnalysis from '@/components/SentimentAnalysis';
import ReportGenerator from '@/components/ReportGenerator';
import RAGDashboard from '@/components/RAGDashboard';

export default function HomePage() {
  const [activeTab, setActiveTab] = useState('compound');

  const renderActiveComponent = () => {
    switch (activeTab) {
      case 'compound':
        return <CompoundCalculator />;
      case 'sentiment':
        return <SentimentAnalysis />;
      case 'report':
        return <ReportGenerator />;
      case 'rag':
        return <RAGDashboard />;
      default:
        return null;
    }
  };

  return (
    <>
      <Navbar />
      <div className="bg-gradient-to-b from-blue-50 to-white min-h-screen">
        <div className="container mx-auto px-4 py-20">
          <div className="max-w-4xl mx-auto text-center mb-16">
            <h1 className="text-5xl font-bold text-blue-800 mb-6">
              Investment Recommendation System
            </h1>
            <p className="text-xl text-gray-600 mb-10">
              AI-powered investment analysis, predictions, and recommendations to optimize your
              portfolio and maximize returns.
            </p>

            <div className="flex flex-col sm:flex-row justify-center gap-4 mb-16">
              <Link
                href="/stock"
                className="bg-blue-600 hover:bg-blue-700 text-white font-medium rounded-lg px-8 py-3 text-lg transition-colors"
              >
                Stock Analysis
              </Link>
              <Link
                href="/dashboard"
                className="bg-white hover:bg-gray-100 text-blue-600 font-medium rounded-lg px-8 py-3 text-lg border border-blue-200 transition-colors"
              >
                Dashboard
              </Link>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-8 mb-16">
            <div className="bg-white p-6 rounded-lg shadow-sm">
              <div className="text-blue-600 text-2xl mb-4">🧠</div>
              <h3 className="text-xl font-semibold mb-2">AI-Driven Models</h3>
              <p className="text-gray-600">
                Dynamically switches between LSTM, ARIMA-GARCH, and Prophet models to provide the most
                accurate predictions for different market conditions.
              </p>
            </div>

            <div className="bg-white p-6 rounded-lg shadow-sm">
              <div className="text-blue-600 text-2xl mb-4">📈</div>
              <h3 className="text-xl font-semibold mb-2">Market Sentiment</h3>
              <p className="text-gray-600">
                Advanced sentiment analysis of news, social media, and financial reports to gauge
                market mood and predict potential stock movements.
              </p>
            </div>

            <div className="bg-white p-6 rounded-lg shadow-sm">
              <div className="text-blue-600 text-2xl mb-4">💬</div>
              <h3 className="text-xl font-semibold mb-2">AI Research Assistant</h3>
              <p className="text-gray-600">
                Ask questions about stocks, markets, or investment strategies and get insights powered
                by retrieval-augmented generation.
              </p>
            </div>
          </div>

          <Navigation activeTab={activeTab} onTabChange={setActiveTab} />
          
          <div className="mt-8 transition-all duration-300">
            {renderActiveComponent()}
          </div>
        </div>
      </div>
    </>
  );
} 