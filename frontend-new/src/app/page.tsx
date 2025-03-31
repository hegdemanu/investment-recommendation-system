'use client';

import { useState } from 'react';
import { Navigation } from '@/components/Navigation';
import CompoundCalculator from '@/components/CompoundCalculator';
import SentimentAnalysis from '@/components/SentimentAnalysis';
import ReportGenerator from '@/components/ReportGenerator';
import RAGDashboard from '@/components/RAGDashboard';
import { Button } from '@/components/ui/button';
import Link from 'next/link';

export default function Home() {
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
    <main className="flex min-h-screen flex-col items-center p-4 md:p-8">
      <div className="w-full max-w-7xl mx-auto space-y-8">
        <div className="flex items-center justify-between w-full">
          <Link href="/">
            <Button variant="ghost" className="flex items-center gap-2">
              <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"></path>
                <polyline points="9 22 9 12 15 12 15 22"></polyline>
              </svg>
              Home
            </Button>
          </Link>
          <Link href="/dashboard">
            <Button variant="outline" className="flex items-center gap-2">
              <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <rect x="3" y="3" width="7" height="7"></rect>
                <rect x="14" y="3" width="7" height="7"></rect>
                <rect x="14" y="14" width="7" height="7"></rect>
                <rect x="3" y="14" width="7" height="7"></rect>
              </svg>
              Dashboard
            </Button>
          </Link>
        </div>

        <div className="text-center space-y-4">
          <h1 className="text-4xl md:text-5xl font-bold bg-gradient-to-r from-purple-600 to-blue-500 bg-clip-text text-transparent">
            Investment Recommendation System
          </h1>
          <p className="text-lg text-gray-600 dark:text-gray-300 max-w-2xl mx-auto">
            Make informed investment decisions with our AI-powered tools and analytics
          </p>
        </div>

        <Navigation activeTab={activeTab} onTabChange={setActiveTab} />
        
        <div className="w-full bg-white dark:bg-gray-800 rounded-lg shadow-lg p-6">
          {renderActiveComponent()}
        </div>

        <div className="flex justify-center gap-4 mt-8">
          <Button 
            variant="default" 
            className="flex items-center gap-2 bg-gradient-to-r from-purple-600 to-blue-500 text-white"
            onClick={() => window.scrollTo({ top: 0, behavior: 'smooth' })}
          >
            <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M12 19V5M5 12l7-7 7 7"/>
            </svg>
            Back to Top
          </Button>
          <Link href="/dashboard">
            <Button variant="outline" className="flex items-center gap-2">
              View Full Dashboard
              <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M5 12h14M12 5l7 7-7 7"/>
              </svg>
            </Button>
          </Link>
        </div>
      </div>
    </main>
  );
} 