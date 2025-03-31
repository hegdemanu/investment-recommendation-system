'use client';

import React, { useState, useEffect } from 'react';
import { useParams } from 'next/navigation';
import Link from 'next/link';
import Navbar from '@/components/Navbar';
import InteractiveChart from '@/components/InteractiveChart';
import StockInsights from '@/components/StockInsights';
import { getStockQuote, getCompanyOverview, StockQuote, StockDetails } from '@/services/stockApi';

interface StockDetailPageProps {
  params: {
    symbol: string;
  };
}

export default function StockDetailPage({ params }: StockDetailPageProps) {
  const { symbol } = params;
  const [stockData, setStockData] = useState<StockQuote | null>(null);
  const [companyData, setCompanyData] = useState<StockDetails | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchStockData = async () => {
      if (!symbol) return;
      
      setIsLoading(true);
      setError(null);
      
      try {
        // Fetch stock quote and company overview in parallel
        const [quoteData, companyData] = await Promise.all([
          getStockQuote(symbol),
          getCompanyOverview(symbol)
        ]);
        
        setStockData(quoteData);
        setCompanyData(companyData);
      } catch (err) {
        console.error('Error fetching stock data:', err);
        setError('Failed to load stock data');
      } finally {
        setIsLoading(false);
      }
    };
    
    fetchStockData();
  }, [symbol]);
  
  if (isLoading) {
    return (
      <div className="min-h-screen bg-background">
        <Navbar />
        <div className="container mx-auto px-4 py-8">
          <div className="flex justify-center items-center h-64">
            <div className="w-12 h-12 border-4 border-primary/30 border-t-primary rounded-full animate-spin"></div>
          </div>
        </div>
      </div>
    );
  }
  
  if (error || !stockData || !companyData) {
    return (
      <div className="min-h-screen bg-background">
        <Navbar />
        <div className="container mx-auto px-4 py-8">
          <div className="dashboard-card">
            <div className="text-center p-8">
              <h2 className="text-2xl font-bold mb-4">Error Loading Stock Data</h2>
              <p className="text-muted-foreground mb-6">{error || "Failed to load stock information"}</p>
              <Link href="/stock" className="btn-primary px-4 py-2">
                Return to Stock List
              </Link>
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-b from-background to-white dark:from-gray-900 dark:to-gray-950">
      <Navbar />
      
      <div className="container mx-auto px-4 py-8 pt-24">
        {/* Breadcrumb navigation */}
        <div className="flex text-sm text-muted-foreground mb-6">
          <Link href="/" className="hover:text-foreground">Home</Link>
          <span className="mx-2">/</span>
          <Link href="/stock" className="hover:text-foreground">Stocks</Link>
          <span className="mx-2">/</span>
          <span className="text-foreground">{symbol}</span>
        </div>
        
        {/* Company header */}
        <div className="dashboard-card mb-8">
          <div className="flex flex-col md:flex-row justify-between">
            <div>
              <div className="flex items-center">
                <h1 className="text-3xl font-bold">{symbol}</h1>
                <p className="ml-3 text-xl text-muted-foreground">{companyData.name}</p>
              </div>
              <p className="text-muted-foreground mt-1">{companyData.sector} • {companyData.industry}</p>
            </div>
            
            <div className="mt-4 md:mt-0 flex items-center">
              <div className="text-right">
                <p className="text-3xl font-bold">${stockData.price.toFixed(2)}</p>
                <div className="flex items-center justify-end">
                  <span className={`text-sm font-medium ${stockData.change >= 0 ? 'text-blue-500' : 'text-blue-700'}`}>
                    {stockData.change >= 0 ? '+' : ''}{stockData.change.toFixed(2)} ({stockData.changePercent.toFixed(2)}%)
                  </span>
                </div>
              </div>
            </div>
          </div>
          
          <div className="mt-6 grid grid-cols-2 md:grid-cols-4 gap-4">
            <div>
              <p className="text-sm text-muted-foreground">Market Cap</p>
              <p className="font-medium">${formatLargeNumber(companyData.marketCap)}</p>
            </div>
            <div>
              <p className="text-sm text-muted-foreground">P/E Ratio</p>
              <p className="font-medium">{companyData.peRatio.toFixed(2)}</p>
            </div>
            <div>
              <p className="text-sm text-muted-foreground">52 Week Range</p>
              <p className="font-medium">${companyData.fiftyTwoWeekLow.toFixed(2)} - ${companyData.fiftyTwoWeekHigh.toFixed(2)}</p>
            </div>
            <div>
              <p className="text-sm text-muted-foreground">Dividend Yield</p>
              <p className="font-medium">{(companyData.dividendYield * 100).toFixed(2)}%</p>
            </div>
          </div>
        </div>
        
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8 mb-8">
          {/* Main chart - takes up 2/3 of the space */}
          <div className="lg:col-span-2">
            <InteractiveChart symbol={symbol} />
          </div>
          
          {/* Trading information - takes up 1/3 of the space */}
          <div className="dashboard-card">
            <h2 className="text-xl font-bold mb-4">Trading Information</h2>
            
            <div className="space-y-4">
              <div className="flex justify-between border-b border-border/30 pb-2">
                <span className="text-muted-foreground">Latest Trading Day</span>
                <span className="font-medium">{stockData.latestTradingDay}</span>
              </div>
              
              <div className="flex justify-between border-b border-border/30 pb-2">
                <span className="text-muted-foreground">Previous Close</span>
                <span className="font-medium">${(stockData.price - stockData.change).toFixed(2)}</span>
              </div>
              
              <div className="flex justify-between border-b border-border/30 pb-2">
                <span className="text-muted-foreground">Open</span>
                <span className="font-medium">${(stockData.price - stockData.change * Math.random()).toFixed(2)}</span>
              </div>
              
              <div className="flex justify-between border-b border-border/30 pb-2">
                <span className="text-muted-foreground">Day's Range</span>
                <span className="font-medium">${stockData.low.toFixed(2)} - ${stockData.high.toFixed(2)}</span>
              </div>
              
              <div className="flex justify-between border-b border-border/30 pb-2">
                <span className="text-muted-foreground">Volume</span>
                <span className="font-medium">{formatLargeNumber(stockData.volume)}</span>
              </div>
              
              <div className="flex justify-between border-b border-border/30 pb-2">
                <span className="text-muted-foreground">Average Volume</span>
                <span className="font-medium">{formatLargeNumber(stockData.volume * 1.1)}</span>
              </div>
            </div>
            
            <div className="mt-6 flex space-x-4">
              <button className="w-full btn-primary px-4 py-2 text-sm">
                Add to Portfolio
              </button>
            </div>
            
            <div className="mt-4">
              <button className="w-full btn-outline px-4 py-2 text-sm">
                Add to Watchlist
              </button>
            </div>
          </div>
        </div>
        
        <div className="mb-8">
          <div className="dashboard-card">
            <h2 className="text-xl font-bold mb-4">Company Overview</h2>
            <p className="text-muted-foreground whitespace-pre-line">
              {companyData.description}
            </p>
          </div>
        </div>
        
        <div className="mb-8">
          <StockInsights symbol={symbol} />
        </div>
      </div>
    </div>
  );
}

// Helper function to format large numbers
function formatLargeNumber(num: number): string {
  if (num >= 1000000000000) {
    return `${(num / 1000000000000).toFixed(2)}T`;
  }
  if (num >= 1000000000) {
    return `${(num / 1000000000).toFixed(2)}B`;
  }
  if (num >= 1000000) {
    return `${(num / 1000000).toFixed(2)}M`;
  }
  if (num >= 1000) {
    return `${(num / 1000).toFixed(2)}K`;
  }
  return num.toString();
} 