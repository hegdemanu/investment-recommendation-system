'use client';

import React, { useState, useEffect } from 'react';
import Link from 'next/link';
import Navbar from '@/components/Navbar';
import StockSearch from '@/components/StockSearch';
import { getStockQuote, StockQuote } from '@/services/stockApi';

export default function StocksPage() {
  const [stocks, setStocks] = useState<StockQuote[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [sector, setSector] = useState<string>('all');
  const [sortBy, setSortBy] = useState<string>('symbol');
  const [sortOrder, setSortOrder] = useState<'asc' | 'desc'>('asc');
  
  // Default stocks to load
  const defaultStocks = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 
    'TSLA', 'NVDA', 'JPM', 'V', 'WMT',
    'JNJ', 'PG', 'UNH', 'HD', 'BAC'
  ];
  
  // Sector mapping (mock data since the API doesn't provide real-time sector performance)
  const sectors = {
    all: 'All Sectors',
    technology: 'Technology',
    healthcare: 'Healthcare',
    financial: 'Financial Services',
    consumer: 'Consumer Goods',
    industrial: 'Industrial',
    energy: 'Energy'
  };
  
  // Mock sector assignments for stocks
  const stockSectors: Record<string, string> = {
    'AAPL': 'technology',
    'MSFT': 'technology',
    'GOOGL': 'technology',
    'AMZN': 'consumer',
    'META': 'technology',
    'TSLA': 'consumer',
    'NVDA': 'technology',
    'JPM': 'financial',
    'V': 'financial',
    'WMT': 'consumer',
    'JNJ': 'healthcare',
    'PG': 'consumer',
    'UNH': 'healthcare',
    'HD': 'consumer',
    'BAC': 'financial'
  };
  
  useEffect(() => {
    const fetchStocks = async () => {
      setIsLoading(true);
      setError(null);
      
      try {
        // Fetch all stocks in parallel
        const results = await Promise.all(defaultStocks.map(symbol => getStockQuote(symbol)));
        setStocks(results);
      } catch (err) {
        console.error('Error fetching stock data:', err);
        setError('Failed to load stock data');
      } finally {
        setIsLoading(false);
      }
    };
    
    fetchStocks();
  }, []);
  
  const handleAddStock = async (symbol: string) => {
    // Skip if stock is already in list
    if (stocks.some(stock => stock.symbol === symbol)) {
      return;
    }
    
    try {
      setIsLoading(true);
      const newStock = await getStockQuote(symbol);
      setStocks(prev => [...prev, newStock]);
    } catch (err) {
      console.error('Error adding stock:', err);
      setError(`Failed to add ${symbol}`);
    } finally {
      setIsLoading(false);
    }
  };
  
  // Filter stocks by sector
  const filteredStocks = stocks.filter(stock => {
    if (sector === 'all') return true;
    return stockSectors[stock.symbol] === sector;
  });
  
  // Sort stocks
  const sortedStocks = [...filteredStocks].sort((a, b) => {
    let comparison = 0;
    
    switch (sortBy) {
      case 'symbol':
        comparison = a.symbol.localeCompare(b.symbol);
        break;
      case 'price':
        comparison = a.price - b.price;
        break;
      case 'change':
        comparison = a.change - b.change;
        break;
      case 'changePercent':
        comparison = a.changePercent - b.changePercent;
        break;
      case 'volume':
        comparison = a.volume - b.volume;
        break;
      default:
        comparison = 0;
    }
    
    return sortOrder === 'asc' ? comparison : -comparison;
  });
  
  // Handle sort changes
  const handleSort = (field: string) => {
    if (sortBy === field) {
      // Toggle sort order if same field clicked
      setSortOrder(sortOrder === 'asc' ? 'desc' : 'asc');
    } else {
      // Set new sort field and reset to ascending
      setSortBy(field);
      setSortOrder('asc');
    }
  };
  
  return (
    <div className="min-h-screen bg-gradient-to-b from-background to-white dark:from-gray-900 dark:to-gray-950">
      <Navbar />
      
      <div className="container mx-auto px-4 py-8 pt-24">
        <div className="flex flex-col md:flex-row justify-between items-center mb-8">
          <h1 className="text-3xl font-bold mb-4 md:mb-0">Stock Explorer</h1>
          <StockSearch 
            onSelectStock={handleAddStock}
            placeholder="Search for a stock..."
            className="w-full md:w-80"
          />
        </div>
        
        {error && (
          <div className="bg-red-100 dark:bg-red-900/20 text-red-800 dark:text-red-200 p-4 rounded-md mb-6">
            {error}
          </div>
        )}
        
        {/* Sector filter */}
        <div className="flex flex-wrap gap-2 mb-6">
          {Object.entries(sectors).map(([key, label]) => (
            <button
              key={key}
              onClick={() => setSector(key)}
              className={`px-3 py-1 text-sm rounded-full ${
                sector === key
                  ? 'bg-primary text-white'
                  : 'bg-muted text-muted-foreground hover:bg-muted/80'
              }`}
            >
              {label}
            </button>
          ))}
        </div>
        
        {/* Stocks table */}
        <div className="dashboard-card overflow-hidden">
          {isLoading && stocks.length === 0 ? (
            <div className="flex justify-center items-center p-12">
              <div className="w-10 h-10 border-4 border-primary/30 border-t-primary rounded-full animate-spin"></div>
            </div>
          ) : stocks.length === 0 ? (
            <div className="text-center p-12">
              <p className="text-muted-foreground">No stocks found</p>
              <p className="text-sm text-muted-foreground mt-2">Try searching for a stock to add it to the list</p>
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="border-b border-border">
                    <th className="p-4 text-left">
                      <button 
                        className="flex items-center font-medium"
                        onClick={() => handleSort('symbol')}
                      >
                        Symbol
                        {sortBy === 'symbol' && (
                          <span className="ml-1">{sortOrder === 'asc' ? '↑' : '↓'}</span>
                        )}
                      </button>
                    </th>
                    <th className="p-4 text-left">Company</th>
                    <th className="p-4 text-right">
                      <button 
                        className="flex items-center font-medium ml-auto"
                        onClick={() => handleSort('price')}
                      >
                        Price
                        {sortBy === 'price' && (
                          <span className="ml-1">{sortOrder === 'asc' ? '↑' : '↓'}</span>
                        )}
                      </button>
                    </th>
                    <th className="p-4 text-right">
                      <button 
                        className="flex items-center font-medium ml-auto"
                        onClick={() => handleSort('change')}
                      >
                        Change
                        {sortBy === 'change' && (
                          <span className="ml-1">{sortOrder === 'asc' ? '↑' : '↓'}</span>
                        )}
                      </button>
                    </th>
                    <th className="p-4 text-right">
                      <button 
                        className="flex items-center font-medium ml-auto"
                        onClick={() => handleSort('changePercent')}
                      >
                        % Change
                        {sortBy === 'changePercent' && (
                          <span className="ml-1">{sortOrder === 'asc' ? '↑' : '↓'}</span>
                        )}
                      </button>
                    </th>
                    <th className="p-4 text-right">
                      <button 
                        className="flex items-center font-medium ml-auto"
                        onClick={() => handleSort('volume')}
                      >
                        Volume
                        {sortBy === 'volume' && (
                          <span className="ml-1">{sortOrder === 'asc' ? '↑' : '↓'}</span>
                        )}
                      </button>
                    </th>
                    <th className="p-4 text-right">Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {sortedStocks.map((stock) => (
                    <tr 
                      key={stock.symbol}
                      className="border-b border-border/40 last:border-0 hover:bg-muted/50 transition-colors"
                    >
                      <td className="p-4">
                        <Link 
                          href={`/stock/${stock.symbol}`}
                          className="font-medium text-primary hover:underline"
                        >
                          {stock.symbol}
                        </Link>
                      </td>
                      <td className="p-4 text-muted-foreground">
                        {sectors[(stockSectors[stock.symbol] || 'technology') as keyof typeof sectors]}
                      </td>
                      <td className="p-4 text-right">
                        ${stock.price.toFixed(2)}
                      </td>
                      <td className={stock.change >= 0 ? 'text-blue-500 p-4 text-right' : 'text-blue-700 p-4 text-right'}>
                        {stock.change >= 0 ? '+' : ''}{stock.change.toFixed(2)}
                      </td>
                      <td className={stock.change >= 0 ? 'text-blue-500 p-4 text-right' : 'text-blue-700 p-4 text-right'}>
                        {stock.change >= 0 ? '+' : ''}{stock.changePercent.toFixed(2)}%
                      </td>
                      <td className="p-4 text-right">
                        {formatLargeNumber(stock.volume)}
                      </td>
                      <td className="p-4 text-right">
                        <div className="flex justify-end space-x-2">
                          <button
                            className="p-1 text-muted-foreground hover:text-primary"
                            title="Add to watchlist"
                          >
                            <svg
                              xmlns="http://www.w3.org/2000/svg"
                              width="16"
                              height="16"
                              viewBox="0 0 24 24"
                              fill="none"
                              stroke="currentColor"
                              strokeWidth="2"
                              strokeLinecap="round"
                              strokeLinejoin="round"
                            >
                              <path d="M19 21l-7-5-7 5V5a2 2 0 0 1 2-2h10a2 2 0 0 1 2 2z"></path>
                            </svg>
                          </button>
                          <button
                            className="p-1 text-muted-foreground hover:text-primary"
                            title="Add to portfolio"
                          >
                            <svg
                              xmlns="http://www.w3.org/2000/svg"
                              width="16"
                              height="16"
                              viewBox="0 0 24 24"
                              fill="none"
                              stroke="currentColor"
                              strokeWidth="2"
                              strokeLinecap="round"
                              strokeLinejoin="round"
                            >
                              <rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect>
                              <line x1="12" y1="8" x2="12" y2="16"></line>
                              <line x1="8" y1="12" x2="16" y2="12"></line>
                            </svg>
                          </button>
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
        
        {/* Market overview */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-8 mt-8">
          <div className="dashboard-card">
            <h2 className="text-xl font-bold mb-4">Market Movers</h2>
            <div className="space-y-4">
              <div>
                <h3 className="font-medium text-primary mb-2">Top Gainers</h3>
                <div className="space-y-2">
                  {sortedStocks
                    .filter(stock => stock.change > 0)
                    .sort((a, b) => b.changePercent - a.changePercent)
                    .slice(0, 3)
                    .map(stock => (
                      <div key={stock.symbol} className="flex justify-between items-center p-2 border-b border-border/30 last:border-0">
                        <div>
                          <Link 
                            href={`/stock/${stock.symbol}`}
                            className="font-medium text-primary hover:underline"
                          >
                            {stock.symbol}
                          </Link>
                          <p className="text-xs text-muted-foreground">${stock.price.toFixed(2)}</p>
                        </div>
                        <div className="text-blue-500">
                          +{stock.changePercent.toFixed(2)}%
                        </div>
                      </div>
                    ))}
                </div>
              </div>
              
              <div>
                <h3 className="font-medium text-primary mb-2">Top Losers</h3>
                <div className="space-y-2">
                  {sortedStocks
                    .filter(stock => stock.change < 0)
                    .sort((a, b) => a.changePercent - b.changePercent)
                    .slice(0, 3)
                    .map(stock => (
                      <div key={stock.symbol} className="flex justify-between items-center p-2 border-b border-border/30 last:border-0">
                        <div>
                          <Link 
                            href={`/stock/${stock.symbol}`}
                            className="font-medium text-primary hover:underline"
                          >
                            {stock.symbol}
                          </Link>
                          <p className="text-xs text-muted-foreground">${stock.price.toFixed(2)}</p>
                        </div>
                        <div className="text-blue-700">
                          {stock.changePercent.toFixed(2)}%
                        </div>
                      </div>
                    ))}
                </div>
              </div>
            </div>
          </div>
          
          <div className="dashboard-card">
            <h2 className="text-xl font-bold mb-4">Sector Performance</h2>
            {/* Mock sector performance data */}
            {Object.entries(sectors)
              .filter(([key]) => key !== 'all')
              .map(([key, label]) => {
                // Calculate average performance for each sector
                const sectorStocks = stocks.filter(stock => stockSectors[stock.symbol] === key);
                const avgChange = sectorStocks.length > 0
                  ? sectorStocks.reduce((sum, stock) => sum + stock.changePercent, 0) / sectorStocks.length
                  : 0;
                
                return (
                  <div key={key} className="flex justify-between items-center p-3 border-b border-border/30 last:border-0">
                    <div>
                      <p className="font-medium">{label}</p>
                      <p className="text-xs text-muted-foreground">{sectorStocks.length} stocks</p>
                    </div>
                    <div className={avgChange >= 0 ? 'text-blue-500' : 'text-blue-700'}>
                      {avgChange >= 0 ? '+' : ''}{avgChange.toFixed(2)}%
                    </div>
                  </div>
                );
              })}
          </div>
        </div>
      </div>
    </div>
  );
}

// Helper function to format large numbers
function formatLargeNumber(num: number): string {
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