'use client';

import React, { useState, useEffect } from 'react';
import Link from 'next/link';
import Navbar from '@/components/Navbar';
import InteractiveChart from '@/components/InteractiveChart';
import StockSearch from '@/components/StockSearch';
import { getStockQuote, StockQuote } from '@/services/stockApi';

export default function DashboardPage() {
  const [portfolioStocks, setPortfolioStocks] = useState<StockQuote[]>([]);
  const [watchlistStocks, setWatchlistStocks] = useState<StockQuote[]>([]);
  const [selectedStock, setSelectedStock] = useState<string>('AAPL');
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  
  // Default stocks for portfolio and watchlist
  const defaultPortfolio = ['AAPL', 'MSFT', 'GOOGL'];
  const defaultWatchlist = ['AMZN', 'TSLA', 'META'];
  
  useEffect(() => {
    const fetchStocks = async () => {
      setIsLoading(true);
      setError(null);
      
      try {
        // Fetch all stocks in parallel
        const allSymbols = [...defaultPortfolio, ...defaultWatchlist];
        const results = await Promise.all(allSymbols.map(symbol => getStockQuote(symbol)));
        
        // Split results into portfolio and watchlist
        const portfolioData = results.slice(0, defaultPortfolio.length);
        const watchlistData = results.slice(defaultPortfolio.length);
        
        setPortfolioStocks(portfolioData);
        setWatchlistStocks(watchlistData);
      } catch (err) {
        console.error('Error fetching stock data:', err);
        setError('Failed to load stock data');
      } finally {
        setIsLoading(false);
      }
    };
    
    fetchStocks();
  }, []);
  
  const handleAddToWatchlist = async (symbol: string) => {
    // Skip if stock is already in watchlist
    if (watchlistStocks.some(stock => stock.symbol === symbol) || 
        portfolioStocks.some(stock => stock.symbol === symbol)) {
      return;
    }
    
    try {
      setIsLoading(true);
      const newStock = await getStockQuote(symbol);
      setWatchlistStocks(prev => [...prev, newStock]);
    } catch (err) {
      console.error('Error adding stock to watchlist:', err);
      setError(`Failed to add ${symbol} to watchlist`);
    } finally {
      setIsLoading(false);
    }
  };
  
  const handleRemoveFromWatchlist = (symbol: string) => {
    setWatchlistStocks(prev => prev.filter(stock => stock.symbol !== symbol));
  };
  
  // Calculate portfolio performance
  const calculatePortfolioPerformance = () => {
    if (portfolioStocks.length === 0) return { value: 0, change: 0, changePercent: 0 };
    
    const totalValue = portfolioStocks.reduce((sum, stock) => sum + stock.price, 0);
    const totalChange = portfolioStocks.reduce((sum, stock) => sum + stock.change, 0);
    const changePercent = (totalChange / (totalValue - totalChange)) * 100;
    
    return {
      value: totalValue,
      change: totalChange,
      changePercent
    };
  };
  
  const performance = calculatePortfolioPerformance();
  
  // Mock recent trades
  const recentTrades = [
    { date: '2025-03-27', symbol: 'AAPL', action: 'BUY', shares: 10, price: 185.92 },
    { date: '2025-03-25', symbol: 'TSLA', action: 'SELL', shares: 5, price: 175.34 },
    { date: '2025-03-22', symbol: 'MSFT', action: 'BUY', shares: 8, price: 416.42 }
  ];

  return (
    <div className="min-h-screen bg-gradient-to-b from-background to-white dark:from-gray-900 dark:to-gray-950">
      <Navbar />
      
      <div className="container mx-auto px-4 py-8 pt-24">
        <div className="flex flex-col md:flex-row justify-between items-center mb-8">
          <h1 className="text-3xl font-bold mb-4 md:mb-0">Investment Dashboard</h1>
          <StockSearch 
            onSelectStock={handleAddToWatchlist}
            placeholder="Add a stock to watchlist..."
            className="w-full md:w-64"
          />
        </div>
        
        {error && (
          <div className="bg-red-100 dark:bg-red-900/20 text-red-800 dark:text-red-200 p-4 rounded-md mb-6">
            {error}
          </div>
        )}
        
        {/* Portfolio Overview */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
          <div className="dashboard-stat-card">
            <div className="dashboard-metric-label">Portfolio Value</div>
            <div className="dashboard-metric">${performance.value.toFixed(2)}</div>
            <div className={performance.change >= 0 ? 'trend-up' : 'trend-down'}>
              {performance.change >= 0 ? '↑' : '↓'} ${Math.abs(performance.change).toFixed(2)} ({Math.abs(performance.changePercent).toFixed(2)}%)
            </div>
          </div>
          
          <div className="dashboard-stat-card">
            <div className="dashboard-metric-label">Holdings</div>
            <div className="dashboard-metric">{portfolioStocks.length}</div>
            <div className="text-sm text-muted-foreground">Active positions</div>
          </div>
          
          <div className="dashboard-stat-card">
            <div className="dashboard-metric-label">Watchlist</div>
            <div className="dashboard-metric">{watchlistStocks.length}</div>
            <div className="text-sm text-muted-foreground">Tracked stocks</div>
          </div>
          
          <div className="dashboard-stat-card">
            <div className="dashboard-metric-label">Market Trend</div>
            <div className="dashboard-metric">
              <span className="text-blue-500">Bullish</span>
            </div>
            <div className="text-sm text-muted-foreground">Overall market analysis</div>
          </div>
        </div>
        
        {/* Stock Chart */}
        <div className="mb-8">
          <div className="flex justify-between items-center mb-4">
            <h2 className="text-xl font-bold">Performance Chart</h2>
            <div className="flex space-x-2">
              {portfolioStocks.map(stock => (
                <button
                  key={stock.symbol}
                  className={`px-3 py-1 text-xs rounded-md ${
                    selectedStock === stock.symbol 
                      ? 'bg-primary text-white' 
                      : 'bg-muted text-muted-foreground'
                  }`}
                  onClick={() => setSelectedStock(stock.symbol)}
                >
                  {stock.symbol}
                </button>
              ))}
            </div>
          </div>
          
          <InteractiveChart symbol={selectedStock} />
        </div>
        
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
          {/* Portfolio Holdings */}
          <div className="dashboard-card">
            <h2 className="text-xl font-bold mb-4">My Portfolio</h2>
            
            {isLoading && portfolioStocks.length === 0 ? (
              <div className="flex justify-center items-center p-8">
                <div className="w-8 h-8 border-4 border-primary/30 border-t-primary rounded-full animate-spin"></div>
              </div>
            ) : portfolioStocks.length === 0 ? (
              <div className="text-center p-8">
                <p className="text-muted-foreground">No stocks in your portfolio</p>
              </div>
            ) : (
              <div className="overflow-x-auto">
                <table className="dashboard-table">
                  <thead>
                    <tr>
                      <th>Symbol</th>
                      <th>Price</th>
                      <th>Change</th>
                      <th>Value</th>
                    </tr>
                  </thead>
                  <tbody>
                    {portfolioStocks.map((holding) => (
                      <tr key={holding.symbol}>
                        <td>
                          <Link 
                            href={`/stock/${holding.symbol}`}
                            className="font-medium text-primary hover:underline"
                          >
                            {holding.symbol}
                          </Link>
                        </td>
                        <td>${holding.price.toFixed(2)}</td>
                        <td className={holding.change >= 0 ? 'text-blue-500' : 'text-blue-700'}>
                          {holding.change >= 0 ? '+' : ''}{holding.change.toFixed(2)} ({holding.changePercent.toFixed(2)}%)
                        </td>
                        <td>${(holding.price * 10).toFixed(2)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
            
            <div className="mt-4 flex justify-center">
              <Link 
                href="/stock" 
                className="text-primary hover:text-primary/80 text-sm font-medium"
              >
                Manage Portfolio →
              </Link>
            </div>
          </div>
          
          {/* Watchlist */}
          <div className="dashboard-card">
            <h2 className="text-xl font-bold mb-4">My Watchlist</h2>
            
            {isLoading && watchlistStocks.length === 0 ? (
              <div className="flex justify-center items-center p-8">
                <div className="w-8 h-8 border-4 border-primary/30 border-t-primary rounded-full animate-spin"></div>
              </div>
            ) : watchlistStocks.length === 0 ? (
              <div className="text-center p-8">
                <p className="text-muted-foreground">Your watchlist is empty</p>
                <p className="text-sm text-muted-foreground mt-2">Use the search box to add stocks</p>
              </div>
            ) : (
              <div className="overflow-x-auto">
                <table className="dashboard-table">
                  <thead>
                    <tr>
                      <th>Symbol</th>
                      <th>Price</th>
                      <th>Change</th>
                      <th>Actions</th>
                    </tr>
                  </thead>
                  <tbody>
                    {watchlistStocks.map((stock) => (
                      <tr key={stock.symbol}>
                        <td>
                          <Link 
                            href={`/stock/${stock.symbol}`}
                            className="font-medium text-primary hover:underline"
                          >
                            {stock.symbol}
                          </Link>
                        </td>
                        <td>${stock.price.toFixed(2)}</td>
                        <td className={stock.change >= 0 ? 'text-blue-500' : 'text-blue-700'}>
                          {stock.change >= 0 ? '+' : ''}{stock.change.toFixed(2)} ({stock.changePercent.toFixed(2)}%)
                        </td>
                        <td>
                          <div className="flex space-x-2">
                            <button
                              onClick={() => handleRemoveFromWatchlist(stock.symbol)}
                              className="text-muted-foreground hover:text-destructive"
                              title="Remove from watchlist"
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
                                <path d="M3 6h18"></path>
                                <path d="M19 6v14c0 1-1 2-2 2H7c-1 0-2-1-2-2V6"></path>
                                <path d="M8 6V4c0-1 1-2 2-2h4c1 0 2 1 2 2v2"></path>
                                <line x1="10" y1="11" x2="10" y2="17"></line>
                                <line x1="14" y1="11" x2="14" y2="17"></line>
                              </svg>
                            </button>
                            <button
                              className="text-muted-foreground hover:text-primary"
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
            
            <div className="mt-4 flex justify-center">
              <Link 
                href="/stock" 
                className="text-primary hover:text-primary/80 text-sm font-medium"
              >
                Browse More Stocks →
              </Link>
            </div>
          </div>
        </div>
        
        {/* Recent Trades */}
        <div className="dashboard-card mb-8">
          <h2 className="text-xl font-bold mb-4">Recent Trades</h2>
          
          <div className="overflow-x-auto">
            <table className="dashboard-table">
              <thead>
                <tr>
                  <th>Date</th>
                  <th>Symbol</th>
                  <th>Action</th>
                  <th>Shares</th>
                  <th>Price</th>
                  <th>Total</th>
                </tr>
              </thead>
              <tbody>
                {recentTrades.map((trade, index) => (
                  <tr key={index}>
                    <td>{trade.date}</td>
                    <td>
                      <Link 
                        href={`/stock/${trade.symbol}`}
                        className="font-medium text-primary hover:underline"
                      >
                        {trade.symbol}
                      </Link>
                    </td>
                    <td className={trade.action === 'BUY' ? 'text-blue-500' : 'text-blue-700'}>
                      {trade.action}
                    </td>
                    <td>{trade.shares}</td>
                    <td>${trade.price.toFixed(2)}</td>
                    <td>${(trade.shares * trade.price).toFixed(2)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
        
        {/* Quick Actions */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="dashboard-card text-center">
            <div className="flex justify-center mb-4">
              <div className="h-12 w-12 bg-primary/10 rounded-full flex items-center justify-center">
                <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-primary">
                  <rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect>
                  <line x1="12" y1="8" x2="12" y2="16"></line>
                  <line x1="8" y1="12" x2="16" y2="12"></line>
                </svg>
              </div>
            </div>
            <h3 className="text-lg font-bold mb-2">Add New Stock</h3>
            <p className="text-sm text-muted-foreground mb-4">Discover and add new stocks to your portfolio</p>
            <Link
              href="/stock"
              className="inline-block px-4 py-2 bg-primary/10 text-primary rounded-md hover:bg-primary/20 transition-colors"
            >
              Explore Stocks
            </Link>
          </div>
          
          <div className="dashboard-card text-center">
            <div className="flex justify-center mb-4">
              <div className="h-12 w-12 bg-secondary/10 rounded-full flex items-center justify-center">
                <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-secondary">
                  <path d="M22 12h-4l-3 9L9 3l-3 9H2"></path>
                </svg>
              </div>
            </div>
            <h3 className="text-lg font-bold mb-2">Performance Analytics</h3>
            <p className="text-sm text-muted-foreground mb-4">Detailed analysis of your portfolio performance</p>
            <button
              className="inline-block px-4 py-2 bg-secondary/10 text-secondary rounded-md hover:bg-secondary/20 transition-colors"
            >
              View Analytics
            </button>
          </div>
          
          <div className="dashboard-card text-center">
            <div className="flex justify-center mb-4">
              <div className="h-12 w-12 bg-accent/10 rounded-full flex items-center justify-center">
                <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-accent">
                  <circle cx="12" cy="12" r="10"></circle>
                  <polyline points="12 6 12 12 16 14"></polyline>
                </svg>
              </div>
            </div>
            <h3 className="text-lg font-bold mb-2">Price Alerts</h3>
            <p className="text-sm text-muted-foreground mb-4">Set up notifications for price changes</p>
            <button
              className="inline-block px-4 py-2 bg-accent/10 text-accent rounded-md hover:bg-accent/20 transition-colors"
            >
              Set Alerts
            </button>
          </div>
        </div>
      </div>
    </div>
  );
} 