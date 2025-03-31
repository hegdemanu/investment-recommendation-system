'use client';

import React from 'react';
import Link from 'next/link';

interface Stock {
  symbol: string;
  price: number;
  change: number;
  changePercent: number;
  value?: number;
}

interface Trade {
  date: string;
  symbol: string;
  action: 'BUY' | 'SELL';
  shares: number;
  price: number;
  total: number;
}

interface PortfolioDashboardProps {
  portfolioValue: number;
  portfolio: Stock[];
  watchlist: Stock[];
  recentTrades: Trade[];
}

const PortfolioDashboard: React.FC<PortfolioDashboardProps> = ({
  portfolioValue,
  portfolio,
  watchlist,
  recentTrades,
}) => {
  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
    }).format(value);
  };

  const formatChange = (change: number, changePercent: number) => {
    const isPositive = change >= 0;
    return (
      <span className={`flex items-center ${isPositive ? 'text-green-600' : 'text-red-600'}`}>
        {isPositive ? '+' : ''}{formatCurrency(change)} ({isPositive ? '+' : ''}{changePercent.toFixed(2)}%)
      </span>
    );
  };

  const handleBuyStock = async (symbol: string) => {
    // TODO: Implement buy stock logic
    console.log(`Buying stock: ${symbol}`);
  };

  const handleRemoveFromWatchlist = async (symbol: string) => {
    // TODO: Implement remove from watchlist logic
    console.log(`Removing from watchlist: ${symbol}`);
  };

  return (
    <div className="space-y-6">
      {/* Portfolio Value */}
      <div className="rounded-lg bg-white p-6 shadow-sm dark:bg-gray-800">
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white">
          {formatCurrency(portfolioValue)}
        </h2>
        <p className="text-sm text-gray-500 dark:text-gray-400">Total Portfolio Value</p>
      </div>

      {/* Portfolio Section */}
      <div className="rounded-lg bg-white p-6 shadow-sm dark:bg-gray-800">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-xl font-semibold text-gray-900 dark:text-white">My Portfolio</h2>
          <Link 
            href="/manage-portfolio"
            className="text-sm text-blue-600 hover:text-blue-800 dark:text-blue-400 dark:hover:text-blue-300"
          >
            Manage Portfolio →
          </Link>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="border-b border-gray-200 dark:border-gray-700">
                <th className="py-3 text-left text-sm font-medium text-gray-500 dark:text-gray-400">Symbol</th>
                <th className="py-3 text-right text-sm font-medium text-gray-500 dark:text-gray-400">Price</th>
                <th className="py-3 text-right text-sm font-medium text-gray-500 dark:text-gray-400">Change</th>
                <th className="py-3 text-right text-sm font-medium text-gray-500 dark:text-gray-400">Value</th>
              </tr>
            </thead>
            <tbody>
              {portfolio.map((stock) => (
                <tr key={stock.symbol} className="border-b border-gray-100 dark:border-gray-700">
                  <td className="py-4">
                    <Link 
                      href={`/stock/${stock.symbol}`}
                      className="font-medium text-gray-900 hover:text-blue-600 dark:text-white dark:hover:text-blue-400"
                    >
                      {stock.symbol}
                    </Link>
                  </td>
                  <td className="py-4 text-right">{formatCurrency(stock.price)}</td>
                  <td className="py-4 text-right">{formatChange(stock.change, stock.changePercent)}</td>
                  <td className="py-4 text-right">{formatCurrency(stock.value || 0)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Watchlist Section */}
      <div className="rounded-lg bg-white p-6 shadow-sm dark:bg-gray-800">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-xl font-semibold text-gray-900 dark:text-white">My Watchlist</h2>
          <Link 
            href="/browse-stocks"
            className="text-sm text-blue-600 hover:text-blue-800 dark:text-blue-400 dark:hover:text-blue-300"
          >
            Browse More Stocks →
          </Link>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="border-b border-gray-200 dark:border-gray-700">
                <th className="py-3 text-left text-sm font-medium text-gray-500 dark:text-gray-400">Symbol</th>
                <th className="py-3 text-right text-sm font-medium text-gray-500 dark:text-gray-400">Price</th>
                <th className="py-3 text-right text-sm font-medium text-gray-500 dark:text-gray-400">Change</th>
                <th className="py-3 text-right text-sm font-medium text-gray-500 dark:text-gray-400">Actions</th>
              </tr>
            </thead>
            <tbody>
              {watchlist.map((stock) => (
                <tr key={stock.symbol} className="border-b border-gray-100 dark:border-gray-700">
                  <td className="py-4">
                    <Link 
                      href={`/stock/${stock.symbol}`}
                      className="font-medium text-gray-900 hover:text-blue-600 dark:text-white dark:hover:text-blue-400"
                    >
                      {stock.symbol}
                    </Link>
                  </td>
                  <td className="py-4 text-right">{formatCurrency(stock.price)}</td>
                  <td className="py-4 text-right">{formatChange(stock.change, stock.changePercent)}</td>
                  <td className="py-4 text-right">
                    <div className="flex justify-end space-x-2">
                      <button
                        className="rounded-md bg-blue-50 px-2.5 py-1.5 text-sm font-medium text-blue-600 hover:bg-blue-100 dark:bg-blue-900/20 dark:text-blue-400 dark:hover:bg-blue-900/30"
                        onClick={() => handleBuyStock(stock.symbol)}
                      >
                        Buy
                      </button>
                      <button
                        className="rounded-md bg-gray-50 px-2.5 py-1.5 text-sm font-medium text-gray-600 hover:bg-gray-100 dark:bg-gray-700 dark:text-gray-300 dark:hover:bg-gray-600"
                        onClick={() => handleRemoveFromWatchlist(stock.symbol)}
                      >
                        Remove
                      </button>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Recent Trades Section */}
      <div className="rounded-lg bg-white p-6 shadow-sm dark:bg-gray-800">
        <h2 className="text-xl font-semibold text-gray-900 dark:text-white mb-4">Recent Trades</h2>
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="border-b border-gray-200 dark:border-gray-700">
                <th className="py-3 text-left text-sm font-medium text-gray-500 dark:text-gray-400">Date</th>
                <th className="py-3 text-left text-sm font-medium text-gray-500 dark:text-gray-400">Symbol</th>
                <th className="py-3 text-right text-sm font-medium text-gray-500 dark:text-gray-400">Action</th>
                <th className="py-3 text-right text-sm font-medium text-gray-500 dark:text-gray-400">Shares</th>
                <th className="py-3 text-right text-sm font-medium text-gray-500 dark:text-gray-400">Price</th>
                <th className="py-3 text-right text-sm font-medium text-gray-500 dark:text-gray-400">Total</th>
              </tr>
            </thead>
            <tbody>
              {recentTrades.map((trade, index) => (
                <tr key={index} className="border-b border-gray-100 dark:border-gray-700">
                  <td className="py-4 text-sm text-gray-600 dark:text-gray-300">{trade.date}</td>
                  <td className="py-4">
                    <Link 
                      href={`/stock/${trade.symbol}`}
                      className="font-medium text-gray-900 hover:text-blue-600 dark:text-white dark:hover:text-blue-400"
                    >
                      {trade.symbol}
                    </Link>
                  </td>
                  <td className="py-4 text-right">
                    <span className={`font-medium ${
                      trade.action === 'BUY' ? 'text-green-600' : 'text-red-600'
                    }`}>
                      {trade.action}
                    </span>
                  </td>
                  <td className="py-4 text-right">{trade.shares}</td>
                  <td className="py-4 text-right">{formatCurrency(trade.price)}</td>
                  <td className="py-4 text-right">{formatCurrency(trade.total)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};

export default PortfolioDashboard; 