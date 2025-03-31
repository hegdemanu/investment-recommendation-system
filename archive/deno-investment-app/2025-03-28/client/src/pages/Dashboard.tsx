import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar } from 'recharts';
import { Briefcase, TrendingUp, DollarSign, PieChart } from 'lucide-react';

// Mock data for demonstration
const portfolioData = [
  { month: 'Jan', value: 10000 },
  { month: 'Feb', value: 12000 },
  { month: 'Mar', value: 11500 },
  { month: 'Apr', value: 13200 },
  { month: 'May', value: 14100 },
  { month: 'Jun', value: 13800 },
  { month: 'Jul', value: 15400 },
];

const sectorAllocation = [
  { name: 'Technology', value: 35 },
  { name: 'Healthcare', value: 20 },
  { name: 'Finance', value: 15 },
  { name: 'Consumer', value: 12 },
  { name: 'Energy', value: 10 },
  { name: 'Other', value: 8 },
];

const Dashboard = () => {
  const [isLoading, setIsLoading] = useState(true);
  const [portfolioValue, setPortfolioValue] = useState(0);
  const [portfolioGrowth, setPortfolioGrowth] = useState(0);
  const [recommendations, setRecommendations] = useState(0);

  useEffect(() => {
    // Simulate data loading
    const timer = setTimeout(() => {
      setPortfolioValue(15400);
      setPortfolioGrowth(8.5);
      setRecommendations(5);
      setIsLoading(false);
    }, 1000);

    return () => clearTimeout(timer);
  }, []);

  return (
    <div className="container mx-auto p-4">
      <h1 className="text-2xl font-bold mb-6">Investment Dashboard</h1>
      
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 mb-8">
        <div className="card">
          <div className="card-header">
            <h2 className="card-title">Portfolio Overview</h2>
          </div>
          <div className="card-body">
            <p className="mb-3">Total Value: $124,500.00</p>
            <p className="text-success mb-3">+$1,245.00 (1.01%)</p>
            <div className="h-40">
              {/* Chart component will go here */}
            </div>
            <button className="btn btn-primary w-full mt-4">
              View Full Portfolio
            </button>
          </div>
        </div>
        
        <div className="card">
          <div className="card-header">
            <h2 className="card-title">Recent Transactions</h2>
          </div>
          <div className="card-body">
            <div className="space-y-3">
              <div className="flex justify-between items-center">
                <div>
                  <p className="font-medium">AAPL</p>
                  <p className="text-sm text-muted-foreground">Purchase</p>
                </div>
                <div className="text-right">
                  <p className="font-medium">$145.86</p>
                  <p className="text-sm text-muted-foreground">2 hrs ago</p>
                </div>
              </div>
              <div className="flex justify-between items-center">
                <div>
                  <p className="font-medium">MSFT</p>
                  <p className="text-sm text-muted-foreground">Sale</p>
                </div>
                <div className="text-right">
                  <p className="font-medium">$349.12</p>
                  <p className="text-sm text-muted-foreground">Yesterday</p>
                </div>
              </div>
            </div>
            <button className="btn btn-outline w-full mt-4">
              View All Transactions
            </button>
          </div>
        </div>
        
        <div className="card">
          <div className="card-header">
            <h2 className="card-title">Top Recommendations</h2>
          </div>
          <div className="card-body">
            <div className="space-y-3">
              <div className="flex justify-between items-center">
                <div>
                  <p className="font-medium">NVDA</p>
                  <p className="text-sm text-success">Strong Buy</p>
                </div>
                <div className="text-right">
                  <p className="font-medium">$492.18</p>
                  <p className="text-sm text-success">+2.4%</p>
                </div>
              </div>
              <div className="flex justify-between items-center">
                <div>
                  <p className="font-medium">AMD</p>
                  <p className="text-sm text-success">Buy</p>
                </div>
                <div className="text-right">
                  <p className="font-medium">$159.88</p>
                  <p className="text-sm text-success">+1.2%</p>
                </div>
              </div>
            </div>
            <button className="btn btn-outline w-full mt-4">
              View All Recommendations
            </button>
          </div>
        </div>
      </div>
      
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="card">
          <div className="card-header">
            <h2 className="card-title">Market Overview</h2>
          </div>
          <div className="card-body">
            <div className="space-y-3">
              <div className="flex justify-between">
                <span>S&P 500</span>
                <span className="text-success">+0.8%</span>
              </div>
              <div className="flex justify-between">
                <span>Nasdaq</span>
                <span className="text-success">+1.2%</span>
              </div>
              <div className="flex justify-between">
                <span>Dow Jones</span>
                <span className="text-danger">-0.3%</span>
              </div>
            </div>
            <div className="h-48 mt-4">
              {/* Chart component will go here */}
            </div>
          </div>
        </div>
        
        <div className="card">
          <div className="card-header">
            <h2 className="card-title">News Feed</h2>
          </div>
          <div className="card-body">
            <div className="space-y-4">
              <div>
                <h3 className="font-medium">Fed Signals Rate Cut in September</h3>
                <p className="text-sm text-muted-foreground">The Federal Reserve indicated plans to cut interest rates by 25 basis points...</p>
                <p className="text-xs text-muted-foreground mt-1">2 hours ago</p>
              </div>
              <div>
                <h3 className="font-medium">Tech Stocks Rally on Strong Earnings</h3>
                <p className="text-sm text-muted-foreground">Major tech companies reported better-than-expected quarterly results...</p>
                <p className="text-xs text-muted-foreground mt-1">4 hours ago</p>
              </div>
            </div>
            <button className="btn btn-outline w-full mt-4">
              More News
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard; 