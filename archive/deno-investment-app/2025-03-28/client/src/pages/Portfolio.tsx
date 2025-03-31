import React, { useState, useEffect } from 'react';
import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip } from 'recharts';
import { Search, Filter, TrendingUp, TrendingDown, ArrowUpRight, ArrowDownRight, SlidersHorizontal } from 'lucide-react';
import * as Tabs from '@radix-ui/react-tabs';

// Mock portfolio data
const portfolioStocks = [
  {
    id: 1,
    symbol: 'AAPL',
    name: 'Apple Inc.',
    shares: 10,
    averageCost: 160.50,
    currentPrice: 178.85,
    value: 1788.50,
    gain: 183.50,
    gainPercentage: 11.43,
    sector: 'Technology',
  },
  {
    id: 2,
    symbol: 'MSFT',
    name: 'Microsoft Corporation',
    shares: 5,
    averageCost: 290.20,
    currentPrice: 315.75,
    value: 1578.75,
    gain: 127.75,
    gainPercentage: 8.80,
    sector: 'Technology',
  },
  {
    id: 3,
    symbol: 'AMZN',
    name: 'Amazon.com Inc.',
    shares: 8,
    averageCost: 130.40,
    currentPrice: 142.60,
    value: 1140.80,
    gain: 97.60,
    gainPercentage: 9.36,
    sector: 'Consumer Cyclical',
  },
  {
    id: 4,
    symbol: 'GOOGL',
    name: 'Alphabet Inc.',
    shares: 12,
    averageCost: 125.30,
    currentPrice: 132.85,
    value: 1594.20,
    gain: 90.60,
    gainPercentage: 6.03,
    sector: 'Technology',
  },
  {
    id: 5,
    symbol: 'TSLA',
    name: 'Tesla, Inc.',
    shares: 7,
    averageCost: 245.80,
    currentPrice: 228.50,
    value: 1599.50,
    gain: -121.10,
    gainPercentage: -7.04,
    sector: 'Consumer Cyclical',
  }
];

// Sector allocation data for pie chart
const sectorAllocation = [
  { name: 'Technology', value: 70 },
  { name: 'Consumer Cyclical', value: 25 },
  { name: 'Other', value: 5 },
];

const COLORS = ['#4f46e5', '#3b82f6', '#0ea5e9', '#06b6d4', '#14b8a6', '#10b981'];

const Portfolio = () => {
  const [stocks, setStocks] = useState<any[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [searchQuery, setSearchQuery] = useState('');
  const [sortBy, setSortBy] = useState('value');
  const [sortOrder, setSortOrder] = useState('desc');
  const [showFilters, setShowFilters] = useState(false);

  useEffect(() => {
    const timer = setTimeout(() => {
      setStocks(portfolioStocks);
      setIsLoading(false);
    }, 1000);

    return () => clearTimeout(timer);
  }, []);

  const handleSort = (column: string) => {
    if (sortBy === column) {
      setSortOrder(sortOrder === 'asc' ? 'desc' : 'asc');
    } else {
      setSortBy(column);
      setSortOrder('desc');
    }
  };

  const filteredStocks = stocks.filter(stock => 
    stock.symbol.toLowerCase().includes(searchQuery.toLowerCase()) || 
    stock.name.toLowerCase().includes(searchQuery.toLowerCase())
  );

  const sortedStocks = [...filteredStocks].sort((a, b) => {
    const compareValue = (a: any, b: any) => {
      if (typeof a[sortBy] === 'string') {
        return a[sortBy].localeCompare(b[sortBy]);
      }
      return a[sortBy] - b[sortBy];
    };

    return sortOrder === 'asc' ? compareValue(a, b) : compareValue(b, a);
  });

  // Calculate portfolio totals
  const portfolioValue = stocks.reduce((sum, stock) => sum + stock.value, 0);
  const totalGain = stocks.reduce((sum, stock) => sum + stock.gain, 0);
  const totalGainPercentage = totalGain / (portfolioValue - totalGain) * 100;

  return (
    <div>
      <h1 className="text-2xl font-semibold mb-6">Portfolio</h1>
      
      {isLoading ? (
        <div className="flex justify-center items-center h-64">
          <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-primary"></div>
        </div>
      ) : (
        <>
          {/* Portfolio Overview */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
            <div className="bg-card p-6 rounded-lg shadow-sm border border-border">
              <p className="text-muted-foreground">Portfolio Value</p>
              <p className="text-2xl font-semibold mt-1">${portfolioValue.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</p>
            </div>
            
            <div className="bg-card p-6 rounded-lg shadow-sm border border-border">
              <p className="text-muted-foreground">Total Gain/Loss</p>
              <div className="flex items-center mt-1">
                {totalGain > 0 ? (
                  <ArrowUpRight className="mr-2 h-5 w-5 text-green-500" />
                ) : (
                  <ArrowDownRight className="mr-2 h-5 w-5 text-red-500" />
                )}
                <p className={`text-2xl font-semibold ${totalGain > 0 ? 'text-green-500' : 'text-red-500'}`}>
                  ${Math.abs(totalGain).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                </p>
              </div>
            </div>
            
            <div className="bg-card p-6 rounded-lg shadow-sm border border-border">
              <p className="text-muted-foreground">Total Return</p>
              <div className="flex items-center mt-1">
                {totalGainPercentage > 0 ? (
                  <ArrowUpRight className="mr-2 h-5 w-5 text-green-500" />
                ) : (
                  <ArrowDownRight className="mr-2 h-5 w-5 text-red-500" />
                )}
                <p className={`text-2xl font-semibold ${totalGainPercentage > 0 ? 'text-green-500' : 'text-red-500'}`}>
                  {Math.abs(totalGainPercentage).toFixed(2)}%
                </p>
              </div>
            </div>
            
            <div className="bg-card p-6 rounded-lg shadow-sm border border-border">
              <p className="text-muted-foreground">Number of Stocks</p>
              <p className="text-2xl font-semibold mt-1">{stocks.length}</p>
            </div>
          </div>
          
          <Tabs.Root defaultValue="holdings" className="mb-8">
            <Tabs.List className="flex border-b border-border">
              <Tabs.Trigger 
                value="holdings" 
                className="px-4 py-2 font-medium data-[state=active]:text-primary data-[state=active]:border-b-2 data-[state=active]:border-primary"
              >
                Holdings
              </Tabs.Trigger>
              <Tabs.Trigger 
                value="allocation" 
                className="px-4 py-2 font-medium data-[state=active]:text-primary data-[state=active]:border-b-2 data-[state=active]:border-primary"
              >
                Allocation
              </Tabs.Trigger>
            </Tabs.List>
            
            <Tabs.Content value="holdings" className="pt-4">
              {/* Search and Filter */}
              <div className="flex flex-col sm:flex-row justify-between mb-6 gap-4">
                <div className="relative flex-1">
                  <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                  <input
                    type="text"
                    placeholder="Search investments..."
                    className="input pl-9"
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                  />
                </div>
                <button
                  onClick={() => setShowFilters(!showFilters)}
                  className="btn btn-ghost"
                >
                  <SlidersHorizontal className="h-4 w-4 mr-2" />
                  Filters
                </button>
              </div>
              
              {/* Stocks table */}
              <div className="overflow-x-auto">
                <table className="w-full divide-y divide-border">
                  <thead className="bg-muted">
                    <tr>
                      <th 
                        scope="col" 
                        className="px-6 py-3 text-left text-xs font-medium text-muted-foreground uppercase tracking-wider cursor-pointer"
                        onClick={() => handleSort('symbol')}
                      >
                        Symbol
                      </th>
                      <th 
                        scope="col" 
                        className="px-6 py-3 text-left text-xs font-medium text-muted-foreground uppercase tracking-wider cursor-pointer"
                        onClick={() => handleSort('shares')}
                      >
                        Shares
                      </th>
                      <th 
                        scope="col" 
                        className="px-6 py-3 text-left text-xs font-medium text-muted-foreground uppercase tracking-wider cursor-pointer"
                        onClick={() => handleSort('averageCost')}
                      >
                        Avg Cost
                      </th>
                      <th 
                        scope="col" 
                        className="px-6 py-3 text-left text-xs font-medium text-muted-foreground uppercase tracking-wider cursor-pointer"
                        onClick={() => handleSort('currentPrice')}
                      >
                        Current Price
                      </th>
                      <th 
                        scope="col" 
                        className="px-6 py-3 text-left text-xs font-medium text-muted-foreground uppercase tracking-wider cursor-pointer"
                        onClick={() => handleSort('value')}
                      >
                        Value
                      </th>
                      <th 
                        scope="col" 
                        className="px-6 py-3 text-left text-xs font-medium text-muted-foreground uppercase tracking-wider cursor-pointer"
                        onClick={() => handleSort('gainPercentage')}
                      >
                        Return
                      </th>
                    </tr>
                  </thead>
                  <tbody className="bg-card divide-y divide-border">
                    {sortedStocks.map((stock) => (
                      <tr key={stock.id}>
                        <td className="px-6 py-4 whitespace-nowrap">
                          <div className="flex flex-col">
                            <div className="text-sm font-medium">{stock.symbol}</div>
                            <div className="text-xs text-muted-foreground">{stock.name}</div>
                          </div>
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm">
                          {stock.shares}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm">
                          ${stock.averageCost.toFixed(2)}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm">
                          ${stock.currentPrice.toFixed(2)}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm">
                          ${stock.value.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap">
                          <div className="flex items-center">
                            {stock.gainPercentage > 0 ? (
                              <TrendingUp className="h-4 w-4 mr-1 text-green-500" />
                            ) : (
                              <TrendingDown className="h-4 w-4 mr-1 text-red-500" />
                            )}
                            <span className={`text-sm ${stock.gainPercentage > 0 ? 'text-green-500' : 'text-red-500'}`}>
                              {stock.gainPercentage > 0 ? '+' : ''}{stock.gainPercentage.toFixed(2)}%
                            </span>
                          </div>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </Tabs.Content>
            
            <Tabs.Content value="allocation" className="pt-4">
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <div className="bg-card p-6 rounded-lg shadow-sm border border-border">
                  <h2 className="text-lg font-medium mb-4">Sector Allocation</h2>
                  <div className="h-80">
                    <ResponsiveContainer width="100%" height="100%">
                      <PieChart>
                        <Pie
                          data={sectorAllocation}
                          cx="50%"
                          cy="50%"
                          labelLine={false}
                          outerRadius={100}
                          fill="#8884d8"
                          dataKey="value"
                          label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                        >
                          {sectorAllocation.map((entry, index) => (
                            <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                          ))}
                        </Pie>
                        <Tooltip />
                      </PieChart>
                    </ResponsiveContainer>
                  </div>
                </div>
                
                <div className="bg-card p-6 rounded-lg shadow-sm border border-border">
                  <h2 className="text-lg font-medium mb-4">Allocation Details</h2>
                  <div className="space-y-4">
                    {sectorAllocation.map((sector, index) => (
                      <div key={index} className="flex items-center justify-between">
                        <div className="flex items-center">
                          <div 
                            className="h-4 w-4 rounded-full mr-2" 
                            style={{ backgroundColor: COLORS[index % COLORS.length] }}
                          />
                          <span>{sector.name}</span>
                        </div>
                        <span className="font-medium">{sector.value}%</span>
                      </div>
                    ))}
                  </div>
                  
                  <div className="mt-8">
                    <h3 className="text-md font-medium mb-4">Risk Assessment</h3>
                    <div className="space-y-3">
                      <div>
                        <div className="flex justify-between text-sm mb-1">
                          <span>Diversification Score</span>
                          <span className="font-medium">7/10</span>
                        </div>
                        <div className="w-full bg-muted rounded-full h-2">
                          <div className="bg-primary h-2 rounded-full" style={{ width: '70%' }}></div>
                        </div>
                      </div>
                      
                      <div>
                        <div className="flex justify-between text-sm mb-1">
                          <span>Volatility</span>
                          <span className="font-medium">Medium</span>
                        </div>
                        <div className="w-full bg-muted rounded-full h-2">
                          <div className="bg-yellow-500 h-2 rounded-full" style={{ width: '50%' }}></div>
                        </div>
                      </div>
                      
                      <div>
                        <div className="flex justify-between text-sm mb-1">
                          <span>Growth Potential</span>
                          <span className="font-medium">High</span>
                        </div>
                        <div className="w-full bg-muted rounded-full h-2">
                          <div className="bg-green-500 h-2 rounded-full" style={{ width: '80%' }}></div>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </Tabs.Content>
          </Tabs.Root>
        </>
      )}
    </div>
  );
};

export default Portfolio; 