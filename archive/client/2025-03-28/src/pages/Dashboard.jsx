import { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import axios from 'axios';

export default function Dashboard() {
  const [marketOverview, setMarketOverview] = useState(null);
  const [topPerformers, setTopPerformers] = useState([]);
  const [recommendations, setRecommendations] = useState([]);
  const [sentimentTrends, setSentimentTrends] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  useEffect(() => {
    async function fetchDashboardData() {
      try {
        setLoading(true);
        
        // Fetch market overview data
        const marketResponse = await axios.get('/api/market/overview');
        setMarketOverview(marketResponse.data);
        
        // Fetch top performing stocks
        const performersResponse = await axios.get('/api/market/top-performers');
        setTopPerformers(performersResponse.data);
        
        // Fetch recommended stocks
        const recommendationsResponse = await axios.get('/api/recommendations');
        setRecommendations(recommendationsResponse.data);
        
        // Fetch sentiment trends
        const sentimentResponse = await axios.get('/api/market/sentiment-trends');
        setSentimentTrends(sentimentResponse.data);
        
        setError('');
      } catch (err) {
        setError('Failed to load dashboard data');
        console.error(err);
      } finally {
        setLoading(false);
      }
    }

    fetchDashboardData();
  }, []);

  if (loading) {
    return (
      <div className="flex justify-center items-center h-64">
        <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-blue-500"></div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="text-center py-10">
        <h2 className="text-2xl font-bold text-red-600 mb-4">Error</h2>
        <p className="text-gray-700">{error}</p>
      </div>
    );
  }

  return (
    <div className="max-w-6xl mx-auto">
      <h1 className="text-3xl font-bold text-gray-800 mb-6">Investment Dashboard</h1>
      
      {/* Market Overview */}
      {marketOverview && (
        <div className="bg-white rounded-lg shadow-md p-6 mb-6">
          <h2 className="text-xl font-semibold mb-4">Market Overview</h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="bg-gray-50 p-4 rounded-lg">
              <h3 className="text-md font-medium text-gray-700 mb-2">S&P 500</h3>
              <p className="text-2xl font-bold">{marketOverview.sp500.value.toFixed(2)}</p>
              <p className={`text-sm font-medium ${marketOverview.sp500.change >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                {marketOverview.sp500.change >= 0 ? '+' : ''}{marketOverview.sp500.change.toFixed(2)}%
              </p>
            </div>
            <div className="bg-gray-50 p-4 rounded-lg">
              <h3 className="text-md font-medium text-gray-700 mb-2">NASDAQ</h3>
              <p className="text-2xl font-bold">{marketOverview.nasdaq.value.toFixed(2)}</p>
              <p className={`text-sm font-medium ${marketOverview.nasdaq.change >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                {marketOverview.nasdaq.change >= 0 ? '+' : ''}{marketOverview.nasdaq.change.toFixed(2)}%
              </p>
            </div>
            <div className="bg-gray-50 p-4 rounded-lg">
              <h3 className="text-md font-medium text-gray-700 mb-2">Dow Jones</h3>
              <p className="text-2xl font-bold">{marketOverview.dowjones.value.toFixed(2)}</p>
              <p className={`text-sm font-medium ${marketOverview.dowjones.change >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                {marketOverview.dowjones.change >= 0 ? '+' : ''}{marketOverview.dowjones.change.toFixed(2)}%
              </p>
            </div>
          </div>
          
          <div className="mt-6">
            <h3 className="text-md font-medium text-gray-700 mb-3">Market Sentiment</h3>
            <div className="flex items-center">
              <div className="w-full bg-gray-200 rounded-full h-5">
                <div 
                  className={`h-5 rounded-full ${marketOverview.market_sentiment >= 0.6 ? 'bg-green-500' : marketOverview.market_sentiment >= 0.4 ? 'bg-yellow-500' : 'bg-red-500'}`}
                  style={{ width: `${marketOverview.market_sentiment * 100}%` }}
                ></div>
              </div>
              <span className="ml-4 font-semibold">{(marketOverview.market_sentiment * 100).toFixed(2)}%</span>
            </div>
            <p className="text-gray-600 mt-2">{marketOverview.sentiment_summary}</p>
          </div>
        </div>
      )}
      
      {/* Top Performers */}
      {topPerformers.length > 0 && (
        <div className="bg-white rounded-lg shadow-md p-6 mb-6">
          <h2 className="text-xl font-semibold mb-4">Top Performers Today</h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {topPerformers.map((stock) => (
              <Link
                key={stock.symbol}
                to={`/stock/${stock.symbol}`}
                className="bg-gray-50 p-4 rounded-lg hover:shadow-md transition-shadow"
              >
                <div className="flex justify-between items-start">
                  <div>
                    <h3 className="font-semibold text-gray-800">{stock.symbol}</h3>
                    <p className="text-gray-600 text-sm">{stock.name}</p>
                  </div>
                  <div className="text-right">
                    <p className="font-bold">${stock.price.toFixed(2)}</p>
                    <p className="text-green-600 font-semibold">+{stock.change.toFixed(2)}%</p>
                  </div>
                </div>
              </Link>
            ))}
          </div>
        </div>
      )}
      
      {/* Recommendations */}
      {recommendations.length > 0 && (
        <div className="bg-white rounded-lg shadow-md p-6 mb-6">
          <h2 className="text-xl font-semibold mb-4">Top Recommendations</h2>
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Stock</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Price</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Change</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Recommendation</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Confidence</th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {recommendations.map((stock) => (
                  <tr key={stock.symbol} className="hover:bg-gray-50">
                    <td className="px-6 py-4 whitespace-nowrap">
                      <Link to={`/stock/${stock.symbol}`} className="text-blue-600 hover:underline font-medium">
                        {stock.symbol}
                      </Link>
                      <p className="text-gray-500 text-sm">{stock.name}</p>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap font-medium">
                      ${stock.price.toFixed(2)}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <span className={`font-medium ${stock.change >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                        {stock.change >= 0 ? '+' : ''}{stock.change.toFixed(2)}%
                      </span>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <span className={`px-2 py-1 rounded-full text-xs font-bold ${
                        stock.recommendation === 'BUY' 
                          ? 'bg-green-100 text-green-800' 
                          : stock.recommendation === 'SELL' 
                            ? 'bg-red-100 text-red-800' 
                            : 'bg-yellow-100 text-yellow-800'
                      }`}>
                        {stock.recommendation}
                      </span>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      {stock.confidence.toFixed(2)}%
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
      
      {/* Sentiment Trends */}
      {sentimentTrends.length > 0 && (
        <div className="bg-white rounded-lg shadow-md p-6">
          <h2 className="text-xl font-semibold mb-4">Sector Sentiment Trends</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {sentimentTrends.map((sector) => (
              <div key={sector.name} className="bg-gray-50 p-4 rounded-lg">
                <h3 className="font-medium text-gray-800 mb-2">{sector.name}</h3>
                <div className="flex items-center mb-2">
                  <div className="w-full bg-gray-200 rounded-full h-4">
                    <div 
                      className={`h-4 rounded-full ${
                        sector.sentiment >= 0.6 ? 'bg-green-500' : sector.sentiment >= 0.4 ? 'bg-yellow-500' : 'bg-red-500'
                      }`}
                      style={{ width: `${sector.sentiment * 100}%` }}
                    ></div>
                  </div>
                  <span className="ml-3 font-semibold">{(sector.sentiment * 100).toFixed(2)}%</span>
                </div>
                <p className="text-gray-600 text-sm">{sector.trend === 'up' ? 'Improving' : sector.trend === 'down' ? 'Declining' : 'Stable'} sentiment</p>
                <p className="text-gray-600 text-sm mt-1">Top stock: <span className="font-medium">{sector.top_stock}</span></p>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
} 