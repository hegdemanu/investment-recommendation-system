import { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import axios from 'axios';

export default function HomePage() {
  const [searchTerm, setSearchTerm] = useState('');
  const [stocks, setStocks] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [popularStocks, setPopularStocks] = useState([]);

  // Fetch popular stocks on component mount
  useEffect(() => {
    async function fetchPopularStocks() {
      try {
        setLoading(true);
        const response = await axios.get('/api/stocks/popular');
        setPopularStocks(response.data);
        setError('');
      } catch (err) {
        setError('Failed to load popular stocks');
        console.error(err);
      } finally {
        setLoading(false);
      }
    }

    fetchPopularStocks();
  }, []);

  // Handle search
  const handleSearch = async (e) => {
    e.preventDefault();
    
    if (!searchTerm.trim()) return;
    
    try {
      setLoading(true);
      const response = await axios.get(`/api/stocks/search?query=${searchTerm}`);
      setStocks(response.data);
      setError('');
    } catch (err) {
      setError('No stocks found matching your search');
      setStocks([]);
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-6xl mx-auto">
      <div className="text-center mb-12">
        <h1 className="text-3xl font-bold text-gray-800 mb-4">
          Investment Recommendation System
        </h1>
        <p className="text-gray-600 max-w-2xl mx-auto">
          Get AI-powered investment recommendations based on technical analysis, price forecasting, and sentiment analysis.
        </p>
      </div>

      {/* Search Form */}
      <div className="max-w-md mx-auto mb-10">
        <form onSubmit={handleSearch} className="flex items-center">
          <input
            type="text"
            className="flex-1 px-4 py-2 border border-gray-300 rounded-l-md focus:outline-none focus:ring-2 focus:ring-blue-500"
            placeholder="Search for stocks (e.g., AAPL, MSFT, GOOGL)"
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
          />
          <button
            type="submit"
            className="bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-r-md"
            disabled={loading}
          >
            {loading ? 'Searching...' : 'Search'}
          </button>
        </form>
      </div>

      {/* Search Results */}
      {error && <p className="text-red-500 text-center mb-6">{error}</p>}
      
      {stocks.length > 0 && (
        <div className="mb-10">
          <h2 className="text-xl font-semibold mb-4">Search Results</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {stocks.map((stock) => (
              <Link
                key={stock.symbol}
                to={`/stock/${stock.symbol}`}
                className="bg-white rounded-lg shadow-md p-4 hover:shadow-lg transition-shadow"
              >
                <h3 className="font-bold text-lg">{stock.symbol}</h3>
                <p className="text-gray-700">{stock.name}</p>
              </Link>
            ))}
          </div>
        </div>
      )}

      {/* Popular Stocks */}
      <div>
        <h2 className="text-xl font-semibold mb-4">Popular Stocks</h2>
        {loading && !stocks.length ? (
          <p className="text-center text-gray-500">Loading popular stocks...</p>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {popularStocks.map((stock) => (
              <Link
                key={stock.symbol}
                to={`/stock/${stock.symbol}`}
                className="bg-white rounded-lg shadow-md p-4 hover:shadow-lg transition-shadow"
              >
                <h3 className="font-bold text-lg">{stock.symbol}</h3>
                <p className="text-gray-700">{stock.name}</p>
                <div className="mt-2 flex justify-between">
                  <span className={`font-semibold ${stock.change >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                    ${stock.price.toFixed(2)}
                  </span>
                  <span className={`${stock.change >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                    {stock.change >= 0 ? '+' : ''}{stock.change.toFixed(2)}%
                  </span>
                </div>
              </Link>
            ))}
          </div>
        )}
      </div>
    </div>
  );
} 