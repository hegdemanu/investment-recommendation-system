import React, { useEffect, useState } from 'react';
import axios from 'axios';

const HomePage = () => {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [popularStocks, setPopularStocks] = useState([]);
  const [mutualFunds, setMutualFunds] = useState([]);

  useEffect(() => {
    async function fetchData() {
      try {
        setLoading(true);
        
        // Fetch stocks from archive
        const stocksResponse = await axios.get('/api/archive/stocks');
        setPopularStocks(stocksResponse.data);
        
        // Fetch mutual funds from archive
        const mfResponse = await axios.get('/api/archive/mutual-funds');
        setMutualFunds(mfResponse.data);
        
        setError('');
      } catch (err) {
        setError('Failed to load data');
        console.error(err);
      } finally {
        setLoading(false);
      }
    }

    fetchData();
  }, []);

  return (
    <div>
      {/* Add this JSX to display mutual funds */}
      {mutualFunds.length > 0 && (
        <div className="mt-8">
          <h2 className="text-xl font-semibold mb-4">Featured Mutual Funds</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {mutualFunds.map((fund) => (
              <div key={fund.symbol} className="bg-white rounded-lg shadow-md p-4">
                <h3 className="font-bold text-lg">{fund.symbol}</h3>
                <p className="text-gray-700">{fund.name}</p>
                <div className="mt-2 flex justify-between">
                  <span className="font-semibold">${fund.nav.toFixed(2)}</span>
                  <span className={fund.change >= 0 ? 'text-green-600' : 'text-red-600'}>
                    {fund.change >= 0 ? '+' : ''}{fund.change.toFixed(2)}%
                  </span>
                </div>
                <p className="text-sm text-gray-500 mt-1">Expense Ratio: {fund.expense_ratio}%</p>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default HomePage;
