import React from 'react';

type StockRecommendation = {
  asset: string;
  action: 'Buy' | 'Sell' | 'Hold';
  price: string;
  confidence: string;
};

interface StockRecommendationsProps {
  recommendations: StockRecommendation[];
  title?: string;
}

const getActionColorClass = (action: string) => {
  switch (action.toLowerCase()) {
    case 'buy':
      return 'text-green-600 dark:text-green-400 font-medium';
    case 'sell':
      return 'text-red-600 dark:text-red-400 font-medium';
    case 'hold':
      return 'text-amber-600 dark:text-amber-400 font-medium';
    default:
      return 'text-gray-600 dark:text-gray-400';
  }
};

const StockRecommendations: React.FC<StockRecommendationsProps> = ({
  recommendations = [
    { asset: 'AAPL', action: 'Buy', price: '$184.25', confidence: '92%' },
    { asset: 'MSFT', action: 'Hold', price: '$405.68', confidence: '87%' },
    { asset: 'AMZN', action: 'Buy', price: '$175.35', confidence: '89%' },
  ],
  title = 'Recent Recommendations'
}) => {
  return (
    <div>
      <h2 className="mb-4 text-xl font-semibold">{title}</h2>
      <div className="rounded-lg border border-gray-200 bg-white shadow-sm dark:border-gray-800 dark:bg-gray-900">
        <div className="overflow-x-auto">
          <table className="w-full text-left">
            <thead className="border-b border-gray-200 bg-gray-50 dark:border-gray-800 dark:bg-gray-950">
              <tr>
                <th className="px-6 py-3 text-sm font-medium text-gray-500 dark:text-gray-400">Asset</th>
                <th className="px-6 py-3 text-sm font-medium text-gray-500 dark:text-gray-400">Action</th>
                <th className="px-6 py-3 text-sm font-medium text-gray-500 dark:text-gray-400">Price</th>
                <th className="px-6 py-3 text-sm font-medium text-gray-500 dark:text-gray-400">Confidence</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-200 dark:divide-gray-800">
              {recommendations.map((rec, i) => (
                <tr key={i} className="hover:bg-gray-50 dark:hover:bg-gray-900">
                  <td className="px-6 py-4 font-medium">{rec.asset}</td>
                  <td className={`px-6 py-4 ${getActionColorClass(rec.action)}`}>{rec.action}</td>
                  <td className="px-6 py-4">{rec.price}</td>
                  <td className="px-6 py-4">{rec.confidence}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};

export default StockRecommendations; 