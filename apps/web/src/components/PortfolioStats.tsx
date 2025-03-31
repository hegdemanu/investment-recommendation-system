import React from 'react';

type Stat = {
  title: string;
  value: string;
  change?: string;
};

interface PortfolioStatsProps {
  stats: Stat[];
}

const PortfolioStats: React.FC<PortfolioStatsProps> = ({
  stats = [
    { title: 'Portfolio Value', value: '$125,430.00', change: '+2.3%' },
    { title: 'Total Return', value: '+$12,430.00', change: '+10.8%' },
    { title: 'Assets', value: '15', change: '' },
    { title: 'Risk Score', value: '68/100', change: 'Moderate' },
  ]
}) => {
  return (
    <div className="mb-8 grid grid-cols-1 gap-6 sm:grid-cols-2 lg:grid-cols-4">
      {stats.map((stat, i) => (
        <div
          key={i}
          className="rounded-lg border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-800 dark:bg-gray-900"
        >
          <p className="text-sm font-medium text-gray-500 dark:text-gray-400">{stat.title}</p>
          <p className="mt-2 text-3xl font-bold">{stat.value}</p>
          {stat.change && (
            <p className={`text-sm ${
              stat.change.includes('+') 
                ? 'text-green-600 dark:text-green-400' 
                : stat.change.includes('-') 
                  ? 'text-red-600 dark:text-red-400' 
                  : 'text-gray-600 dark:text-gray-400'
            }`}>
              {stat.change}
            </p>
          )}
        </div>
      ))}
    </div>
  );
};

export default PortfolioStats; 