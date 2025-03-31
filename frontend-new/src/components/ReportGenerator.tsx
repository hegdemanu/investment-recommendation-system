import React, { useState } from 'react';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';

const ReportGenerator = () => {
  const [symbol, setSymbol] = useState('');
  const [timeframe, setTimeframe] = useState('1y');
  const [loading, setLoading] = useState(false);
  const [report, setReport] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);

  const generateReport = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/api/report`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          symbol: symbol.toUpperCase(),
          timeframe: timeframe
        }),
      });

      if (!response.ok) {
        throw new Error('Failed to generate report');
      }

      const data = await response.json();
      setReport(data);
    } catch (error) {
      console.error('Error generating report:', error);
      setError('Failed to generate report. Please try again.');
    }
    setLoading(false);
  };

  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat('en-IN', {
      style: 'currency',
      currency: 'INR',
      maximumFractionDigits: 0
    }).format(value);
  };

  return (
    <Card className="p-6">
      <h2 className="text-2xl font-bold bg-gradient-to-r from-purple-600 to-blue-500 bg-clip-text text-transparent mb-6">
        Investment Report Generator
      </h2>
      <div className="space-y-4">
        <div>
          <label className="block text-sm font-medium mb-1">Stock Symbol</label>
          <input
            type="text"
            value={symbol}
            onChange={(e) => setSymbol(e.target.value.toUpperCase())}
            placeholder="e.g., RELIANCE, TCS, INFY"
            className="w-full p-2 border rounded-md"
          />
        </div>
        <div>
          <label className="block text-sm font-medium mb-1">Timeframe</label>
          <select
            value={timeframe}
            onChange={(e) => setTimeframe(e.target.value)}
            className="w-full p-2 border rounded-md"
          >
            <option value="1m">1 Month</option>
            <option value="3m">3 Months</option>
            <option value="6m">6 Months</option>
            <option value="1y">1 Year</option>
            <option value="5y">5 Years</option>
          </select>
        </div>
        <Button 
          onClick={generateReport} 
          className="w-full"
          disabled={loading || !symbol}
        >
          {loading ? 'Generating...' : 'Generate Report'}
        </Button>

        {error && (
          <div className="p-4 bg-red-50 dark:bg-red-900/20 text-red-600 dark:text-red-400 rounded-md">
            {error}
          </div>
        )}

        {report && (
          <div className="mt-4 space-y-4">
            <div className="p-4 bg-gray-50 dark:bg-gray-800 rounded-md">
              <h3 className="text-lg font-semibold mb-2">Executive Summary</h3>
              <p className="text-sm">{report.summary}</p>
              {report.currentPrice && (
                <p className="mt-2 text-lg font-semibold">
                  Current Price: {formatCurrency(report.currentPrice)}
                </p>
              )}
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="p-4 bg-gray-50 dark:bg-gray-800 rounded-md">
                <h3 className="text-lg font-semibold mb-2">Technical Analysis</h3>
                <ul className="space-y-2">
                  {report.technical?.map((item: string, index: number) => (
                    <li key={index} className="text-sm flex items-start">
                      <span className="mr-2">•</span>
                      <span>{item}</span>
                    </li>
                  ))}
                </ul>
              </div>
              <div className="p-4 bg-gray-50 dark:bg-gray-800 rounded-md">
                <h3 className="text-lg font-semibold mb-2">Fundamental Analysis</h3>
                <ul className="space-y-2">
                  {report.fundamental?.map((item: string, index: number) => (
                    <li key={index} className="text-sm flex items-start">
                      <span className="mr-2">•</span>
                      <span>{item}</span>
                    </li>
                  ))}
                </ul>
              </div>
            </div>
            <div className="p-4 bg-gray-50 dark:bg-gray-800 rounded-md">
              <h3 className="text-lg font-semibold mb-2">Recommendations</h3>
              <ul className="space-y-2">
                {report.recommendations?.map((item: string, index: number) => (
                  <li key={index} className="text-sm flex items-start">
                    <span className="mr-2">•</span>
                    <span>{item}</span>
                  </li>
                ))}
              </ul>
            </div>
            {report.metrics && (
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                {Object.entries(report.metrics).map(([key, value]: [string, any]) => (
                  <div key={key} className="p-4 bg-gray-50 dark:bg-gray-800 rounded-md">
                    <p className="text-sm text-gray-500 dark:text-gray-400">{key}</p>
                    <p className="text-lg font-semibold">
                      {typeof value === 'number' ? formatCurrency(value) : value}
                    </p>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    </Card>
  );
};

export default ReportGenerator; 