import React, { useState } from 'react';
import { Card } from './ui/card';
import { Button } from './ui/button';

const RAGDashboard = () => {
  const [query, setQuery] = useState('');
  const [loading, setLoading] = useState(false);
  const [response, setResponse] = useState<any>(null);
  const [history, setHistory] = useState<Array<{query: string; response: any}>>([]);

  const searchInvestmentInfo = async () => {
    setLoading(true);
    try {
      const response = await fetch('/api/rag', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ query }),
      });
      const data = await response.json();
      setResponse(data);
      setHistory(prev => [...prev, { query, response: data }]);
      setQuery('');
    } catch (error) {
      console.error('Error searching investment info:', error);
    }
    setLoading(false);
  };

  return (
    <Card className="p-6">
      <h2 className="text-2xl font-bold bg-gradient-to-r from-purple-600 to-blue-500 bg-clip-text text-transparent mb-6">
        AI Research Assistant
      </h2>
      <div className="space-y-4">
        <div>
          <label className="block text-sm font-medium mb-1">Ask anything about investments</label>
          <textarea
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="e.g., What are the key factors to consider when investing in tech stocks?"
            className="w-full p-2 border rounded-md h-24 resize-none"
          />
        </div>
        <Button 
          onClick={searchInvestmentInfo} 
          className="w-full"
          disabled={loading || !query.trim()}
        >
          {loading ? 'Searching...' : 'Search'}
        </Button>
        {response && (
          <div className="mt-4 p-4 bg-gray-50 dark:bg-gray-800 rounded-md">
            <h3 className="text-lg font-semibold mb-2">Answer</h3>
            <p className="text-sm whitespace-pre-wrap">{response.answer}</p>
            {response.sources && response.sources.length > 0 && (
              <div className="mt-4">
                <h4 className="text-sm font-semibold mb-2">Sources:</h4>
                <ul className="text-xs space-y-1">
                  {response.sources.map((source: string, index: number) => (
                    <li key={index} className="text-blue-500 hover:underline">
                      <a href={source} target="_blank" rel="noopener noreferrer">{source}</a>
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        )}
        {history.length > 0 && (
          <div className="mt-6">
            <h3 className="text-lg font-semibold mb-4">Search History</h3>
            <div className="space-y-4">
              {history.map((item, index) => (
                <div key={index} className="p-4 bg-gray-50 dark:bg-gray-800 rounded-md">
                  <p className="text-sm font-medium mb-2">Q: {item.query}</p>
                  <p className="text-sm">A: {item.response.answer}</p>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </Card>
  );
};

export default RAGDashboard; 