import React, { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Input } from './ui/input';
import { Button } from './ui/button';
import { Textarea } from './ui/textarea';

interface RAGResponse {
  answer: string;
  sources: Array<{
    title: string;
    content: string;
    url: string;
    relevance: number;
  }>;
}

const RAGDashboard: React.FC = () => {
  const [query, setQuery] = useState('');
  const [loading, setLoading] = useState(false);
  const [response, setResponse] = useState<RAGResponse | null>(null);

  const handleSubmit = async () => {
    if (!query.trim()) return;

    setLoading(true);
    try {
      const res = await fetch('/api/v1/rag/query', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ query }),
      });
      const data = await res.json();
      setResponse(data);
    } catch (error) {
      console.error('Error querying RAG system:', error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <Card className="w-full max-w-4xl mx-auto">
      <CardHeader>
        <CardTitle className="text-2xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
          Investment Research Assistant
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-6">
          <div>
            <label className="block text-sm font-medium mb-2">
              Ask anything about investments, markets, or specific stocks
            </label>
            <div className="flex gap-4">
              <Textarea
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="e.g., What are the key factors affecting Tesla's stock price this quarter?"
                className="flex-1 min-h-[100px]"
              />
              <Button
                onClick={handleSubmit}
                disabled={loading || !query.trim()}
                className="bg-gradient-to-r from-blue-600 to-purple-600 text-white h-fit"
              >
                {loading ? 'Analyzing...' : 'Ask'}
              </Button>
            </div>
          </div>

          {response && (
            <div className="space-y-6">
              <div>
                <h3 className="text-lg font-semibold mb-2">Answer</h3>
                <div className="bg-gray-50 rounded-lg p-4">
                  <p className="whitespace-pre-wrap">{response.answer}</p>
                </div>
              </div>

              <div>
                <h3 className="text-lg font-semibold mb-2">Sources</h3>
                <div className="space-y-4">
                  {response.sources.map((source, index) => (
                    <Card key={index} className="p-4">
                      <h4 className="font-medium mb-2">
                        <a
                          href={source.url}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="text-blue-600 hover:underline"
                        >
                          {source.title}
                        </a>
                      </h4>
                      <p className="text-sm text-gray-600 mb-2">{source.content}</p>
                      <div className="text-xs text-gray-500">
                        Relevance: {(source.relevance * 100).toFixed(1)}%
                      </div>
                    </Card>
                  ))}
                </div>
              </div>
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
};

export default RAGDashboard; 