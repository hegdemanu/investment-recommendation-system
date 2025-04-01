'use client';

import React, { useState, useRef, useEffect } from 'react';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Tooltip } from '@/components/ui/tooltip';
import { InfoIcon, Search, Clock, RefreshCw, Bookmark, ExternalLink, ThumbsUp, ThumbsDown } from 'lucide-react';

const RAGDashboard = () => {
  const [query, setQuery] = useState('');
  const [loading, setLoading] = useState(false);
  const [response, setResponse] = useState<any>(null);
  const [history, setHistory] = useState<Array<{query: string; response: any; saved?: boolean; upvoted?: boolean; downvoted?: boolean}>>([]);
  const [suggestions, setSuggestions] = useState<string[]>([
    "What sectors perform best during high inflation?",
    "Compare index funds vs. ETFs for long-term investing",
    "How do rising interest rates affect tech stocks?",
    "What are Warren Buffett's investment principles?"
  ]);
  const responseRef = useRef<HTMLDivElement>(null);

  const searchInvestmentInfo = async () => {
    if (!query.trim()) return;
    
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
    } catch (error) {
      console.error('Error searching investment info:', error);
    }
    setLoading(false);
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      searchInvestmentInfo();
    }
  };

  const useSuggestion = (suggestion: string) => {
    setQuery(suggestion);
  };

  const saveToFavorites = (index: number) => {
    setHistory(prev => 
      prev.map((item, i) => 
        i === index ? { ...item, saved: !item.saved } : item
      )
    );
  };

  const rateResponse = (index: number, rating: 'up' | 'down') => {
    setHistory(prev => 
      prev.map((item, i) => {
        if (i === index) {
          return { 
            ...item, 
            upvoted: rating === 'up' ? !item.upvoted : false,
            downvoted: rating === 'down' ? !item.downvoted : false
          };
        }
        return item;
      })
    );
  };

  // Scroll to response when it's loaded
  useEffect(() => {
    if (response && responseRef.current) {
      responseRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [response]);

  return (
    <Card className="p-6 border border-gray-200 dark:border-gray-800 shadow-md rounded-xl">
      <h2 className="text-2xl font-bold text-primary-700 dark:text-primary-400 mb-6 flex items-center">
        AI Research Assistant
        <Tooltip content="Powered by RAG (Retrieval-Augmented Generation) with real-time financial data">
          <InfoIcon className="ml-2 h-4 w-4 text-gray-400 cursor-help" />
        </Tooltip>
      </h2>
      
      <div className="space-y-6">
        {/* Query Input */}
        <div className="space-y-3">
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">
            Ask anything about investments or market trends
          </label>
          <div className="relative">
            <textarea
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="e.g., What are the key factors to consider when investing in tech stocks?"
              className="w-full p-3 pr-12 border border-gray-300 dark:border-gray-600 rounded-md h-24 resize-none bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100 focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent"
            />
            <Button 
              onClick={searchInvestmentInfo}
              className="absolute right-2 bottom-2 p-2 h-8 w-8"
              disabled={loading || !query.trim()}
              variant="ghost"
            >
              <Search className="h-4 w-4" />
            </Button>
          </div>
          
          {/* Search Button */}
          <Button 
            onClick={searchInvestmentInfo} 
            className="w-full bg-primary-600 hover:bg-primary-700 text-white"
            disabled={loading || !query.trim()}
          >
            {loading ? (
              <span className="flex items-center">
                <RefreshCw className="h-4 w-4 mr-2 animate-spin" />
                Researching...
              </span>
            ) : (
              <span className="flex items-center">
                <Search className="h-4 w-4 mr-2" />
                Research
              </span>
            )}
          </Button>
          
          {/* Suggestions */}
          {!response && !loading && (
            <div className="mt-4">
              <h3 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">Suggestions:</h3>
              <div className="flex flex-wrap gap-2">
                {suggestions.map((suggestion, index) => (
                  <button
                    key={index}
                    onClick={() => useSuggestion(suggestion)}
                    className="text-xs px-3 py-1.5 bg-gray-100 dark:bg-gray-800 text-gray-700 dark:text-gray-300 rounded-full hover:bg-gray-200 dark:hover:bg-gray-700 transition-colors"
                  >
                    {suggestion}
                  </button>
                ))}
              </div>
            </div>
          )}
        </div>
        
        {/* Response */}
        {response && (
          <div ref={responseRef} className="mt-6 p-5 bg-gray-50 dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 transition-all duration-300 animate-fade-in">
            <h3 className="text-lg font-semibold mb-3 text-gray-800 dark:text-gray-100">Answer</h3>
            <div className="prose prose-sm dark:prose-invert max-w-none">
              <p className="text-gray-700 dark:text-gray-300 whitespace-pre-wrap leading-relaxed">
                {response.answer}
              </p>
            </div>
            
            {/* Feedback Controls */}
            <div className="flex items-center justify-between mt-4 pt-3 border-t border-gray-200 dark:border-gray-700">
              <div className="flex items-center space-x-2">
                <Button variant="ghost" size="sm" className="h-8 px-2 text-gray-500">
                  <ThumbsUp className="h-4 w-4 mr-1" />
                  <span className="text-xs">Helpful</span>
                </Button>
                <Button variant="ghost" size="sm" className="h-8 px-2 text-gray-500">
                  <ThumbsDown className="h-4 w-4 mr-1" />
                  <span className="text-xs">Not helpful</span>
                </Button>
                <Button variant="ghost" size="sm" className="h-8 px-2 text-gray-500">
                  <Bookmark className="h-4 w-4 mr-1" />
                  <span className="text-xs">Save</span>
                </Button>
              </div>
              <Button variant="ghost" size="sm" className="h-8 text-primary-600 hover:text-primary-700" onClick={() => setQuery('')}>
                <span className="text-xs">New Question</span>
              </Button>
            </div>
            
            {/* Sources */}
            {response.sources && response.sources.length > 0 && (
              <div className="mt-4 pt-3 border-t border-gray-200 dark:border-gray-700">
                <h4 className="text-sm font-semibold mb-2 flex items-center text-gray-700 dark:text-gray-300">
                  Sources
                  <Tooltip content="These sources were used to provide the most up-to-date information">
                    <InfoIcon className="ml-1 h-3 w-3 text-gray-400 cursor-help" />
                  </Tooltip>
                </h4>
                <ul className="space-y-1.5">
                  {response.sources.map((source: string, index: number) => (
                    <li key={index} className="flex items-center text-primary-600 dark:text-primary-400 text-xs">
                      <ExternalLink className="h-3 w-3 mr-1 flex-shrink-0" />
                      <a 
                        href={source} 
                        target="_blank" 
                        rel="noopener noreferrer"
                        className="hover:underline truncate"
                      >
                        {source.length > 50 ? `${source.substring(0, 50)}...` : source}
                      </a>
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        )}
        
        {/* History */}
        {history.length > 0 && (
          <div className="mt-8 space-y-4">
            <div className="flex items-center justify-between">
              <h3 className="text-lg font-semibold text-gray-800 dark:text-gray-100 flex items-center">
                <Clock className="h-4 w-4 mr-2 text-gray-400" />
                Recent Queries
              </h3>
              <Button
                variant="ghost"
                size="sm"
                className="text-xs text-gray-500"
                onClick={() => setHistory([])}
              >
                Clear History
              </Button>
            </div>
            
            <div className="space-y-3">
              {history.slice(0, 5).map((item, index) => (
                <div 
                  key={index} 
                  className="p-4 bg-gray-50 dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 hover:border-primary-300 dark:hover:border-primary-700 transition-colors"
                >
                  <p className="text-sm font-medium mb-2 text-gray-800 dark:text-gray-200">Q: {item.query}</p>
                  <p className="text-xs text-gray-600 dark:text-gray-400 line-clamp-2">
                    A: {item.response.answer}
                  </p>
                  
                  <div className="flex items-center justify-end mt-2 gap-2">
                    <Button 
                      variant="ghost" 
                      size="sm" 
                      className="h-6 px-2 text-xs"
                      onClick={() => setQuery(item.query)}
                    >
                      Reuse
                    </Button>
                    <Button 
                      variant="ghost" 
                      size="sm" 
                      className={`h-6 px-2 text-xs ${item.saved ? 'text-primary-600' : 'text-gray-500'}`}
                      onClick={() => saveToFavorites(index)}
                    >
                      <Bookmark className="h-3 w-3 mr-1" />
                      {item.saved ? 'Saved' : 'Save'}
                    </Button>
                  </div>
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