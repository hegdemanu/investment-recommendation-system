'use client';

import { useState, useRef, useEffect } from 'react';
import { useRAG } from '../hooks/useRAG';

interface RAGChatProps {
  symbol?: string;
  initialContext?: string;
}

export function RAGChat({ symbol, initialContext }: RAGChatProps) {
  const [query, setQuery] = useState('');
  const [context, setContext] = useState(
    initialContext || (symbol ? `Information about ${symbol} stock` : '')
  );
  const endOfMessagesRef = useRef<HTMLDivElement>(null);

  const {
    queryRAG,
    latestResponse,
    queryHistory,
    isLoading,
    error,
    clearHistory,
  } = useRAG();

  // Scroll to bottom of chat when new messages arrive
  useEffect(() => {
    if (endOfMessagesRef.current) {
      endOfMessagesRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [queryHistory]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (query.trim() === '') return;

    queryRAG(query, context);
    setQuery('');
  };

  return (
    <div className="rounded-lg bg-white p-4 shadow-sm h-full flex flex-col">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold">AI Research Assistant</h3>
        <button
          onClick={clearHistory}
          className="text-xs text-gray-500 hover:text-gray-700"
        >
          Clear Chat
        </button>
      </div>

      {/* Chat Messages */}
      <div className="flex-1 overflow-y-auto mb-4 space-y-4">
        {/* Welcome message */}
        {queryHistory.length === 0 && (
          <div className="bg-blue-50 p-3 rounded-lg">
            <p className="text-sm">
              👋 Hi! I'm your AI research assistant. I can help you analyze {symbol || 'stocks'} and provide investment insights based on market data and news. Ask me anything!
            </p>
          </div>
        )}

        {/* Message history */}
        {queryHistory.map((item, index) => (
          <div key={index}>
            {/* User query */}
            <div className="flex justify-end mb-2">
              <div className="bg-blue-500 text-white p-3 rounded-lg max-w-[80%]">
                <p className="text-sm">{item.query}</p>
              </div>
            </div>

            {/* AI response */}
            <div className="flex mb-2">
              <div className="bg-gray-100 p-3 rounded-lg max-w-[80%]">
                <p className="text-sm whitespace-pre-line">{item.response.response}</p>
                
                {/* Sources if available */}
                {item.response.sources && item.response.sources.length > 0 && (
                  <div className="mt-2 pt-2 border-t border-gray-200">
                    <p className="text-xs font-medium text-gray-500">Sources:</p>
                    <ul className="list-disc list-inside text-xs text-gray-500">
                      {item.response.sources.slice(0, 3).map((source, idx) => (
                        <li key={idx} className="truncate">
                          {source.url ? (
                            <a 
                              href={source.url} 
                              target="_blank" 
                              rel="noopener noreferrer"
                              className="text-blue-600 hover:underline"
                            >
                              {source.title}
                            </a>
                          ) : (
                            source.title
                          )}
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
                
                {/* Related symbols if available */}
                {item.response.relatedSymbols && item.response.relatedSymbols.length > 0 && (
                  <div className="mt-2">
                    <p className="text-xs font-medium text-gray-500">Related symbols:</p>
                    <div className="flex flex-wrap gap-1 mt-1">
                      {item.response.relatedSymbols.map((sym, idx) => (
                        <span key={idx} className="text-xs bg-gray-200 px-2 py-1 rounded">
                          {sym}
                        </span>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>
        ))}

        {/* Loading indicator */}
        {isLoading && (
          <div className="flex mb-2">
            <div className="bg-gray-100 p-3 rounded-lg">
              <div className="flex space-x-2">
                <div className="w-2 h-2 rounded-full bg-gray-400 animate-bounce"></div>
                <div className="w-2 h-2 rounded-full bg-gray-400 animate-bounce delay-75"></div>
                <div className="w-2 h-2 rounded-full bg-gray-400 animate-bounce delay-150"></div>
              </div>
            </div>
          </div>
        )}

        {/* Error message */}
        {error && (
          <div className="flex mb-2">
            <div className="bg-red-100 text-red-800 p-3 rounded-lg max-w-[80%]">
              <p className="text-sm">Error: {error.message}</p>
            </div>
          </div>
        )}

        {/* Scroll anchor */}
        <div ref={endOfMessagesRef} />
      </div>

      {/* Input form */}
      <form onSubmit={handleSubmit} className="mt-auto">
        <div className="flex items-center border rounded-lg overflow-hidden">
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder={`Ask about ${symbol || 'market trends, stocks, or investment strategies'}...`}
            className="flex-1 p-3 focus:outline-none text-sm"
          />
          <button
            type="submit"
            disabled={isLoading}
            className={`p-3 ${
              isLoading
                ? 'bg-gray-300 text-gray-500'
                : 'bg-blue-500 text-white hover:bg-blue-600'
            }`}
          >
            <svg
              className="w-5 h-5"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
              xmlns="http://www.w3.org/2000/svg"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M14 5l7 7m0 0l-7 7m7-7H3"
              />
            </svg>
          </button>
        </div>
      </form>
    </div>
  );
} 