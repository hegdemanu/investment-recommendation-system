'use client';

import React, { useState, useEffect, useRef } from 'react';
import { searchStocks } from '@/services/stockApi';
import { useRouter } from 'next/navigation';

interface StockSearchProps {
  onSelectStock?: (symbol: string) => void;
  placeholder?: string;
  className?: string;
}

const StockSearch: React.FC<StockSearchProps> = ({
  onSelectStock,
  placeholder = 'Search for a stock...',
  className = ''
}) => {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState<Array<{ symbol: string; name: string }>>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [showResults, setShowResults] = useState(false);
  const searchRef = useRef<HTMLDivElement>(null);
  const router = useRouter();

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (searchRef.current && !searchRef.current.contains(event.target as Node)) {
        setShowResults(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, []);

  useEffect(() => {
    const searchStocksDebounced = async () => {
      if (query.length < 2) {
        setResults([]);
        return;
      }

      setIsLoading(true);
      try {
        const data = await searchStocks(query);
        setResults(data);
      } catch (error) {
        console.error('Error searching stocks:', error);
      } finally {
        setIsLoading(false);
      }
    };

    const delayDebounce = setTimeout(() => {
      searchStocksDebounced();
    }, 300);

    return () => clearTimeout(delayDebounce);
  }, [query]);

  const handleSelectStock = (symbol: string) => {
    if (onSelectStock) {
      onSelectStock(symbol);
    } else {
      // Navigate to stock details page if no custom handler
      router.push(`/stock/${symbol}`);
    }
    setShowResults(false);
    setQuery('');
  };

  return (
    <div ref={searchRef} className={`relative w-full ${className}`}>
      <div className="relative">
        <input
          type="text"
          value={query}
          onChange={(e) => {
            setQuery(e.target.value);
            setShowResults(true);
          }}
          onFocus={() => setShowResults(true)}
          placeholder={placeholder}
          className="w-full p-3 pl-10 rounded-md border border-border bg-background focus:outline-none focus:ring-2 focus:ring-primary/50"
        />
        <div className="absolute left-3 top-1/2 transform -translate-y-1/2 text-muted-foreground">
          <svg
            xmlns="http://www.w3.org/2000/svg"
            width="18"
            height="18"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
          >
            <circle cx="11" cy="11" r="8"></circle>
            <line x1="21" y1="21" x2="16.65" y2="16.65"></line>
          </svg>
        </div>
        {isLoading && (
          <div className="absolute right-3 top-1/2 transform -translate-y-1/2">
            <div className="w-4 h-4 border-2 border-primary/60 border-t-transparent rounded-full animate-spin"></div>
          </div>
        )}
      </div>

      {showResults && (query.length > 1 || results.length > 0) && (
        <div className="absolute z-50 mt-1 w-full rounded-md bg-white dark:bg-gray-800 shadow-lg max-h-60 overflow-auto border border-border">
          {results.length === 0 ? (
            <div className="p-3 text-sm text-muted-foreground">
              {isLoading ? 'Searching...' : 'No results found'}
            </div>
          ) : (
            <ul>
              {results.map((stock) => (
                <li
                  key={stock.symbol}
                  className="cursor-pointer hover:bg-gray-100 dark:hover:bg-gray-700 p-3 border-b border-border/30 last:border-0"
                  onClick={() => handleSelectStock(stock.symbol)}
                >
                  <div className="font-medium">{stock.symbol}</div>
                  <div className="text-sm text-muted-foreground">{stock.name}</div>
                </li>
              ))}
            </ul>
          )}
        </div>
      )}
    </div>
  );
};

export default StockSearch; 