import { useState } from 'react';
import { useMutation } from 'react-query';
import {
  mlService,
  type RAGResponse,
} from '@repo/api';

export function useRAG() {
  const [queryHistory, setQueryHistory] = useState<{ query: string; response: RAGResponse }[]>([]);
  
  // Query the RAG system
  const {
    mutate: queryRAG,
    data: latestResponse,
    isLoading,
    error,
    reset,
  } = useMutation(
    ({ query, context }: { query: string; context?: string }) =>
      mlService.queryRAG(query, context),
    {
      onSuccess: (data, variables) => {
        // Add to query history
        setQueryHistory(prev => [...prev, { query: variables.query, response: data }]);
      },
    }
  );
  
  // Clear query history
  const clearHistory = () => {
    setQueryHistory([]);
    reset();
  };
  
  // Get the last N queries
  const getRecentQueries = (n: number = 5) => {
    return queryHistory.slice(-n);
  };
  
  // Execute a RAG query with optional context
  const executeQuery = (query: string, context?: string) => {
    queryRAG({ query, context });
  };
  
  return {
    queryRAG: executeQuery,
    latestResponse,
    queryHistory,
    isLoading,
    error,
    clearHistory,
    getRecentQueries,
  };
} 