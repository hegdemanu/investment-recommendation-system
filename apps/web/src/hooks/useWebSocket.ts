import { useEffect, useRef, useState, useCallback } from 'react';

interface WebSocketMessage {
  type: 'marketData' | 'trade' | 'error';
  payload: any;
}

export const useWebSocket = (
  symbol: string,
  onMessage: (data: any) => void
) => {
  const ws = useRef<WebSocket | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const connect = useCallback(() => {
    try {
      ws.current = new WebSocket(`wss://api.example.com/ws/market-data/${symbol}`);

      ws.current.onopen = () => {
        setIsConnected(true);
        setError(null);
        
        // Subscribe to market data
        if (ws.current) {
          ws.current.send(JSON.stringify({
            type: 'subscribe',
            symbol,
            channels: ['marketData', 'trades', 'patterns']
          }));
        }
      };

      ws.current.onmessage = (event) => {
        try {
          const message: WebSocketMessage = JSON.parse(event.data);
          onMessage(message.payload);
        } catch (err) {
          console.error('Error parsing WebSocket message:', err);
        }
      };

      ws.current.onerror = (event) => {
        setError('WebSocket error occurred');
        console.error('WebSocket error:', event);
      };

      ws.current.onclose = () => {
        setIsConnected(false);
        // Attempt to reconnect after 5 seconds
        setTimeout(connect, 5000);
      };
    } catch (err) {
      setError('Failed to connect to WebSocket');
      console.error('WebSocket connection error:', err);
    }
  }, [symbol, onMessage]);

  useEffect(() => {
    connect();

    return () => {
      if (ws.current) {
        ws.current.close();
      }
    };
  }, [connect]);

  const sendMessage = useCallback((message: any) => {
    if (ws.current && isConnected) {
      ws.current.send(JSON.stringify(message));
    }
  }, [isConnected]);

  return {
    isConnected,
    error,
    sendMessage
  };
}; 