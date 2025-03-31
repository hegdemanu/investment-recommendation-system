import { useState, useEffect, useRef } from 'react';

interface UseWebSocketOptions {
  url: string;
  onMessage?: (data: any) => void;
  onOpen?: () => void;
  onClose?: () => void;
  onError?: (error: Event) => void;
  reconnect?: boolean;
  reconnectInterval?: number;
  reconnectAttempts?: number;
}

export const useWebSocket = ({
  url,
  onMessage,
  onOpen,
  onClose,
  onError,
  reconnect = true,
  reconnectInterval = 3000,
  reconnectAttempts = 5
}: UseWebSocketOptions) => {
  const [isConnected, setIsConnected] = useState(false);
  const [lastMessage, setLastMessage] = useState<any>(null);
  const [error, setError] = useState<Event | null>(null);
  const socket = useRef<WebSocket | null>(null);
  const reconnectCount = useRef(0);

  const connect = () => {
    try {
      const ws = new WebSocket(url);
      
      ws.onopen = () => {
        setIsConnected(true);
        setError(null);
        reconnectCount.current = 0;
        if (onOpen) onOpen();
      };
      
      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          setLastMessage(data);
          if (onMessage) onMessage(data);
        } catch (err) {
          setLastMessage(event.data);
          if (onMessage) onMessage(event.data);
        }
      };
      
      ws.onclose = () => {
        setIsConnected(false);
        if (onClose) onClose();
        
        if (reconnect && reconnectCount.current < reconnectAttempts) {
          reconnectCount.current += 1;
          setTimeout(() => connect(), reconnectInterval);
        }
      };
      
      ws.onerror = (event) => {
        setError(event);
        if (onError) onError(event);
      };
      
      socket.current = ws;
    } catch (err) {
      console.error('WebSocket connection error:', err);
    }
  };

  const disconnect = () => {
    if (socket.current) {
      socket.current.close();
    }
  };

  const sendMessage = (data: any) => {
    if (socket.current && isConnected) {
      const message = typeof data === 'string' ? data : JSON.stringify(data);
      socket.current.send(message);
    } else {
      console.error('WebSocket is not connected');
    }
  };

  useEffect(() => {
    connect();
    
    return () => {
      disconnect();
    };
  }, [url]); // Reconnect if URL changes

  return {
    isConnected,
    lastMessage,
    error,
    sendMessage,
    disconnect
  };
};

export default useWebSocket; 