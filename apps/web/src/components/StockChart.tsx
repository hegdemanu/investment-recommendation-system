'use client';

import React, { useEffect, useRef, useState } from 'react';
import { Chart, registerables } from 'chart.js';
import 'chartjs-adapter-date-fns';
import zoomPlugin from 'chartjs-plugin-zoom';
import { useTheme } from 'next-themes';
import axios from 'axios';

// Register Chart.js components
Chart.register(...registerables, zoomPlugin);

interface StockData {
  date: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  ticker: string;
}

interface StockPrediction {
  date: string;
  prediction: number;
  lower_bound?: number;
  upper_bound?: number;
  model: string;
}

interface StockChartProps {
  ticker: string;
  timeRange?: '1d' | '5d' | '1m' | '3m' | '6m' | '1y' | '5y';
  chartType?: 'line' | 'candlestick';
  showVolume?: boolean;
  showPredictions?: boolean;
  height?: number;
  width?: number;
  useWebSocket?: boolean;
}

const StockChart: React.FC<StockChartProps> = ({
  ticker,
  timeRange = '3m',
  chartType = 'line',
  showVolume = true,
  showPredictions = true,
  height = 400,
  width = 800,
  useWebSocket = true,
}) => {
  const chartRef = useRef<HTMLCanvasElement | null>(null);
  const chartInstance = useRef<Chart | null>(null);
  const { theme } = useTheme();
  const [stockData, setStockData] = useState<StockData[]>([]);
  const [predictions, setPredictions] = useState<StockPrediction[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const wsRef = useRef<WebSocket | null>(null);

  // Colors based on theme
  const getChartColors = () => {
    return {
      lineColor: theme === 'dark' ? 'rgba(75, 192, 192, 1)' : 'rgba(75, 192, 192, 1)',
      fillColor: theme === 'dark' ? 'rgba(75, 192, 192, 0.2)' : 'rgba(75, 192, 192, 0.2)',
      gridColor: theme === 'dark' ? 'rgba(255, 255, 255, 0.1)' : 'rgba(0, 0, 0, 0.1)',
      textColor: theme === 'dark' ? 'rgba(255, 255, 255, 0.7)' : 'rgba(0, 0, 0, 0.7)',
      volumeColor: theme === 'dark' ? 'rgba(128, 128, 128, 0.5)' : 'rgba(128, 128, 128, 0.5)',
      predictionColor: theme === 'dark' ? 'rgba(255, 159, 64, 1)' : 'rgba(255, 159, 64, 1)',
      predictionFillColor: theme === 'dark' ? 'rgba(255, 159, 64, 0.2)' : 'rgba(255, 159, 64, 0.2)',
      upColor: 'rgba(75, 192, 192, 1)',
      downColor: 'rgba(255, 99, 132, 1)',
    };
  };

  // Fetch stock data from API
  const fetchStockData = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await axios.get(`/api/v1/stocks/${ticker}/prices`, {
        params: { time_range: timeRange },
      });
      setStockData(response.data);
      
      if (showPredictions) {
        const predictionResponse = await axios.get(`/api/v1/stocks/${ticker}/predictions`, {
          params: { time_range: timeRange },
        });
        setPredictions(predictionResponse.data);
      }
      
      setLoading(false);
    } catch (err) {
      console.error('Error fetching stock data:', err);
      setError('Failed to load stock data');
      setLoading(false);
    }
  };

  // Initialize WebSocket connection
  const initWebSocket = () => {
    if (!useWebSocket) return;

    const clientId = `client_${Math.random().toString(36).substring(2, 9)}`;
    const ws = new WebSocket(`${window.location.protocol === 'https:' ? 'wss:' : 'ws:'}//${window.location.host}/ws/ticker/${ticker}/${clientId}`);
    
    ws.onopen = () => {
      console.log('WebSocket connected');
    };
    
    ws.onmessage = (event) => {
      try {
        const message = JSON.parse(event.data);
        
        if (message.action === 'update' && message.ticker === ticker) {
          // Update stock data with real-time values
          setStockData(prevData => {
            const newData = [...prevData];
            const latestData = message.data;
            
            // Check if we need to update the last entry or add a new one
            const lastEntry = newData[newData.length - 1];
            if (lastEntry && lastEntry.date === latestData.date) {
              // Update existing entry
              newData[newData.length - 1] = latestData;
            } else {
              // Add new entry
              newData.push(latestData);
              
              // Keep only the data points within our time range
              // This is a simple approach - might need adjustment based on actual data structure
              if (newData.length > 100) {
                newData.shift();  // Remove oldest data point
              }
            }
            
            return newData;
          });
          
          // Update chart
          if (chartInstance.current) {
            chartInstance.current.update();
          }
        }
      } catch (e) {
        console.error('Error processing WebSocket message', e);
      }
    };
    
    ws.onclose = () => {
      console.log('WebSocket disconnected');
      
      // Try to reconnect after a delay
      setTimeout(() => {
        if (wsRef.current === ws) {  // Only reconnect if this is still the current WS
          initWebSocket();
        }
      }, 3000);
    };
    
    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
      ws.close();
    };
    
    wsRef.current = ws;
    
    return () => {
      ws.close();
      wsRef.current = null;
    };
  };

  // Initialize chart
  const initChart = () => {
    if (!chartRef.current) return;
    
    // Clear any existing chart
    if (chartInstance.current) {
      chartInstance.current.destroy();
    }
    
    const ctx = chartRef.current.getContext('2d');
    if (!ctx) return;
    
    const colors = getChartColors();
    
    // Prepare data
    const dates = stockData.map(d => new Date(d.date));
    const prices = stockData.map(d => d.close);
    const volumes = stockData.map(d => d.volume);
    
    // Prepare prediction data if available
    const predictionDates = predictions.map(p => new Date(p.date));
    const predictionValues = predictions.map(p => p.prediction);
    const lowerBounds = predictions.map(p => p.lower_bound || null);
    const upperBounds = predictions.map(p => p.upper_bound || null);
    
    // Create datasets based on chart type
    const datasets = [];
    
    if (chartType === 'line') {
      // Price line chart
      datasets.push({
        label: `${ticker} Price`,
        data: prices,
        borderColor: colors.lineColor,
        backgroundColor: colors.fillColor,
        pointRadius: 0,
        borderWidth: 2,
        fill: true,
        tension: 0.2,
        yAxisID: 'y',
      });
    } else if (chartType === 'candlestick') {
      // Candlestick implementation - simplified for this example
      // Note: For a proper candlestick, you should use a specific chart.js plugin
      // like chartjs-chart-financial
      datasets.push({
        label: `${ticker} OHLC`,
        data: stockData.map((d, i) => ({
          x: new Date(d.date),
          o: d.open,
          h: d.high,
          l: d.low,
          c: d.close,
          color: d.close >= d.open ? colors.upColor : colors.downColor
        })),
        borderColor: stockData.map(d => d.close >= d.open ? colors.upColor : colors.downColor),
        backgroundColor: stockData.map(d => d.close >= d.open ? 
          'rgba(75, 192, 192, 0.3)' : 'rgba(255, 99, 132, 0.3)'),
        yAxisID: 'y',
        type: 'bar',  // Using bar as a simplified alternative to candlestick
        barPercentage: 0.8,
      });
    }
    
    // Add volume dataset if enabled
    if (showVolume) {
      datasets.push({
        label: 'Volume',
        data: volumes,
        backgroundColor: colors.volumeColor,
        borderWidth: 0,
        yAxisID: 'volume',
        type: 'bar',
        order: 2,
      });
    }
    
    // Add prediction datasets if enabled and available
    if (showPredictions && predictions.length > 0) {
      datasets.push({
        label: 'Prediction',
        data: predictionValues,
        borderColor: colors.predictionColor,
        backgroundColor: 'transparent',
        borderDash: [5, 5],
        borderWidth: 2,
        pointRadius: 0,
        tension: 0,
        yAxisID: 'y',
        order: 0,
      });
      
      // Add prediction bounds if available
      if (lowerBounds.some(v => v !== null) && upperBounds.some(v => v !== null)) {
        datasets.push({
          label: 'Prediction Range',
          data: predictionValues,
          borderColor: 'transparent',
          backgroundColor: colors.predictionFillColor,
          fill: {
            target: 'origin',
            above: colors.predictionFillColor,
            below: colors.predictionFillColor
          },
          pointRadius: 0,
          yAxisID: 'y',
          order: 1,
        });
      }
    }
    
    // Create chart
    chartInstance.current = new Chart(ctx, {
      type: 'line',
      data: {
        labels: dates,
        datasets: datasets,
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: {
          mode: 'index',
          intersect: false,
        },
        plugins: {
          legend: {
            position: 'top',
            labels: {
              color: colors.textColor,
            },
          },
          tooltip: {
            mode: 'index',
            intersect: false,
            callbacks: {
              label: function(context) {
                const label = context.dataset.label || '';
                const value = context.parsed.y;
                return `${label}: ${value.toLocaleString()}`;
              }
            }
          },
          zoom: {
            pan: {
              enabled: true,
              mode: 'x',
            },
            zoom: {
              wheel: {
                enabled: true,
              },
              pinch: {
                enabled: true,
              },
              mode: 'x',
            },
          },
        },
        scales: {
          x: {
            type: 'time',
            time: {
              unit: timeRange === '1d' || timeRange === '5d' ? 'hour' : 'day',
              displayFormats: {
                hour: 'HH:mm',
                day: 'MMM d',
              },
            },
            grid: {
              color: colors.gridColor,
            },
            ticks: {
              color: colors.textColor,
            },
          },
          y: {
            position: 'right',
            grid: {
              color: colors.gridColor,
            },
            ticks: {
              color: colors.textColor,
              callback: function(value) {
                return value.toLocaleString();
              },
            },
          },
          volume: {
            position: 'left',
            grid: {
              drawOnChartArea: false,
            },
            ticks: {
              color: colors.textColor,
              callback: function(value) {
                return value >= 1000000 ? (value / 1000000).toFixed(1) + 'M' : 
                       value >= 1000 ? (value / 1000).toFixed(1) + 'K' : value;
              },
            },
            display: showVolume,
          },
        },
      },
    });
  };

  // Effect to fetch data when ticker or timeRange changes
  useEffect(() => {
    fetchStockData();
    
    // Clean up WebSocket on component unmount or ticker change
    return () => {
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }
    };
  }, [ticker, timeRange, showPredictions]);

  // Effect to initialize or update chart when data or theme changes
  useEffect(() => {
    if (stockData.length > 0) {
      initChart();
    }
  }, [stockData, predictions, theme, chartType, showVolume, showPredictions]);

  // Effect to initialize WebSocket after data is loaded
  useEffect(() => {
    if (stockData.length > 0 && useWebSocket) {
      initWebSocket();
    }
  }, [stockData]);

  // Helper to format large numbers
  const formatNumber = (num: number): string => {
    return new Intl.NumberFormat('en-US', {
      maximumFractionDigits: 2,
    }).format(num);
  };

  return (
    <div className="relative bg-white dark:bg-gray-800 p-4 rounded-lg shadow-md">
      <div className="flex justify-between items-center mb-4">
        <div>
          <h2 className="text-xl font-bold text-gray-800 dark:text-white">{ticker}</h2>
          {!loading && stockData.length > 0 && (
            <p className="text-2xl font-semibold text-gray-900 dark:text-gray-100">
              ${formatNumber(stockData[stockData.length - 1].close)}
              <span className={`ml-2 text-sm ${
                stockData[stockData.length - 1].close > stockData[stockData.length - 2]?.close 
                  ? 'text-green-500' 
                  : 'text-red-500'
              }`}>
                {stockData[stockData.length - 1].close > stockData[stockData.length - 2]?.close ? '▲' : '▼'}
                {formatNumber(Math.abs(stockData[stockData.length - 1].close - stockData[stockData.length - 2]?.close))}
                ({formatNumber(Math.abs((stockData[stockData.length - 1].close - stockData[stockData.length - 2]?.close) / stockData[stockData.length - 2]?.close * 100))}%)
              </span>
            </p>
          )}
        </div>
        <div className="flex space-x-2">
          {['1d', '5d', '1m', '3m', '6m', '1y', '5y'].map((range) => (
            <button
              key={range}
              className={`px-2 py-1 text-xs rounded ${
                timeRange === range 
                  ? 'bg-blue-500 text-white' 
                  : 'bg-gray-200 text-gray-700 dark:bg-gray-700 dark:text-gray-300'
              }`}
              onClick={() => setTimeRange(range as StockChartProps['timeRange'])}
            >
              {range}
            </button>
          ))}
        </div>
      </div>
      
      <div style={{ height, width: '100%' }}>
        {loading ? (
          <div className="flex items-center justify-center h-full">
            <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-blue-500"></div>
          </div>
        ) : error ? (
          <div className="flex items-center justify-center h-full text-red-500">
            {error}
          </div>
        ) : (
          <canvas ref={chartRef}></canvas>
        )}
      </div>
      
      <div className="flex justify-end mt-4 space-x-4">
        <button
          className={`text-xs px-3 py-1 rounded ${
            chartType === 'line' 
              ? 'bg-blue-500 text-white' 
              : 'bg-gray-200 text-gray-700 dark:bg-gray-700 dark:text-gray-300'
          }`}
          onClick={() => setChartType('line')}
        >
          Line
        </button>
        <button
          className={`text-xs px-3 py-1 rounded ${
            chartType === 'candlestick' 
              ? 'bg-blue-500 text-white' 
              : 'bg-gray-200 text-gray-700 dark:bg-gray-700 dark:text-gray-300'
          }`}
          onClick={() => setChartType('candlestick')}
        >
          Candlestick
        </button>
        <button
          className={`text-xs px-3 py-1 rounded ${
            showVolume 
              ? 'bg-blue-500 text-white' 
              : 'bg-gray-200 text-gray-700 dark:bg-gray-700 dark:text-gray-300'
          }`}
          onClick={() => setShowVolume(!showVolume)}
        >
          {showVolume ? 'Hide Volume' : 'Show Volume'}
        </button>
        <button
          className={`text-xs px-3 py-1 rounded ${
            showPredictions 
              ? 'bg-blue-500 text-white' 
              : 'bg-gray-200 text-gray-700 dark:bg-gray-700 dark:text-gray-300'
          }`}
          onClick={() => setShowPredictions(!showPredictions)}
        >
          {showPredictions ? 'Hide Predictions' : 'Show Predictions'}
        </button>
        <button
          className="text-xs px-3 py-1 rounded bg-gray-200 text-gray-700 dark:bg-gray-700 dark:text-gray-300"
          onClick={() => {
            if (chartInstance.current) {
              chartInstance.current.resetZoom();
            }
          }}
        >
          Reset Zoom
        </button>
      </div>
    </div>
  );
};

export default StockChart; 