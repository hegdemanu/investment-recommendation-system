'use client';

import React, { useState, useEffect, useRef } from 'react';
import { getTimeSeriesData, TimeSeriesData } from '@/services/stockApi';

type InteractiveChartProps = {
  symbol: string;
  timeframe?: string;
};

const InteractiveChart = ({ symbol, timeframe = '1M' }: InteractiveChartProps) => {
  const [chartData, setChartData] = useState<TimeSeriesData[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [selectedTimeframe, setSelectedTimeframe] = useState<string>(timeframe);
  const [error, setError] = useState<string | null>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const fetchData = async () => {
      if (!symbol) return;
      
      setIsLoading(true);
      setError(null);
      
      try {
        // Map UI timeframes to API intervals
        let interval: 'daily' | 'weekly' | 'monthly';
        let outputSize: 'compact' | 'full' = 'compact';
        
        switch (selectedTimeframe) {
          case '1D':
          case '1W':
          case '1M':
            interval = 'daily';
            break;
          case '3M':
          case '6M':
            interval = 'weekly';
            break;
          case '1Y':
          case 'ALL':
            interval = 'monthly';
            break;
          default:
            interval = 'daily';
        }
        
        const data = await getTimeSeriesData(symbol, interval, outputSize);
        
        // Filter data based on timeframe
        let filteredData = data;
        const now = new Date();
        
        switch (selectedTimeframe) {
          case '1D':
            filteredData = data.filter(item => {
              const date = new Date(item.date);
              const diffTime = Math.abs(now.getTime() - date.getTime());
              const diffDays = Math.ceil(diffTime / (1000 * 60 * 60 * 24));
              return diffDays <= 1;
            });
            break;
          case '1W':
            filteredData = data.filter(item => {
              const date = new Date(item.date);
              const diffTime = Math.abs(now.getTime() - date.getTime());
              const diffDays = Math.ceil(diffTime / (1000 * 60 * 60 * 24));
              return diffDays <= 7;
            });
            break;
          case '1M':
            filteredData = data.filter(item => {
              const date = new Date(item.date);
              const diffTime = Math.abs(now.getTime() - date.getTime());
              const diffDays = Math.ceil(diffTime / (1000 * 60 * 60 * 24));
              return diffDays <= 30;
            });
            break;
          case '3M':
            filteredData = data.filter(item => {
              const date = new Date(item.date);
              const diffTime = Math.abs(now.getTime() - date.getTime());
              const diffDays = Math.ceil(diffTime / (1000 * 60 * 60 * 24));
              return diffDays <= 90;
            });
            break;
          case '6M':
            filteredData = data.filter(item => {
              const date = new Date(item.date);
              const diffTime = Math.abs(now.getTime() - date.getTime());
              const diffDays = Math.ceil(diffTime / (1000 * 60 * 60 * 24));
              return diffDays <= 180;
            });
            break;
          case '1Y':
            filteredData = data.filter(item => {
              const date = new Date(item.date);
              const diffTime = Math.abs(now.getTime() - date.getTime());
              const diffDays = Math.ceil(diffTime / (1000 * 60 * 60 * 24));
              return diffDays <= 365;
            });
            break;
        }
        
        setChartData(filteredData);
      } catch (err) {
        console.error('Error fetching chart data:', err);
        setError('Failed to load chart data');
      } finally {
        setIsLoading(false);
      }
    };
    
    fetchData();
  }, [symbol, selectedTimeframe]);
  
  useEffect(() => {
    if (chartData.length === 0 || !canvasRef.current) return;
    
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    
    // Set canvas dimensions
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, rect.width, rect.height);
    
    // Define chart dimensions
    const padding = { top: 20, right: 20, bottom: 30, left: 40 };
    const chartWidth = rect.width - padding.left - padding.right;
    const chartHeight = rect.height - padding.top - padding.bottom;
    
    // Find data range
    const minValue = Math.min(...chartData.map(d => d.low));
    const maxValue = Math.max(...chartData.map(d => d.high));
    const valueRange = maxValue - minValue;
    
    // Scale functions
    const xScale = (i: number) => padding.left + (i / (chartData.length - 1)) * chartWidth;
    const yScale = (value: number) => padding.top + chartHeight - ((value - minValue) / valueRange) * chartHeight;
    
    // Draw axes
    ctx.strokeStyle = '#d1d5db'; // border color
    ctx.lineWidth = 1;
    
    // Draw y-axis
    ctx.beginPath();
    ctx.moveTo(padding.left, padding.top);
    ctx.lineTo(padding.left, padding.top + chartHeight);
    ctx.stroke();
    
    // Draw x-axis
    ctx.beginPath();
    ctx.moveTo(padding.left, padding.top + chartHeight);
    ctx.lineTo(padding.left + chartWidth, padding.top + chartHeight);
    ctx.stroke();
    
    // Draw price line
    ctx.strokeStyle = '#3b82f6'; // primary color
    ctx.lineWidth = 2;
    ctx.beginPath();
    
    chartData.forEach((d, i) => {
      const x = xScale(i);
      const y = yScale(d.close);
      
      if (i === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    });
    
    ctx.stroke();
    
    // Draw area under the line
    ctx.fillStyle = 'rgba(59, 130, 246, 0.1)'; // primary color with opacity
    ctx.beginPath();
    
    // Start from the bottom left
    ctx.moveTo(xScale(0), yScale(minValue));
    
    // Draw the line path
    chartData.forEach((d, i) => {
      const x = xScale(i);
      const y = yScale(d.close);
      ctx.lineTo(x, y);
    });
    
    // Complete the path to the bottom right and fill
    ctx.lineTo(xScale(chartData.length - 1), yScale(minValue));
    ctx.closePath();
    ctx.fill();
    
    // Draw labels
    ctx.fillStyle = '#6b7280'; // text-muted-foreground
    ctx.font = '10px sans-serif';
    ctx.textAlign = 'right';
    
    // Y-axis labels (price)
    const numYTicks = 5;
    for (let i = 0; i <= numYTicks; i++) {
      const value = minValue + (valueRange / numYTicks) * i;
      const y = yScale(value);
      
      ctx.fillText(value.toFixed(2), padding.left - 5, y + 3);
      
      // Draw grid line
      ctx.strokeStyle = '#e5e7eb';
      ctx.lineWidth = 0.5;
      ctx.beginPath();
      ctx.moveTo(padding.left, y);
      ctx.lineTo(padding.left + chartWidth, y);
      ctx.stroke();
    }
    
    // X-axis labels (dates)
    ctx.textAlign = 'center';
    const labelStep = Math.max(1, Math.floor(chartData.length / 5));
    
    for (let i = 0; i < chartData.length; i += labelStep) {
      const d = chartData[i];
      const x = xScale(i);
      
      const date = new Date(d.date);
      const label = date.toLocaleDateString(undefined, { 
        month: 'short', 
        day: 'numeric'
      });
      
      ctx.fillText(label, x, padding.top + chartHeight + 15);
    }
    
  }, [chartData]);

  return (
    <div className="dashboard-card">
      {error && (
        <div className="bg-red-100 dark:bg-red-900/20 text-red-800 dark:text-red-200 p-4 rounded-md mb-4">
          {error}
        </div>
      )}
      
      <div className="flex justify-between items-center mb-4">
        <h3 className="font-medium">{symbol} Price Chart</h3>
        <div className="flex space-x-1">
          {['1D', '1W', '1M', '3M', '6M', '1Y', 'ALL'].map((tf) => (
            <button
              key={tf}
              onClick={() => setSelectedTimeframe(tf)}
              className={`px-2 py-1 text-xs rounded ${
                selectedTimeframe === tf
                  ? 'bg-primary text-white'
                  : 'bg-muted text-muted-foreground'
              }`}
            >
              {tf}
            </button>
          ))}
        </div>
      </div>
      
      <div className="relative h-64 md:h-80">
        {isLoading ? (
          <div className="absolute inset-0 flex items-center justify-center">
            <div className="w-8 h-8 border-4 border-primary/30 border-t-primary rounded-full animate-spin"></div>
          </div>
        ) : chartData.length === 0 ? (
          <div className="absolute inset-0 flex items-center justify-center">
            <p className="text-muted-foreground">No chart data available</p>
          </div>
        ) : (
          <canvas 
            ref={canvasRef} 
            className="w-full h-full"
            style={{ width: '100%', height: '100%' }}
          ></canvas>
        )}
      </div>
      
      {chartData.length > 0 && (
        <div className="mt-4 grid grid-cols-4 gap-4">
          <div>
            <p className="text-xs text-muted-foreground">Open</p>
            <p className="font-medium">${chartData[chartData.length - 1]?.open.toFixed(2)}</p>
          </div>
          <div>
            <p className="text-xs text-muted-foreground">Close</p>
            <p className="font-medium">${chartData[chartData.length - 1]?.close.toFixed(2)}</p>
          </div>
          <div>
            <p className="text-xs text-muted-foreground">High</p>
            <p className="font-medium">${chartData[chartData.length - 1]?.high.toFixed(2)}</p>
          </div>
          <div>
            <p className="text-xs text-muted-foreground">Low</p>
            <p className="font-medium">${chartData[chartData.length - 1]?.low.toFixed(2)}</p>
          </div>
        </div>
      )}
    </div>
  );
};

export default InteractiveChart; 