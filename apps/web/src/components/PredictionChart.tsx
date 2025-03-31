'use client';

import { useState, useEffect } from 'react';
import { Line } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler,
} from 'chart.js';

import { type PredictionResponse, type ModelType } from '@repo/api';
import { formatDate } from '../utils/dateUtils';

// Register ChartJS components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler
);

interface PredictionChartProps {
  prediction: PredictionResponse;
  historicalData?: { date: string; price: number }[];
  title?: string;
  height?: number;
  showConfidence?: boolean;
}

export function PredictionChart({
  prediction,
  historicalData = [],
  title = 'Price Prediction',
  height = 300,
  showConfidence = true,
}: PredictionChartProps) {
  const [chartData, setChartData] = useState<any>(null);

  useEffect(() => {
    if (!prediction) return;

    // Combine historical and prediction data
    const historicalDates = historicalData.map(item => formatDate(new Date(item.date)));
    const historicalPrices = historicalData.map(item => item.price);

    const predictionDates = prediction.predictions.map(item => formatDate(new Date(item.date)));
    const predictionPrices = prediction.predictions.map(item => item.price);
    const upperBounds = prediction.predictions.map(item => item.upperBound || null);
    const lowerBounds = prediction.predictions.map(item => item.lowerBound || null);

    const allDates = [...historicalDates, ...predictionDates];

    // Create datasets
    const datasets = [
      {
        label: 'Historical',
        data: [...historicalPrices, ...Array(predictionDates.length).fill(null)],
        borderColor: 'rgba(54, 162, 235, 1)',
        backgroundColor: 'rgba(54, 162, 235, 0.2)',
        pointRadius: 2,
        tension: 0.1,
      },
      {
        label: `Prediction (${getModelLabel(prediction.modelType)})`,
        data: [...Array(historicalDates.length).fill(null), ...predictionPrices],
        borderColor: 'rgba(255, 99, 132, 1)',
        backgroundColor: 'rgba(255, 99, 132, 0.2)',
        pointRadius: 3,
        borderWidth: 2,
        tension: 0.1,
      },
    ];

    // Add confidence interval if bounds exist and showConfidence is true
    if (showConfidence && upperBounds.some(bound => bound !== null) && lowerBounds.some(bound => bound !== null)) {
      const confidenceData = [...Array(historicalDates.length).fill(null)];
      
      // Add prediction data with upper and lower bounds
      for (let i = 0; i < predictionPrices.length; i++) {
        confidenceData.push({
          y: predictionPrices[i],
          y1: lowerBounds[i],
          y2: upperBounds[i],
        });
      }

      datasets.push({
        label: 'Confidence Interval',
        data: confidenceData,
        fill: true,
        backgroundColor: 'rgba(255, 99, 132, 0.1)',
        borderColor: 'transparent',
        pointRadius: 0,
        tension: 0.1,
      });
    }

    setChartData({
      labels: allDates,
      datasets,
    });
  }, [prediction, historicalData, showConfidence]);

  if (!chartData) {
    return <div className="flex items-center justify-center h-64">Loading chart...</div>;
  }

  return (
    <div className="rounded-lg bg-white p-4 shadow-sm">
      <h3 className="text-lg font-semibold mb-2">{title}</h3>
      <div style={{ height: `${height}px` }}>
        <Line
          data={chartData}
          options={{
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
              mode: 'index',
              intersect: false,
            },
            plugins: {
              legend: {
                position: 'top',
              },
              tooltip: {
                callbacks: {
                  label: function (context) {
                    let label = context.dataset.label || '';
                    if (label) {
                      label += ': ';
                    }
                    if (context.raw === null) {
                      return label + 'N/A';
                    }
                    if (typeof context.raw === 'object' && context.raw !== null) {
                      const dataPoint = context.raw as { y: number; y1?: number; y2?: number };
                      if (dataPoint.y1 !== undefined && dataPoint.y2 !== undefined) {
                        return [
                          `${label}$${dataPoint.y.toFixed(2)}`,
                          `Upper: $${dataPoint.y2.toFixed(2)}`,
                          `Lower: $${dataPoint.y1.toFixed(2)}`
                        ];
                      }
                      return label + `$${dataPoint.y.toFixed(2)}`;
                    }
                    return label + `$${context.raw.toFixed(2)}`;
                  }
                }
              }
            },
            scales: {
              y: {
                ticks: {
                  callback: function (value) {
                    return '$' + value;
                  }
                }
              }
            }
          }}
        />
      </div>
      <div className="mt-2 text-sm text-gray-500 flex items-center">
        <span className="font-medium">Confidence:</span>
        <span className="ml-1">{(prediction.confidence * 100).toFixed(1)}%</span>
        <span className="ml-4 font-medium">Model:</span>
        <span className="ml-1">{getModelLabel(prediction.modelType)}</span>
      </div>
    </div>
  );
}

// Helper function to get a human-readable model label
function getModelLabel(modelType: ModelType): string {
  switch (modelType) {
    case 'LSTM':
      return 'LSTM';
    case 'ARIMA':
      return 'ARIMA-GARCH';
    case 'PROPHET':
      return 'Prophet';
    case 'ENSEMBLE':
      return 'Ensemble';
    case 'SENTIMENT':
      return 'Sentiment';
    default:
      return modelType;
  }
} 