'use client';

import React from 'react';

type InsightData = {
  symbol: string;
  companyName: string;
  currentPrice: string;
  targetPrice: string;
  recommendation: 'BUY' | 'SELL' | 'HOLD';
  confidence: number;
  factors: {
    title: string;
    value: string;
    impact: 'positive' | 'negative' | 'neutral';
  }[];
  technicalIndicators: {
    name: string;
    value: string;
    signal: 'bullish' | 'bearish' | 'neutral';
  }[];
  riskLevel: 'Low' | 'Medium' | 'High';
};

const StockInsights = ({ symbol }: { symbol: string }) => {
  // Mock data for the component demo
  const insightData: InsightData = {
    symbol: symbol || 'AAPL',
    companyName: 'Apple Inc.',
    currentPrice: '$186.40',
    targetPrice: '$210.75',
    recommendation: 'BUY',
    confidence: 87,
    factors: [
      { title: 'Revenue Growth', value: '+8.1% YoY', impact: 'positive' },
      { title: 'Profit Margin', value: '25.3%', impact: 'positive' },
      { title: 'P/E Ratio', value: '30.5', impact: 'neutral' },
      { title: 'Market Trend', value: 'Upward', impact: 'positive' },
      { title: 'Debt-to-Equity', value: '1.8', impact: 'negative' }
    ],
    technicalIndicators: [
      { name: 'Moving Average', value: 'Above 200-day', signal: 'bullish' },
      { name: 'RSI', value: '63', signal: 'neutral' },
      { name: 'MACD', value: 'Positive Crossover', signal: 'bullish' },
      { name: 'Volume', value: '+15% Avg', signal: 'bullish' }
    ],
    riskLevel: 'Medium'
  };

  const getRecommendationColor = (rec: string) => {
    switch (rec) {
      case 'BUY':
        return 'bg-primary/20 text-primary';
      case 'SELL':
        return 'bg-destructive/20 text-destructive';
      case 'HOLD':
        return 'bg-secondary/20 text-secondary';
      default:
        return 'bg-muted text-muted-foreground';
    }
  };

  const getImpactColor = (impact: string) => {
    switch (impact) {
      case 'positive':
        return 'text-blue-500';
      case 'negative':
        return 'text-blue-700';
      case 'neutral':
      default:
        return 'text-muted-foreground';
    }
  };

  const getSignalColor = (signal: string) => {
    switch (signal) {
      case 'bullish':
        return 'text-blue-500';
      case 'bearish':
        return 'text-blue-700';
      case 'neutral':
      default:
        return 'text-muted-foreground';
    }
  };

  const getRiskColor = (risk: string) => {
    switch (risk) {
      case 'Low':
        return 'bg-primary/10 text-primary';
      case 'Medium':
        return 'bg-secondary/10 text-secondary';
      case 'High':
        return 'bg-destructive/10 text-destructive';
      default:
        return 'bg-muted text-muted-foreground';
    }
  };

  return (
    <div className="dashboard-card">
      <div className="flex flex-col md:flex-row justify-between items-start md:items-center mb-6">
        <div>
          <div className="flex items-center">
            <h2 className="text-2xl font-bold">{insightData.symbol}</h2>
            <span className="ml-3 text-muted-foreground">{insightData.companyName}</span>
          </div>
          <p className="text-sm text-muted-foreground mt-1">AI-driven investment analysis</p>
        </div>
        <div className="mt-4 md:mt-0 flex items-center">
          <span className={`px-3 py-1 rounded-full text-sm font-medium ${getRecommendationColor(insightData.recommendation)}`}>
            {insightData.recommendation}
          </span>
          <div className="ml-4 text-right">
            <span className="text-sm text-muted-foreground">AI Confidence</span>
            <div className="flex items-center">
              <div className="w-24 h-2 bg-muted rounded-full mr-2">
                <div 
                  className="h-2 bg-gradient-to-r from-primary/70 to-primary rounded-full" 
                  style={{ width: `${insightData.confidence}%` }}
                ></div>
              </div>
              <span className="text-sm font-medium">{insightData.confidence}%</span>
            </div>
          </div>
        </div>
      </div>
      
      <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
        <div>
          <div className="mb-6">
            <h3 className="text-lg font-medium mb-3">Price Analysis</h3>
            <div className="grid grid-cols-2 gap-4">
              <div className="p-4 border border-border/40 rounded-lg">
                <p className="text-sm text-muted-foreground">Current Price</p>
                <p className="text-xl font-bold mt-1">{insightData.currentPrice}</p>
              </div>
              <div className="p-4 border border-border/40 rounded-lg">
                <p className="text-sm text-muted-foreground">Target Price</p>
                <p className="text-xl font-bold mt-1 text-primary">{insightData.targetPrice}</p>
              </div>
            </div>
          </div>
          
          <div>
            <h3 className="text-lg font-medium mb-3">Key Factors</h3>
            <div className="space-y-3">
              {insightData.factors.map((factor, index) => (
                <div key={index} className="flex justify-between items-center border-b border-border/30 pb-2 last:border-0 last:pb-0">
                  <span className="text-sm">{factor.title}</span>
                  <span className={`text-sm font-medium ${getImpactColor(factor.impact)}`}>
                    {factor.value}
                  </span>
                </div>
              ))}
            </div>
          </div>
        </div>
        
        <div>
          <div className="mb-6">
            <h3 className="text-lg font-medium mb-3">Technical Indicators</h3>
            <div className="grid grid-cols-2 gap-4">
              {insightData.technicalIndicators.map((indicator, index) => (
                <div key={index} className="p-3 border border-border/40 rounded-lg">
                  <p className="text-sm text-muted-foreground">{indicator.name}</p>
                  <div className="flex justify-between items-center mt-1">
                    <p className="font-medium">{indicator.value}</p>
                    <p className={`text-sm ${getSignalColor(indicator.signal)}`}>
                      {indicator.signal === 'bullish' ? '↑' : indicator.signal === 'bearish' ? '↓' : '→'}
                    </p>
                  </div>
                </div>
              ))}
            </div>
          </div>
          
          <div>
            <h3 className="text-lg font-medium mb-3">Risk Assessment</h3>
            <div className="p-4 border border-border/40 rounded-lg">
              <div className="flex justify-between items-center mb-3">
                <span className="text-sm text-muted-foreground">Risk Level</span>
                <span className={`px-2 py-0.5 rounded-full text-xs ${getRiskColor(insightData.riskLevel)}`}>
                  {insightData.riskLevel}
                </span>
              </div>
              <div className="w-full h-2 bg-muted rounded-full">
                <div 
                  className={`h-2 rounded-full ${
                    insightData.riskLevel === 'Low' ? 'bg-primary' :
                    insightData.riskLevel === 'Medium' ? 'bg-secondary' : 'bg-destructive'
                  }`} 
                  style={{ 
                    width: insightData.riskLevel === 'Low' ? '30%' : 
                           insightData.riskLevel === 'Medium' ? '60%' : '90%' 
                  }}
                ></div>
              </div>
            </div>
          </div>
        </div>
      </div>
      
      <div className="mt-8 border-t border-border pt-6">
        <h3 className="text-lg font-medium mb-4">AI Investment Thesis</h3>
        <p className="text-muted-foreground">
          Based on comprehensive analysis of financial data, market trends, and technical indicators, 
          our AI model suggests a {insightData.recommendation.toLowerCase()} position on {insightData.symbol}. 
          The company shows strong revenue growth and healthy profit margins, with technical indicators 
          supporting an upward trend. However, investors should note the slightly elevated debt-to-equity 
          ratio and monitor market volatility closely.
        </p>
        <div className="mt-6 flex space-x-4">
          <button className="btn-primary px-4 py-2 text-sm">
            Add to Watchlist
          </button>
          <button className="btn-secondary px-4 py-2 text-sm">
            View Full Analysis
          </button>
        </div>
      </div>
    </div>
  );
};

export default StockInsights; 