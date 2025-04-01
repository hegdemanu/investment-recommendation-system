import { NextRequest, NextResponse } from 'next/server';

export async function GET(request: NextRequest) {
  const searchParams = request.nextUrl.searchParams;
  const symbol = searchParams.get('symbol') || 'AAPL';
  const period = searchParams.get('period') || '1d';

  // Random price generation based on symbol
  const basePrice = getBasePrice(symbol);
  const volatility = getVolatility(symbol);
  
  // Generate price data
  const priceData = generatePriceData(basePrice, volatility, period);
  
  // Mock stock data response
  const stockData = {
    symbol,
    name: getCompanyName(symbol),
    currentPrice: priceData[priceData.length - 1].price,
    change: +(priceData[priceData.length - 1].price - priceData[0].price).toFixed(2),
    changePercent: +((priceData[priceData.length - 1].price - priceData[0].price) / priceData[0].price * 100).toFixed(2),
    currency: 'USD',
    marketCap: getMarketCap(symbol, priceData[priceData.length - 1].price),
    volume: Math.round(Math.random() * 10000000 + 1000000),
    peRatio: +(Math.random() * 30 + 10).toFixed(2),
    dividendYield: +(Math.random() * 3).toFixed(2),
    priceData,
    lastUpdated: new Date().toISOString()
  };

  return NextResponse.json(stockData);
}

// Helper functions
function getBasePrice(symbol: string): number {
  const prices: Record<string, number> = {
    'AAPL': 175,
    'MSFT': 380,
    'GOOGL': 145,
    'AMZN': 175,
    'META': 450,
    'TSLA': 180,
    'NVDA': 820,
    'JPM': 190,
    'V': 270,
    'WMT': 60
  };
  
  return prices[symbol] || 100 + (symbol.charCodeAt(0) % 10) * 20;
}

function getVolatility(symbol: string): number {
  const volatility: Record<string, number> = {
    'AAPL': 0.015,
    'MSFT': 0.012,
    'GOOGL': 0.018,
    'AMZN': 0.022,
    'META': 0.025,
    'TSLA': 0.035,
    'NVDA': 0.028,
    'JPM': 0.014,
    'V': 0.011,
    'WMT': 0.008
  };
  
  return volatility[symbol] || 0.02;
}

function getCompanyName(symbol: string): string {
  const names: Record<string, string> = {
    'AAPL': 'Apple Inc.',
    'MSFT': 'Microsoft Corporation',
    'GOOGL': 'Alphabet Inc.',
    'AMZN': 'Amazon.com Inc.',
    'META': 'Meta Platforms, Inc.',
    'TSLA': 'Tesla, Inc.',
    'NVDA': 'NVIDIA Corporation',
    'JPM': 'JPMorgan Chase & Co.',
    'V': 'Visa Inc.',
    'WMT': 'Walmart Inc.'
  };
  
  return names[symbol] || `${symbol} Corporation`;
}

function getMarketCap(symbol: string, currentPrice: number): number {
  const sharesOutstanding: Record<string, number> = {
    'AAPL': 15.7,
    'MSFT': 7.4,
    'GOOGL': 12.8,
    'AMZN': 10.3,
    'META': 2.5,
    'TSLA': 3.1,
    'NVDA': 2.4,
    'JPM': 2.9,
    'V': 2.0,
    'WMT': 2.7
  };
  
  const billions = sharesOutstanding[symbol] || 1.5;
  return +(billions * currentPrice).toFixed(2) * 1000000000;
}

function generatePriceData(basePrice: number, volatility: number, period: string) {
  let dataPoints = 30;
  
  switch(period) {
    case '1d':
      dataPoints = 24; // Hourly data for a day
      break;
    case '1w':
      dataPoints = 7; // Daily data for a week
      break;
    case '1m':
      dataPoints = 30; // Daily data for a month
      break;
    case '3m':
      dataPoints = 90; // Daily data for 3 months
      break;
    case '1y':
      dataPoints = 12; // Monthly data for a year
      break;
    case '5y':
      dataPoints = 60; // Monthly data for 5 years
      break;
  }
  
  const prices = [];
  let currentPrice = basePrice;
  
  // Generate data points with some randomness but also a trend
  const trend = Math.random() > 0.5 ? 1 : -1; // Random trend direction
  
  for (let i = 0; i < dataPoints; i++) {
    const change = (Math.random() - 0.5) * volatility * currentPrice;
    const trendFactor = trend * (i / dataPoints) * volatility * currentPrice * 0.5;
    currentPrice = Math.max(currentPrice + change + trendFactor, 1); // Ensure price doesn't go below 1
    
    // Date for this data point
    const date = new Date();
    
    if (period === '1d') {
      date.setHours(date.getHours() - (dataPoints - i));
    } else if (period === '1w') {
      date.setDate(date.getDate() - (dataPoints - i));
    } else if (period === '1m') {
      date.setDate(date.getDate() - (dataPoints - i));
    } else if (period === '3m') {
      date.setDate(date.getDate() - (dataPoints - i));
    } else if (period === '1y') {
      date.setMonth(date.getMonth() - (dataPoints - i));
    } else if (period === '5y') {
      date.setMonth(date.getMonth() - (dataPoints - i));
    }
    
    prices.push({
      date: date.toISOString(),
      price: +currentPrice.toFixed(2)
    });
  }
  
  return prices;
} 