import axios from 'axios';

// Using environment variable for API key
const API_KEY = process.env.NEXT_PUBLIC_ALPHA_VANTAGE_API_KEY || 'demo';
const BASE_URL = 'https://www.alphavantage.co/query';

// Interface for stock quote data
export interface StockQuote {
  symbol: string;
  price: number;
  change: number;
  changePercent: number;
  high: number;
  low: number;
  volume: number;
  latestTradingDay: string;
}

// Interface for stock time series data
export interface TimeSeriesData {
  date: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

// Interface for stock details including company info
export interface StockDetails {
  symbol: string;
  name: string;
  description: string;
  sector: string;
  industry: string;
  marketCap: number;
  peRatio: number;
  dividendYield: number;
  fiftyTwoWeekHigh: number;
  fiftyTwoWeekLow: number;
}

/**
 * Fetch real-time quote for a stock
 */
export const getStockQuote = async (symbol: string): Promise<StockQuote> => {
  try {
    const response = await axios.get(BASE_URL, {
      params: {
        function: 'GLOBAL_QUOTE',
        symbol,
        apikey: API_KEY
      }
    });

    const data = response.data['Global Quote'];
    
    if (!data || Object.keys(data).length === 0) {
      console.warn(`No data available for ${symbol}, using mock data`);
      return getMockStockQuote(symbol);
    }

    return {
      symbol: data['01. symbol'],
      price: parseFloat(data['05. price']),
      change: parseFloat(data['09. change']),
      changePercent: parseFloat(data['10. change percent'].replace('%', '')),
      high: parseFloat(data['03. high']),
      low: parseFloat(data['04. low']),
      volume: parseInt(data['06. volume']),
      latestTradingDay: data['07. latest trading day']
    };
  } catch (error) {
    console.warn(`Error fetching stock quote for ${symbol}, using mock data:`, error);
    return getMockStockQuote(symbol);
  }
};

/**
 * Fetch historical time series data for a stock
 */
export const getTimeSeriesData = async (
  symbol: string, 
  interval: 'daily' | 'weekly' | 'monthly' = 'daily',
  outputSize: 'compact' | 'full' = 'compact'
): Promise<TimeSeriesData[]> => {
  try {
    let functionName;
    switch (interval) {
      case 'weekly':
        functionName = 'TIME_SERIES_WEEKLY';
        break;
      case 'monthly':
        functionName = 'TIME_SERIES_MONTHLY';
        break;
      default:
        functionName = 'TIME_SERIES_DAILY';
    }

    const response = await axios.get(BASE_URL, {
      params: {
        function: functionName,
        symbol,
        outputsize: outputSize,
        apikey: API_KEY
      }
    });

    const timeSeriesKey = `Time Series (${interval === 'daily' ? 'Daily' : interval === 'weekly' ? 'Weekly' : 'Monthly'})`;
    const timeSeries = response.data[timeSeriesKey];
    
    if (!timeSeries) {
      console.warn(`No time series data available for ${symbol}, using mock data`);
      return getMockTimeSeriesData(symbol, interval);
    }

    return Object.entries(timeSeries).map(([date, values]: [string, any]) => ({
      date,
      open: parseFloat(values['1. open']),
      high: parseFloat(values['2. high']),
      low: parseFloat(values['3. low']),
      close: parseFloat(values['4. close']),
      volume: parseInt(values['5. volume'])
    })).sort((a, b) => new Date(a.date).getTime() - new Date(b.date).getTime());
  } catch (error) {
    console.warn(`Error fetching time series data for ${symbol}, using mock data:`, error);
    return getMockTimeSeriesData(symbol, interval);
  }
};

/**
 * Search for stocks by keyword
 */
export const searchStocks = async (keywords: string): Promise<{symbol: string, name: string}[]> => {
  try {
    const response = await axios.get(BASE_URL, {
      params: {
        function: 'SYMBOL_SEARCH',
        keywords,
        apikey: API_KEY
      }
    });

    const matches = response.data.bestMatches;
    
    if (!matches || matches.length === 0) {
      return [];
    }

    return matches.map((match: any) => ({
      symbol: match['1. symbol'],
      name: match['2. name']
    }));
  } catch (error) {
    console.error('Error searching stocks:', error);
    // Return mock data for demo purposes if API call fails
    return getMockSearchResults(keywords);
  }
};

/**
 * Get company overview for a stock
 */
export const getCompanyOverview = async (symbol: string): Promise<StockDetails> => {
  try {
    const response = await axios.get(BASE_URL, {
      params: {
        function: 'OVERVIEW',
        symbol,
        apikey: API_KEY
      }
    });

    const data = response.data;
    
    if (!data || Object.keys(data).length === 0) {
      throw new Error('No company data available');
    }

    return {
      symbol: data.Symbol,
      name: data.Name,
      description: data.Description,
      sector: data.Sector,
      industry: data.Industry,
      marketCap: parseFloat(data.MarketCapitalization),
      peRatio: parseFloat(data.PERatio),
      dividendYield: parseFloat(data.DividendYield),
      fiftyTwoWeekHigh: parseFloat(data['52WeekHigh']),
      fiftyTwoWeekLow: parseFloat(data['52WeekLow'])
    };
  } catch (error) {
    console.error('Error fetching company overview:', error);
    // Return mock data for demo purposes if API call fails
    return getMockCompanyOverview(symbol);
  }
};

// Mock data generators for fallback when API limits are reached
const getMockStockQuote = (symbol: string): StockQuote => {
  // Use a seed based on the symbol to generate consistent mock data
  const seed = symbol.split('').reduce((acc, char) => acc + char.charCodeAt(0), 0);
  const basePrice = 100 + (seed % 1000);
  const change = (Math.sin(seed) * 10);
  const changePercent = (change / basePrice) * 100;
  
  return {
    symbol,
    price: basePrice + change,
    change: change,
    changePercent: changePercent,
    high: basePrice + Math.abs(change) + (Math.random() * 5),
    low: basePrice - Math.abs(change) - (Math.random() * 5),
    volume: Math.floor(1000000 + (Math.random() * 9000000)),
    latestTradingDay: new Date().toISOString().split('T')[0]
  };
};

const getMockTimeSeriesData = (
  symbol: string, 
  interval: 'daily' | 'weekly' | 'monthly'
): TimeSeriesData[] => {
  const data: TimeSeriesData[] = [];
  const days = interval === 'daily' ? 30 : interval === 'weekly' ? 12 : 6;
  const seed = symbol.split('').reduce((acc, char) => acc + char.charCodeAt(0), 0);
  const basePrice = 100 + (seed % 1000);
  
  for (let i = 0; i < days; i++) {
    const date = new Date();
    date.setDate(date.getDate() - (interval === 'daily' ? i : i * (interval === 'weekly' ? 7 : 30)));
    
    const trend = Math.sin(i / (days / 6)) * 20;
    const noise = (Math.random() * 10 - 5);
    const close = basePrice + trend + noise;
    const open = close + (Math.random() * 5 - 2.5);
    const high = Math.max(open, close) + Math.random() * 5;
    const low = Math.min(open, close) - Math.random() * 5;
    
    data.push({
      date: date.toISOString().split('T')[0],
      open,
      high,
      low,
      close,
      volume: Math.floor(1000000 + (Math.random() * 9000000))
    });
  }
  
  return data.sort((a, b) => new Date(a.date).getTime() - new Date(b.date).getTime());
};

const getMockSearchResults = (keywords: string): { symbol: string; name: string }[] => {
  const mockResults = [
    { symbol: 'AAPL', name: 'Apple Inc.' },
    { symbol: 'MSFT', name: 'Microsoft Corporation' },
    { symbol: 'GOOGL', name: 'Alphabet Inc.' },
    { symbol: 'AMZN', name: 'Amazon.com Inc.' },
    { symbol: 'TSLA', name: 'Tesla Inc.' },
    { symbol: 'META', name: 'Meta Platforms Inc.' },
    { symbol: 'NFLX', name: 'Netflix Inc.' },
    { symbol: 'NVDA', name: 'NVIDIA Corporation' },
    { symbol: 'PYPL', name: 'PayPal Holdings Inc.' },
    { symbol: 'ADBE', name: 'Adobe Inc.' }
  ];
  
  // Filter based on search keywords (case insensitive)
  const lowerKeywords = keywords.toLowerCase();
  return mockResults.filter(
    result => 
      result.symbol.toLowerCase().includes(lowerKeywords) || 
      result.name.toLowerCase().includes(lowerKeywords)
  );
};

const getMockCompanyOverview = (symbol: string): StockDetails => {
  const companies: Record<string, Partial<StockDetails>> = {
    'AAPL': {
      name: 'Apple Inc',
      description: 'Apple Inc. designs, manufactures, and markets smartphones, personal computers, tablets, wearables, and accessories worldwide.',
      sector: 'Technology',
      industry: 'Consumer Electronics'
    },
    'MSFT': {
      name: 'Microsoft Corporation',
      description: 'Microsoft Corporation develops, licenses, and supports software, services, devices, and solutions worldwide.',
      sector: 'Technology',
      industry: 'Software—Infrastructure'
    },
    'AMZN': {
      name: 'Amazon.com Inc',
      description: 'Amazon.com, Inc. engages in the retail sale of consumer products and subscriptions through online and physical stores worldwide.',
      sector: 'Consumer Cyclical',
      industry: 'Internet Retail'
    },
    'GOOGL': {
      name: 'Alphabet Inc',
      description: 'Alphabet Inc. provides various products and platforms in the United States, Europe, the Middle East, Africa, the Asia-Pacific, Canada, and Latin America.',
      sector: 'Communication Services',
      industry: 'Internet Content & Information'
    }
  };
  
  const baseDetails = companies[symbol] || {
    name: `${symbol} Corporation`,
    description: `${symbol} is a leading company in its industry.`,
    sector: 'Technology',
    industry: 'Software'
  };
  
  return {
    symbol,
    name: baseDetails.name || `${symbol} Corporation`,
    description: baseDetails.description || `${symbol} is a leading company in its industry.`,
    sector: baseDetails.sector || 'Technology',
    industry: baseDetails.industry || 'Software',
    marketCap: Math.random() * 1000000000000,
    peRatio: 10 + Math.random() * 40,
    dividendYield: Math.random() * 3,
    fiftyTwoWeekHigh: 200 + Math.random() * 100,
    fiftyTwoWeekLow: 100 + Math.random() * 50
  };
}; 