// Chart configuration utility for Recharts/D3 charts

export interface ChartTheme {
  backgroundColor: string;
  gridColor: string;
  textColor: string;
  upColor: string;
  downColor: string;
  lineColor: string;
  volumeColor: string;
  crosshairColor: string;
  indicatorColors: string[];
}

export const THEMES = {
  light: {
    backgroundColor: '#ffffff',
    gridColor: '#f0f0f0',
    textColor: '#333333',
    upColor: '#26a69a',
    downColor: '#ef5350',
    lineColor: '#2962ff',
    volumeColor: '#9e9e9e',
    crosshairColor: 'rgba(0, 0, 0, 0.5)',
    indicatorColors: ['#2962ff', '#f57c00', '#7b1fa2', '#388e3c', '#d32f2f']
  },
  dark: {
    backgroundColor: '#1e1e1e',
    gridColor: '#2a2a2a',
    textColor: '#e0e0e0',
    upColor: '#4caf50',
    downColor: '#f44336',
    lineColor: '#2196f3',
    volumeColor: '#616161',
    crosshairColor: 'rgba(255, 255, 255, 0.5)',
    indicatorColors: ['#2196f3', '#ff9800', '#9c27b0', '#4caf50', '#f44336']
  }
};

export type ChartTimeframe = '1D' | '1W' | '1M' | '3M' | '6M' | '1Y' | '5Y' | 'ALL';

export interface ChartConfig {
  theme: ChartTheme;
  timeframe: ChartTimeframe;
  showVolume: boolean;
  showGrid: boolean;
  showLegend: boolean;
  indicators: string[];
  candlestick: boolean;
  height: number;
  animate: boolean;
}

export const DEFAULT_CONFIG: ChartConfig = {
  theme: THEMES.light,
  timeframe: '1M',
  showVolume: true,
  showGrid: true,
  showLegend: true,
  indicators: ['SMA'],
  candlestick: true,
  height: 500,
  animate: true
};

export function getTimeframeParams(timeframe: ChartTimeframe): { interval: string, period: string } {
  switch (timeframe) {
    case '1D':
      return { interval: '5m', period: '1d' };
    case '1W':
      return { interval: '15m', period: '1w' };
    case '1M':
      return { interval: '1d', period: '1mo' };
    case '3M':
      return { interval: '1d', period: '3mo' };
    case '6M':
      return { interval: '1d', period: '6mo' };
    case '1Y':
      return { interval: '1d', period: '1y' };
    case '5Y':
      return { interval: '1wk', period: '5y' };
    case 'ALL':
      return { interval: '1mo', period: 'max' };
    default:
      return { interval: '1d', period: '1mo' };
  }
}

export default {
  THEMES,
  DEFAULT_CONFIG,
  getTimeframeParams
}; 