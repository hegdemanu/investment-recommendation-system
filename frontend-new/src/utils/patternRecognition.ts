// Pattern recognition utility for stock charts

interface OHLC {
  date: Date;
  open: number;
  high: number;
  close: number;
  low: number;
  volume?: number;
}

export interface Pattern {
  name: string;
  startIndex: number;
  endIndex: number;
  bullish: boolean;
  confidence: number; // 0-1
}

/**
 * Detect common candlestick patterns
 */
export function detectPatterns(data: OHLC[]): Pattern[] {
  if (!data || data.length < 5) return [];
  
  const patterns: Pattern[] = [];
  
  // Check for doji pattern (open and close prices are very close)
  for (let i = 0; i < data.length; i++) {
    const candle = data[i];
    const bodySize = Math.abs(candle.close - candle.open);
    const totalSize = candle.high - candle.low;
    
    if (totalSize > 0 && bodySize / totalSize < 0.1) {
      patterns.push({
        name: 'Doji',
        startIndex: i,
        endIndex: i,
        bullish: true,
        confidence: 0.6
      });
    }
  }
  
  // Check for bullish engulfing pattern
  for (let i = 1; i < data.length; i++) {
    const prev = data[i - 1];
    const curr = data[i];
    
    if (
      prev.close < prev.open && // Previous candle is bearish
      curr.close > curr.open && // Current candle is bullish
      curr.open < prev.close && // Current open is lower than previous close
      curr.close > prev.open // Current close is higher than previous open
    ) {
      patterns.push({
        name: 'Bullish Engulfing',
        startIndex: i - 1,
        endIndex: i,
        bullish: true,
        confidence: 0.7
      });
    }
  }
  
  // Check for bearish engulfing pattern
  for (let i = 1; i < data.length; i++) {
    const prev = data[i - 1];
    const curr = data[i];
    
    if (
      prev.close > prev.open && // Previous candle is bullish
      curr.close < curr.open && // Current candle is bearish
      curr.open > prev.close && // Current open is higher than previous close
      curr.close < prev.open // Current close is lower than previous open
    ) {
      patterns.push({
        name: 'Bearish Engulfing',
        startIndex: i - 1,
        endIndex: i,
        bullish: false,
        confidence: 0.7
      });
    }
  }
  
  return patterns;
}

/**
 * Analyze price action for potential support and resistance levels
 */
export function findSupportResistance(data: OHLC[], lookback: number = 20): { support: number[], resistance: number[] } {
  if (!data || data.length < lookback) {
    return { support: [], resistance: [] };
  }
  
  const recentData = data.slice(-lookback);
  const levels = new Map<number, number>(); // price -> count
  
  // Round prices to significant levels
  recentData.forEach(candle => {
    // Determine rounding factor based on price magnitude
    const priceMagnitude = Math.floor(Math.log10(candle.close));
    const roundingFactor = Math.pow(10, priceMagnitude - 2);
    
    const highRounded = Math.round(candle.high / roundingFactor) * roundingFactor;
    const lowRounded = Math.round(candle.low / roundingFactor) * roundingFactor;
    
    levels.set(highRounded, (levels.get(highRounded) || 0) + 1);
    levels.set(lowRounded, (levels.get(lowRounded) || 0) + 1);
  });
  
  // Filter significant levels (touched multiple times)
  const significantLevels = Array.from(levels.entries())
    .filter(([_, count]) => count >= 3)
    .map(([price, _]) => price)
    .sort((a, b) => a - b);
  
  const lastPrice = data[data.length - 1].close;
  
  // Separate into support and resistance
  return {
    support: significantLevels.filter(level => level < lastPrice),
    resistance: significantLevels.filter(level => level > lastPrice)
  };
}

export default {
  detectPatterns,
  findSupportResistance
}; 