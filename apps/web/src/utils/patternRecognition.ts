interface CandleData {
  date: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

interface Pattern {
  type: 'bullish' | 'bearish';
  name: string;
  description: string;
  confidence: number;
  performance?: {
    winRate: number;
    avgProfit: number;
    avgLoss: number;
    riskRewardRatio: number;
    maxDrawdown: number;
    profitFactor: number;
    sharpeRatio: number;
    recoveryFactor: number;
    consecutiveWins: number;
    consecutiveLosses: number;
    largestWin: number;
    largestLoss: number;
    averageHoldingPeriod: number;
    winLossRatio: number;
  };
}

interface CustomPattern {
  name: string;
  points: { x: number; y: number }[];
  tolerance: number;
  type: 'bullish' | 'bearish';
  description: string;
}

interface PatternVariation {
  name: string;
  criteria: (data: CandleData[]) => boolean;
  confidence: number;
}

interface PatternCombination {
  patterns: Pattern[];
  synergy: number; // How well patterns complement each other
  timeframe: number; // Time window for the combination
  reliability: number; // Historical success rate
}

interface PatternForecast {
  probability: number;
  expectedReturn: number;
  timeframe: number;
  confidence: number;
  factors: {
    technical: number;
    volume: number;
    sentiment: number;
    historical: number;
  };
}

interface CandleFunnelingSettings {
  volumeThreshold: number;
  bodyToWickRatio: number;
  momentumThreshold: number;
  trendStrength: number;
  volatilityFilter: number;
  gapThreshold: number;
}

interface CombinationMetrics {
  trendAlignment: number;      // How well patterns align with the overall trend
  volumeProfile: number;       // Volume confirmation strength
  momentumSync: number;        // Momentum synchronization between patterns
  timeframeHarmony: number;    // How well timeframes complement each other
  reversalStrength: number;    // Combined reversal signal strength
  supportResistance: number;   // Proximity to key levels
  volatilityProfile: number;   // Volatility characteristics match
  patternSequence: number;     // Sequential pattern formation quality
}

interface UserPreferences {
  favoritePatterns: string[];
  customThresholds: { [key: string]: number };
  timeframes: number[];
  riskTolerance: number;
  notificationSettings: {
    email: boolean;
    push: boolean;
    patterns: string[];
    minConfidence: number;
  };
  candleFunneling: CandleFunnelingSettings;
  combinationPreferences: {
    minSynergy: number;
    minReliability: number;
    preferredTimeframes: number[];
    weightings: {
      trendAlignment: number;
      volumeProfile: number;
      momentumSync: number;
      timeframeHarmony: number;
      reversalStrength: number;
      supportResistance: number;
      volatilityProfile: number;
      patternSequence: number;
    };
  };
  chartSettings: {
    candleColors: {
      bullish: string;
      bearish: string;
      neutral: string;
    };
    wickColors: {
      bullish: string;
      bearish: string;
    };
    backgroundColors: {
      main: string;
      alt: string;
    };
    gridLines: {
      show: boolean;
      color: string;
      opacity: number;
    };
    annotations: {
      showPriceLabels: boolean;
      showVolume: boolean;
      showPatternLabels: boolean;
      fontSize: number;
    };
  };
}

export class PatternRecognition {
  private static readonly THRESHOLD = 0.02; // 2% threshold for pattern detection
  private static customPatterns: CustomPattern[] = [];
  private static userPreferences: UserPreferences = {
    favoritePatterns: [],
    customThresholds: {},
    timeframes: [5, 15, 30, 60],
    riskTolerance: 0.5,
    notificationSettings: {
      email: false,
      push: true,
      patterns: [],
      minConfidence: 75
    },
    candleFunneling: {
      volumeThreshold: 1.5,
      bodyToWickRatio: 0.6,
      momentumThreshold: 0.02,
      trendStrength: 0.7,
      volatilityFilter: 0.015,
      gapThreshold: 0.01
    },
    combinationPreferences: {
      minSynergy: 0.7,
      minReliability: 0.65,
      preferredTimeframes: [5, 15, 30, 60],
      weightings: {
        trendAlignment: 0.2,
        volumeProfile: 0.15,
        momentumSync: 0.15,
        timeframeHarmony: 0.1,
        reversalStrength: 0.1,
        supportResistance: 0.1,
        volatilityProfile: 0.1,
        patternSequence: 0.1
      }
    },
    chartSettings: {
      candleColors: {
        bullish: '',
        bearish: '',
        neutral: ''
      },
      wickColors: {
        bullish: '',
        bearish: ''
      },
      backgroundColors: {
        main: '',
        alt: ''
      },
      gridLines: {
        show: true,
        color: '',
        opacity: 0.5
      },
      annotations: {
        showPriceLabels: true,
        showVolume: true,
        showPatternLabels: true,
        fontSize: 12
      }
    }
  };

  // Pattern Variations
  private static readonly patternVariations: { [key: string]: PatternVariation[] } = {
    'Double Bottom': [
      {
        name: 'Classic Double Bottom',
        criteria: (data) => {
          const lows = data.map(d => d.low);
          return Math.abs(lows[0] - lows[lows.length - 1]) / lows[0] < 0.02;
        },
        confidence: 85
      },
      {
        name: 'Complex Double Bottom',
        criteria: (data) => {
          const lows = data.map(d => d.low);
          return lows.filter((v, i) => i > 0 && Math.abs(v - lows[0]) / lows[0] < 0.03).length >= 2;
        },
        confidence: 75
      },
      {
        name: 'Extended Double Bottom',
        criteria: (data) => {
          const lows = data.map(d => d.low);
          return Math.abs(lows[0] - lows[lows.length - 1]) / lows[0] < 0.02 && 
                 data.length > 40;
        },
        confidence: 80
      }
    ],
    'Head and Shoulders': [
      {
        name: 'Perfect Head and Shoulders',
        criteria: (data) => {
          const highs = data.map(d => d.high);
          const peaks = this.findPeaks(highs);
          return peaks.length === 3 && peaks[1] > peaks[0] && peaks[1] > peaks[2];
        },
        confidence: 90
      },
      {
        name: 'Complex Head and Shoulders',
        criteria: (data) => {
          const highs = data.map(d => d.high);
          const peaks = this.findPeaks(highs);
          return peaks.length >= 5;
        },
        confidence: 70
      }
    ]
  };

  // Detect all patterns in a given price series
  static detectPatterns(data: CandleData[]): Pattern[] {
    if (data.length < 20) return [];

    const patterns: Pattern[] = [];

    // Standard pattern detection
    const detectionFunctions = [
      this.detectDoubleBottom,
      this.detectDoubleTop,
      this.detectHeadAndShoulders,
      this.detectInverseHeadAndShoulders,
      this.detectBullishEngulfing,
      this.detectBearishEngulfing,
      this.detectBullishFlag,
      this.detectBearishFlag,
      this.detectTrianglePattern,
      this.detectWedgePattern,
      this.detectCupAndHandle,
      this.detectRoundingBottom,
      this.detectTripleTop,
      this.detectTripleBottom,
      this.detectRisingWedge,
      this.detectFallingWedge,
      this.detectRectanglePattern,
      this.detectDiamondPattern,
      this.detectThreeWhiteSoldiers,
      this.detectThreeBlackCrows,
      this.detectMorningStar,
      this.detectEveningStar,
      this.detectHarami,
      this.detectPiercingLine,
      this.detectDarkCloudCover,
      this.detectGapPattern,
      this.detectIslandReversal,
      this.detectKicker,
      this.detectTasukiGap
    ];

    // Run all standard pattern detections
    detectionFunctions.forEach(detectFn => {
      const pattern = detectFn(data);
      if (pattern) {
        const backtestResults = this.backtest(data, pattern);
        patterns.push({ ...pattern, performance: backtestResults });
      }
    });

    // Run custom pattern detections
    this.customPatterns.forEach(customPattern => {
      const pattern = this.detectCustomPattern(data, customPattern);
      if (pattern) {
        const backtestResults = this.backtest(data, pattern);
        patterns.push({ ...pattern, performance: backtestResults });
      }
    });

    // Apply pattern variations
    const variations = this.detectPatternVariations(data);
    const combinations = this.analyzePatternCombinations(patterns, data);
    
    // Apply user preferences
    return this.applyUserPreferences([...patterns, ...variations, ...combinations]);
  }

  // Double Bottom Pattern
  private static detectDoubleBottom(data: CandleData[]): Pattern | null {
    const last20 = data.slice(-20);
    const lows = last20.map(d => d.low);
    const firstBottom = Math.min(...lows.slice(0, 10));
    const secondBottom = Math.min(...lows.slice(10));
    
    if (Math.abs(firstBottom - secondBottom) / firstBottom < this.THRESHOLD) {
      const confidence = this.calculateConfidence(firstBottom, secondBottom);
      return {
        type: 'bullish',
        name: 'Double Bottom',
        description: 'A bullish reversal pattern indicating potential trend change',
        confidence
      };
    }
    return null;
  }

  // Head and Shoulders Pattern
  private static detectHeadAndShoulders(data: CandleData[]): Pattern | null {
    const last30 = data.slice(-30);
    const highs = last30.map(d => d.high);
    
    // Find three peaks
    const peaks = this.findPeaks(highs);
    if (peaks.length >= 3) {
      const [leftShoulder, head, rightShoulder] = peaks;
      if (head > leftShoulder && head > rightShoulder &&
          Math.abs(leftShoulder - rightShoulder) / leftShoulder < this.THRESHOLD) {
        return {
          type: 'bearish',
          name: 'Head and Shoulders',
          description: 'A bearish reversal pattern suggesting trend exhaustion',
          confidence: this.calculateConfidence(head, (leftShoulder + rightShoulder) / 2)
        };
      }
    }
    return null;
  }

  // Bullish Flag Pattern
  private static detectBullishFlag(data: CandleData[]): Pattern | null {
    const last15 = data.slice(-15);
    const trend = this.calculateTrend(last15.map(d => d.close));
    const consolidation = this.calculateConsolidation(last15.map(d => d.close));

    if (trend > 0 && consolidation < this.THRESHOLD) {
      return {
        type: 'bullish',
        name: 'Bullish Flag',
        description: 'A continuation pattern suggesting further upward movement',
        confidence: Math.min(trend * 100, 95)
      };
    }
    return null;
  }

  // Triangle Pattern
  private static detectTrianglePattern(data: CandleData[]): Pattern | null {
    const last20 = data.slice(-20);
    const highs = last20.map(d => d.high);
    const lows = last20.map(d => d.low);

    const highTrend = this.calculateTrend(highs);
    const lowTrend = this.calculateTrend(lows);

    if (Math.abs(highTrend) < this.THRESHOLD && Math.abs(lowTrend) < this.THRESHOLD) {
      const type = highTrend > lowTrend ? 'bullish' : 'bearish';
      return {
        type,
        name: `${type === 'bullish' ? 'Ascending' : 'Descending'} Triangle`,
        description: `A ${type} continuation pattern showing price convergence`,
        confidence: Math.abs(highTrend - lowTrend) * 100
      };
    }
    return null;
  }

  // Wedge Pattern
  private static detectWedgePattern(data: CandleData[]): Pattern | null {
    const last25 = data.slice(-25);
    const highs = last25.map(d => d.high);
    const lows = last25.map(d => d.low);

    const highTrend = this.calculateTrend(highs);
    const lowTrend = this.calculateTrend(lows);

    if (Math.sign(highTrend) === Math.sign(lowTrend) && 
        Math.abs(highTrend - lowTrend) > this.THRESHOLD) {
      const type = highTrend < 0 ? 'bullish' : 'bearish';
      return {
        type,
        name: `${type === 'bullish' ? 'Falling' : 'Rising'} Wedge`,
        description: `A ${type} reversal pattern showing convergence with trend`,
        confidence: Math.abs(highTrend - lowTrend) * 100
      };
    }
    return null;
  }

  // New Pattern Detection Methods

  private static detectCupAndHandle(data: CandleData[]): Pattern | null {
    const last50 = data.slice(-50);
    const prices = last50.map(d => d.close);
    
    // Cup detection (U-shaped pattern)
    const cupDepth = Math.max(...prices) - Math.min(...prices);
    const cupWidth = 30; // Typical cup width
    
    if (this.isUShape(prices.slice(-cupWidth))) {
      const confidence = this.calculateCupConfidence(prices);
      return {
        type: 'bullish',
        name: 'Cup and Handle',
        description: 'Bullish continuation pattern resembling a cup with handle',
        confidence
      };
    }
    return null;
  }

  private static detectRoundingBottom(data: CandleData[]): Pattern | null {
    const last40 = data.slice(-40);
    const prices = last40.map(d => d.close);
    
    if (this.isRoundingPattern(prices)) {
      return {
        type: 'bullish',
        name: 'Rounding Bottom',
        description: 'Long-term reversal pattern showing gradual shift in sentiment',
        confidence: this.calculateRoundingConfidence(prices)
      };
    }
    return null;
  }

  private static detectTripleTop(data: CandleData[]): Pattern | null {
    const last60 = data.slice(-60);
    const highs = last60.map(d => d.high);
    const peaks = this.findPeaks(highs);
    
    if (peaks.length >= 3 && this.areEqualPeaks(peaks.slice(0, 3))) {
      return {
        type: 'bearish',
        name: 'Triple Top',
        description: 'Bearish reversal pattern with three equal peaks',
        confidence: this.calculateTriplePeakConfidence(peaks)
      };
    }
    return null;
  }

  private static detectThreeWhiteSoldiers(data: CandleData[]): Pattern | null {
    const last3 = data.slice(-3);
    if (last3.every((candle, i) => 
      i === 0 || (
        candle.close > candle.open && // Bullish candle
        candle.close > last3[i-1].close && // Higher close
        candle.open > last3[i-1].open // Higher open
      )
    )) {
      return {
        type: 'bullish',
        name: 'Three White Soldiers',
        description: 'Three consecutive bullish candles with higher highs and higher lows',
        confidence: this.calculateConfidence(last3[0].close, last3[2].close)
      };
    }
    return null;
  }

  private static detectThreeBlackCrows(data: CandleData[]): Pattern | null {
    const last3 = data.slice(-3);
    if (last3.every((candle, i) => 
      i === 0 || (
        candle.close < candle.open && // Bearish candle
        candle.close < last3[i-1].close && // Lower close
        candle.open < last3[i-1].open // Lower open
      )
    )) {
      return {
        type: 'bearish',
        name: 'Three Black Crows',
        description: 'Three consecutive bearish candles with lower highs and lower lows',
        confidence: this.calculateConfidence(last3[0].close, last3[2].close)
      };
    }
    return null;
  }

  private static detectMorningStar(data: CandleData[]): Pattern | null {
    const last3 = data.slice(-3);
    if (
      last3[0].close < last3[0].open && // First day bearish
      Math.abs(last3[1].open - last3[1].close) < (last3[1].high - last3[1].low) * 0.3 && // Small body
      last3[2].close > last3[2].open && // Third day bullish
      last3[2].close > last3[0].close // Closes above first day
    ) {
      return {
        type: 'bullish',
        name: 'Morning Star',
        description: 'A three-candle reversal pattern signaling a potential bottom',
        confidence: 85
      };
    }
    return null;
  }

  private static detectEveningStar(data: CandleData[]): Pattern | null {
    const last3 = data.slice(-3);
    if (
      last3[0].close > last3[0].open && // First day bullish
      Math.abs(last3[1].close - last3[1].open) < (last3[1].high - last3[1].low) * 0.3 && // Small body
      last3[2].close < last3[2].open && // Third day bearish
      last3[2].close < last3[0].close // Closes below first day
    ) {
      return {
        type: 'bearish',
        name: 'Evening Star',
        description: 'A three-candle reversal pattern signaling a potential top',
        confidence: 85
      };
    }
    return null;
  }

  private static detectHarami(data: CandleData[]): Pattern | null {
    const last2 = data.slice(-2);
    if (
      last2[0].close > last2[0].open && // First day bullish
      last2[1].close < last2[1].open && // Second day bearish
      last2[1].close < last2[0].close // Second day closes below first day
    ) {
      return {
        type: 'bearish',
        name: 'Harami',
        description: 'A bearish reversal pattern indicating a potential trend change',
        confidence: this.calculateConfidence(last2[0].close, last2[1].close)
      };
    }
    return null;
  }

  private static detectPiercingLine(data: CandleData[]): Pattern | null {
    const last2 = data.slice(-2);
    if (
      last2[0].close > last2[0].open && // First day bullish
      last2[1].close < last2[1].open && // Second day bearish
      last2[1].close < last2[0].close && // Second day closes below first day
      last2[1].high > last2[0].low // Second day high is above first day low
    ) {
      return {
        type: 'bullish',
        name: 'Piercing Line',
        description: 'A bullish reversal pattern indicating a potential trend change',
        confidence: this.calculateConfidence(last2[0].close, last2[1].close)
      };
    }
    return null;
  }

  private static detectDarkCloudCover(data: CandleData[]): Pattern | null {
    const last2 = data.slice(-2);
    if (
      last2[0].close > last2[0].open && // First day bullish
      last2[1].close < last2[1].open && // Second day bearish
      last2[1].close < last2[0].close && // Second day closes below first day
      last2[1].high < last2[0].low // Second day high is below first day low
    ) {
      return {
        type: 'bearish',
        name: 'Dark Cloud Cover',
        description: 'A bearish reversal pattern indicating a potential trend change',
        confidence: this.calculateConfidence(last2[0].close, last2[1].close)
      };
    }
    return null;
  }

  private static detectGapPattern(data: CandleData[]): Pattern | null {
    const last2 = data.slice(-2);
    if (
      last2[0].close > last2[0].open && // First day bullish
      last2[1].close < last2[1].open && // Second day bearish
      last2[1].close < last2[0].close && // Second day closes below first day
      last2[1].high < last2[0].low // Second day high is below first day low
    ) {
      return {
        type: 'bearish',
        name: 'Gap Pattern',
        description: 'A bearish reversal pattern indicating a potential trend change',
        confidence: this.calculateConfidence(last2[0].close, last2[1].close)
      };
    }
    return null;
  }

  private static detectIslandReversal(data: CandleData[]): Pattern | null {
    const last5 = data.slice(-5);
    // Check for gaps on both sides
    const hasGapDown = last5[1].high < last5[0].low;
    const hasGapUp = last5[3].low > last5[2].high;
    
    if (hasGapDown && hasGapUp) {
      return {
        type: 'bullish',
        name: 'Island Reversal',
        description: 'Price gaps down and then gaps up, forming an isolated price range',
        confidence: 80
      };
    }
    return null;
  }

  private static detectKicker(data: CandleData[]): Pattern | null {
    const last2 = data.slice(-2);
    if (
      last2[0].close > last2[0].open && // First day bullish
      last2[1].close < last2[1].open && // Second day bearish
      last2[1].close < last2[0].close && // Second day closes below first day
      last2[1].high < last2[0].low // Second day high is below first day low
    ) {
      return {
        type: 'bearish',
        name: 'Kicker',
        description: 'A bearish reversal pattern indicating a potential trend change',
        confidence: this.calculateConfidence(last2[0].close, last2[1].close)
      };
    }
    return null;
  }

  private static detectTasukiGap(data: CandleData[]): Pattern | null {
    const last2 = data.slice(-2);
    if (
      last2[0].close > last2[0].open && // First day bullish
      last2[1].close < last2[1].open && // Second day bearish
      last2[1].close < last2[0].close && // Second day closes below first day
      last2[1].high < last2[0].low // Second day high is below first day low
    ) {
      return {
        type: 'bearish',
        name: 'Tasuki Gap',
        description: 'A bearish reversal pattern indicating a potential trend change',
        confidence: this.calculateConfidence(last2[0].close, last2[1].close)
      };
    }
    return null;
  }

  // Backtesting Functionality

  private static backtest(data: CandleData[], pattern: Pattern) {
    const trades = this.simulateTrades(data, pattern);
    const winningTrades = trades.filter(t => t.profit > 0);
    const losingTrades = trades.filter(t => t.profit < 0);
    
    // Basic metrics
    const winRate = winningTrades.length / trades.length;
    const avgProfit = trades.reduce((sum, t) => sum + Math.max(0, t.profit), 0) / trades.length;
    const avgLoss = Math.abs(trades.reduce((sum, t) => sum + Math.min(0, t.profit), 0)) / trades.length;
    
    // Advanced metrics
    const maxDrawdown = this.calculateMaxDrawdown(trades);
    const profitFactor = this.calculateProfitFactor(trades);
    const sharpeRatio = this.calculateSharpeRatio(trades);
    const recoveryFactor = this.calculateRecoveryFactor(trades, maxDrawdown);
    const { consecutiveWins, consecutiveLosses } = this.calculateConsecutiveStats(trades);
    const { largestWin, largestLoss } = this.calculateExtremeReturns(trades);
    const averageHoldingPeriod = this.calculateAverageHoldingPeriod(trades);
    const winLossRatio = avgProfit / (avgLoss || 1);
    
    return {
      winRate,
      avgProfit,
      avgLoss,
      riskRewardRatio: avgProfit / (avgLoss || 1),
      maxDrawdown,
      profitFactor,
      sharpeRatio,
      recoveryFactor,
      consecutiveWins,
      consecutiveLosses,
      largestWin,
      largestLoss,
      averageHoldingPeriod,
      winLossRatio
    };
  }

  private static simulateTrades(data: CandleData[], pattern: Pattern) {
    const trades = [];
    let inPosition = false;
    let entryPrice = 0;
    
    for (let i = 0; i < data.length - 1; i++) {
      if (!inPosition && this.isPatternComplete(data.slice(0, i + 1), pattern)) {
        inPosition = true;
        entryPrice = data[i + 1].open;
      } else if (inPosition) {
        const exitPrice = data[i + 1].close;
        const profit = pattern.type === 'bullish' 
          ? (exitPrice - entryPrice) / entryPrice
          : (entryPrice - exitPrice) / entryPrice;
        
        trades.push({ profit });
        inPosition = false;
      }
    }
    
    return trades;
  }

  // Custom Pattern Support

  static addCustomPattern(pattern: CustomPattern) {
    this.customPatterns.push(pattern);
  }

  static removeCustomPattern(name: string) {
    this.customPatterns = this.customPatterns.filter(p => p.name !== name);
  }

  private static detectCustomPattern(data: CandleData[], customPattern: CustomPattern): Pattern | null {
    const prices = data.map(d => d.close);
    const normalizedPrices = this.normalizePrices(prices);
    const normalizedPattern = this.normalizePrices(customPattern.points.map(p => p.y));
    
    const similarity = this.calculatePatternSimilarity(normalizedPrices, normalizedPattern);
    
    if (similarity >= 1 - customPattern.tolerance) {
      return {
        type: customPattern.type,
        name: customPattern.name,
        description: customPattern.description,
        confidence: similarity * 100
      };
    }
    return null;
  }

  // Helper Methods

  private static isUShape(prices: number[]): boolean {
    const first = prices[0];
    const middle = prices[Math.floor(prices.length / 2)];
    const last = prices[prices.length - 1];
    
    return first > middle && last > middle;
  }

  private static isRoundingPattern(prices: number[]): boolean {
    const trendChanges = this.calculateTrendChanges(prices);
    return trendChanges.length >= 3 && 
           trendChanges[0] < 0 && 
           trendChanges[trendChanges.length - 1] > 0;
  }

  private static areEqualPeaks(peaks: number[]): boolean {
    const mean = peaks.reduce((a, b) => a + b, 0) / peaks.length;
    return peaks.every(p => Math.abs(p - mean) / mean < this.THRESHOLD);
  }

  private static normalizePrices(prices: number[]): number[] {
    const min = Math.min(...prices);
    const max = Math.max(...prices);
    return prices.map(p => (p - min) / (max - min));
  }

  private static calculatePatternSimilarity(prices1: number[], prices2: number[]): number {
    // Dynamic Time Warping algorithm for pattern matching
    const dtw = Array(prices1.length + 1).fill(null)
      .map(() => Array(prices2.length + 1).fill(Infinity));
    
    dtw[0][0] = 0;
    
    for (let i = 1; i <= prices1.length; i++) {
      for (let j = 1; j <= prices2.length; j++) {
        const cost = Math.abs(prices1[i - 1] - prices2[j - 1]);
        dtw[i][j] = cost + Math.min(dtw[i - 1][j], dtw[i][j - 1], dtw[i - 1][j - 1]);
      }
    }
    
    return 1 / (1 + dtw[prices1.length][prices2.length]);
  }

  private static calculateTrendChanges(prices: number[]): number[] {
    const changes = [];
    for (let i = 1; i < prices.length; i++) {
      changes.push(prices[i] - prices[i - 1]);
    }
    return changes;
  }

  private static calculateTrend(prices: number[]): number {
    const x = Array.from({ length: prices.length }, (_, i) => i);
    const n = prices.length;
    
    const sumX = x.reduce((a, b) => a + b, 0);
    const sumY = prices.reduce((a, b) => a + b, 0);
    const sumXY = x.reduce((acc, curr, i) => acc + curr * prices[i], 0);
    const sumX2 = x.reduce((a, b) => a + b * b, 0);
    
    return (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX);
  }

  private static calculateConsolidation(prices: number[]): number {
    const mean = prices.reduce((a, b) => a + b, 0) / prices.length;
    const variance = prices.reduce((acc, price) => acc + Math.pow(price - mean, 2), 0) / prices.length;
    return Math.sqrt(variance) / mean;
  }

  private static findPeaks(data: number[]): number[] {
    const peaks: number[] = [];
    for (let i = 1; i < data.length - 1; i++) {
      if (data[i] > data[i - 1] && data[i] > data[i + 1]) {
        peaks.push(data[i]);
      }
    }
    return peaks;
  }

  private static calculateConfidence(value1: number, value2: number): number {
    const diff = Math.abs(value1 - value2);
    const avg = (value1 + value2) / 2;
    return Math.max(0, Math.min(100, 100 * (1 - diff / avg)));
  }

  private static calculateMaxDrawdown(trades: any[]): number {
    let peak = 0;
    let maxDrawdown = 0;
    let runningTotal = 0;

    trades.forEach(trade => {
      runningTotal += trade.profit;
      if (runningTotal > peak) peak = runningTotal;
      const drawdown = peak - runningTotal;
      if (drawdown > maxDrawdown) maxDrawdown = drawdown;
    });

    return maxDrawdown;
  }

  private static calculateProfitFactor(trades: any[]): number {
    const grossProfit = trades.reduce((sum, t) => sum + Math.max(0, t.profit), 0);
    const grossLoss = Math.abs(trades.reduce((sum, t) => sum + Math.min(0, t.profit), 0));
    return grossLoss === 0 ? grossProfit : grossProfit / grossLoss;
  }

  private static calculateSharpeRatio(trades: any[]): number {
    const returns = trades.map(t => t.profit);
    const avgReturn = returns.reduce((a, b) => a + b, 0) / returns.length;
    const stdDev = Math.sqrt(
      returns.reduce((sum, r) => sum + Math.pow(r - avgReturn, 2), 0) / returns.length
    );
    return stdDev === 0 ? 0 : avgReturn / stdDev;
  }

  private static calculateRecoveryFactor(trades: any[], maxDrawdown: number): number {
    const netProfit = trades.reduce((sum, t) => sum + t.profit, 0);
    return maxDrawdown === 0 ? netProfit : netProfit / maxDrawdown;
  }

  private static calculateConsecutiveStats(trades: any[]) {
    let currentWins = 0;
    let currentLosses = 0;
    let maxWins = 0;
    let maxLosses = 0;

    trades.forEach(trade => {
      if (trade.profit > 0) {
        currentWins++;
        currentLosses = 0;
        maxWins = Math.max(maxWins, currentWins);
      } else {
        currentLosses++;
        currentWins = 0;
        maxLosses = Math.max(maxLosses, currentLosses);
      }
    });

    return { consecutiveWins: maxWins, consecutiveLosses: maxLosses };
  }

  private static calculateExtremeReturns(trades: any[]) {
    return {
      largestWin: Math.max(...trades.map(t => t.profit)),
      largestLoss: Math.min(...trades.map(t => t.profit))
    };
  }

  private static calculateAverageHoldingPeriod(trades: any[]): number {
    return trades.reduce((sum, t) => sum + t.holdingPeriod, 0) / trades.length;
  }

  // Pattern Export/Import

  static exportPatterns(): string {
    const patterns = {
      custom: this.customPatterns,
      config: {
        threshold: this.THRESHOLD
      }
    };
    return btoa(JSON.stringify(patterns));
  }

  static importPatterns(encodedPatterns: string): boolean {
    try {
      const patterns = JSON.parse(atob(encodedPatterns));
      if (patterns.custom && Array.isArray(patterns.custom)) {
        this.customPatterns = patterns.custom;
        return true;
      }
      return false;
    } catch (e) {
      console.error('Failed to import patterns:', e);
      return false;
    }
  }

  // Add new methods for pattern combination analysis
  private static analyzePatternCombinations(patterns: Pattern[], data: CandleData[]): PatternCombination[] {
    const combinations: PatternCombination[] = [];
    const prefs = this.userPreferences.combinationPreferences;
    
    for (let i = 0; i < patterns.length; i++) {
      for (let j = i + 1; j < patterns.length; j++) {
        const metrics = this.calculateCombinationMetrics(patterns[i], patterns[j], data);
        const synergy = this.calculateWeightedSynergy(metrics, prefs.weightings);
        const reliability = this.calculateCombinationReliability([patterns[i], patterns[j]]);
        
        if (synergy >= prefs.minSynergy && reliability >= prefs.minReliability) {
          combinations.push({
            patterns: [patterns[i], patterns[j]],
            synergy,
            timeframe: Math.max(
              this.getPatternTimeframe(patterns[i]),
              this.getPatternTimeframe(patterns[j])
            ),
            reliability,
            metrics
          });
        }
      }
    }
    
    return combinations.sort((a, b) => b.synergy - a.synergy);
  }

  private static calculateCombinationMetrics(p1: Pattern, p2: Pattern, data: CandleData[]): CombinationMetrics {
    const trendAlignment = this.calculateTrendAlignment(p1, p2, data);
    const volumeProfile = this.analyzeVolumeProfile(data);
    const momentumSync = this.calculateMomentumSync(p1, p2, data);
    const timeframeHarmony = this.calculateTimeframeHarmony(p1, p2);
    const reversalStrength = this.calculateReversalStrength(p1, p2, data);
    const supportResistance = this.analyzeSupportResistance(data);
    const volatilityProfile = this.analyzeVolatilityProfile(data);
    const patternSequence = this.analyzePatternSequence(p1, p2);
    
    return {
      trendAlignment,
      volumeProfile,
      momentumSync,
      timeframeHarmony,
      reversalStrength,
      supportResistance,
      volatilityProfile,
      patternSequence
    };
  }

  private static calculateTrendAlignment(p1: Pattern, p2: Pattern, data: CandleData[]): number {
    const trend = this.calculateTrend(data.map(d => d.close));
    const p1Alignment = p1.type === 'bullish' ? trend > 0 ? 1 : 0 : trend < 0 ? 1 : 0;
    const p2Alignment = p2.type === 'bullish' ? trend > 0 ? 1 : 0 : trend < 0 ? 1 : 0;
    return (p1Alignment + p2Alignment) / 2;
  }

  private static calculateMomentumSync(p1: Pattern, p2: Pattern, data: CandleData[]): number {
    const momentum = this.calculateMomentum(data);
    const p1Momentum = p1.type === 'bullish' ? momentum : -momentum;
    const p2Momentum = p2.type === 'bullish' ? momentum : -momentum;
    return 1 - Math.abs(p1Momentum - p2Momentum);
  }

  private static calculateTimeframeHarmony(p1: Pattern, p2: Pattern): number {
    const t1 = this.getPatternTimeframe(p1);
    const t2 = this.getPatternTimeframe(p2);
    return 1 - Math.abs(t1 - t2) / Math.max(t1, t2);
  }

  private static calculateReversalStrength(p1: Pattern, p2: Pattern, data: CandleData[]): number {
    const trend = this.calculateTrend(data.map(d => d.close));
    const isReversal = (p1.type !== p2.type) || 
                      (p1.type === 'bullish' && trend < 0) || 
                      (p1.type === 'bearish' && trend > 0);
    return isReversal ? Math.abs(trend) : 0;
  }

  private static analyzeSupportResistance(data: CandleData[]): number {
    const prices = data.map(d => d.close);
    const levels = this.findSupportResistanceLevels(prices);
    const currentPrice = prices[prices.length - 1];
    
    return levels.reduce((closest, level) => {
      const distance = Math.abs(currentPrice - level) / currentPrice;
      return Math.min(closest, distance);
    }, 1);
  }

  private static analyzeVolatilityProfile(data: CandleData[]): number {
    const volatility = this.calculateVolatility(data);
    const optimalVolatility = 0.02; // 2% optimal volatility
    return 1 - Math.abs(volatility - optimalVolatility) / optimalVolatility;
  }

  private static analyzePatternSequence(p1: Pattern, p2: Pattern): number {
    // Analyze if patterns form a logical sequence
    const validSequences = [
      ['Double Bottom', 'Bullish Flag'],
      ['Head and Shoulders', 'Bearish Flag'],
      ['Cup and Handle', 'Bullish Breakout'],
      // Add more valid sequences
    ];
    
    return validSequences.some(([first, second]) => 
      (p1.name === first && p2.name === second) || 
      (p2.name === first && p1.name === second)
    ) ? 1 : 0;
  }

  private static findSupportResistanceLevels(prices: number[]): number[] {
    const levels: number[] = [];
    const window = 20;
    
    for (let i = window; i < prices.length - window; i++) {
      const current = prices[i];
      const before = prices.slice(i - window, i);
      const after = prices.slice(i + 1, i + window + 1);
      
      if (Math.min(...before) === current && Math.min(...after) === current) {
        levels.push(current); // Support level
      }
      if (Math.max(...before) === current && Math.max(...after) === current) {
        levels.push(current); // Resistance level
      }
    }
    
    return [...new Set(levels)]; // Remove duplicates
  }

  private static calculateWeightedSynergy(metrics: CombinationMetrics, weights: UserPreferences['combinationPreferences']['weightings']): number {
    return Object.entries(metrics).reduce((sum, [key, value]) => {
      return sum + value * weights[key as keyof typeof weights];
    }, 0);
  }

  // User Preferences Management
  static setUserPreferences(prefs: Partial<UserPreferences>) {
    this.userPreferences = {
      ...this.userPreferences,
      ...prefs
    };
  }

  static getUserPreferences(): UserPreferences {
    return { ...this.userPreferences };
  }

  // Enhanced pattern detection with variations
  private static detectPatternVariations(data: CandleData[]): Pattern[] {
    const variations: Pattern[] = [];
    
    Object.entries(this.patternVariations).forEach(([basePattern, vars]) => {
      vars.forEach(variation => {
        if (variation.criteria(data)) {
          variations.push({
            type: this.getPatternType(basePattern),
            name: variation.name,
            description: `${variation.name} - A variation of ${basePattern}`,
            confidence: variation.confidence
          });
        }
      });
    });
    
    return variations;
  }

  private static applyUserPreferences(patterns: Pattern[]): Pattern[] {
    const { favoritePatterns, customThresholds, riskTolerance } = this.userPreferences;
    
    return patterns
      .filter(p => {
        const threshold = customThresholds[p.name] || this.THRESHOLD;
        return p.confidence >= threshold;
      })
      .map(p => ({
        ...p,
        confidence: favoritePatterns.includes(p.name) ? p.confidence * 1.1 : p.confidence,
        performance: p.performance ? {
          ...p.performance,
          riskRewardRatio: p.performance.riskRewardRatio * (1 + (1 - riskTolerance))
        } : undefined
      }));
  }

  // Helper methods for forecasting
  private static analyzeTechnicalFactors(data: CandleData[]): number {
    const trend = this.calculateTrend(data.map(d => d.close));
    const momentum = this.calculateMomentum(data);
    const volatility = this.calculateVolatility(data);
    
    return (Math.abs(trend) * 0.4 + momentum * 0.3 + (1 - volatility) * 0.3);
  }

  private static analyzeVolumeProfile(data: CandleData[]): number {
    const volumeChange = data[data.length - 1].volume / data[0].volume;
    const volumeTrend = this.calculateTrend(data.map(d => d.volume));
    
    return (volumeChange > 1 ? 0.7 : 0.3) + Math.abs(volumeTrend) * 0.3;
  }

  private static analyzeSentimentImpact(data: CandleData[]): number {
    // Implement sentiment analysis based on price action and volume
    const priceAction = this.calculatePriceAction(data);
    const volumeProfile = this.analyzeVolumeProfile(data);
    
    return (priceAction * 0.6 + volumeProfile * 0.4);
  }

  private static analyzeHistoricalSuccess(pattern: Pattern): number {
    if (!pattern.performance) return 0.5;
    
    const { winRate, profitFactor, recoveryFactor } = pattern.performance;
    return (winRate * 0.4 + (profitFactor / 3) * 0.3 + (recoveryFactor / 2) * 0.3);
  }

  private static calculateExpectedReturn(pattern: Pattern, probability: number): number {
    if (!pattern.performance) return 0;
    
    const { avgProfit, avgLoss } = pattern.performance;
    return (avgProfit * probability) - (avgLoss * (1 - probability));
  }

  private static getPatternTimeframe(pattern: Pattern): number {
    // Estimate pattern completion timeframe based on type
    const timeframes: { [key: string]: number } = {
      'Double Bottom': 20,
      'Head and Shoulders': 30,
      'Cup and Handle': 50,
      'Triangle': 15,
      'Flag': 10
    };
    
    return timeframes[pattern.name] || 15;
  }

  private static calculatePerformanceSync(p1: Pattern, p2: Pattern): number {
    if (!p1.performance || !p2.performance) return 0.5;
    
    const winRateSync = 1 - Math.abs(p1.performance.winRate - p2.performance.winRate);
    const profitSync = 1 - Math.abs(p1.performance.avgProfit - p2.performance.avgProfit);
    
    return (winRateSync * 0.6 + profitSync * 0.4);
  }

  private static calculateMomentum(data: CandleData[]): number {
    const closes = data.map(d => d.close);
    const roc = (closes[closes.length - 1] - closes[0]) / closes[0];
    return Math.min(Math.abs(roc), 1);
  }

  private static calculateVolatility(data: CandleData[]): number {
    const returns = data.slice(1).map((d, i) => (d.close - data[i].close) / data[i].close);
    const stdDev = Math.sqrt(
      returns.reduce((sum, r) => sum + r * r, 0) / returns.length
    );
    return Math.min(stdDev * 100, 1);
  }

  private static calculatePriceAction(data: CandleData[]): number {
    const trend = this.calculateTrend(data.map(d => d.close));
    const momentum = this.calculateMomentum(data);
    return (Math.abs(trend) * 0.7 + momentum * 0.3);
  }
} 