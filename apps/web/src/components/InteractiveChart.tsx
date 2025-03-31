import React, { useState, useCallback, useEffect, useMemo } from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  Brush,
  ReferenceArea,
  ReferenceLine,
} from 'recharts';
import { useWebSocket } from '@/hooks/useWebSocket';
import { PatternRecognition } from '@/utils/patternRecognition';
import { ChartConfigManager } from '@/utils/chartConfig';

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
    maxDrawdown?: number;
    profitFactor?: number;
    sharpeRatio?: number;
    recoveryFactor?: number;
    consecutiveWins?: number;
    consecutiveLosses?: number;
    largestWin?: number;
    largestLoss?: number;
    averageHoldingPeriod?: number;
  };
  patternCombinations?: PatternCombination[];
  forecast?: {
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
  };
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
  candleFunneling: {
    volumeThreshold: number;
    bodyToWickRatio: number;
    momentumThreshold: number;
    trendStrength: number;
    volatilityFilter: number;
    gapThreshold: number;
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

interface DataPoint {
  date: string;
  actual: number;
  predicted?: number;
  volume?: number;
  ma20?: number;
  ma50?: number;
  ma200?: number;
  rsi?: number;
  macd?: number;
  signal?: number;
  histogram?: number;
  bollingerUpper?: number;
  bollingerLower?: number;
  sentiment?: number;
  patterns?: Pattern[];
  patternCombinations?: PatternCombination[];
  forecast?: {
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
  };
}

interface InteractiveChartProps {
  data: DataPoint[];
  title?: string;
  showPrediction?: boolean;
  showIndicators?: boolean;
  showVolume?: boolean;
  showSentiment?: boolean;
  showPatterns?: boolean;
  height?: number;
  onDataUpdate?: (data: DataPoint[]) => void;
  symbol?: string;
  refreshInterval?: number;
}

interface CombinationMetrics {
  trendAlignment: number;
  volumeProfile: number;
  momentumSync: number;
  timeframeHarmony: number;
  reversalStrength: number;
  supportResistance: number;
  volatilityProfile: number;
  patternSequence: number;
}

interface PatternCombination {
  patterns: Pattern[];
  synergy: number;
  timeframe: number;
  reliability: number;
  metrics: CombinationMetrics;
}

const InteractiveChart: React.FC<InteractiveChartProps> = ({
  data: initialData,
  title = 'Stock Price Analysis',
  showPrediction = true,
  showIndicators = true,
  showVolume = true,
  showSentiment = true,
  showPatterns = true,
  height = 500,
  onDataUpdate,
  symbol,
  refreshInterval = 60000,
}) => {
  const [data, setData] = useState(initialData);
  const [config, setConfig] = useState(ChartConfigManager.getConfig());
  const [zoomState, setZoomState] = useState<{
    refAreaLeft: string | null;
    refAreaRight: string | null;
    left: string | null;
    right: string | null;
  }>({
    refAreaLeft: null,
    refAreaRight: null,
    left: 'dataMin',
    right: 'dataMax',
  });

  const [selectedDataPoint, setSelectedDataPoint] = useState<DataPoint | null>(null);

  const [selectedPattern, setSelectedPattern] = useState<{
    date: string;
    pattern: Pattern;
  } | null>(null);

  const [isDrawingPattern, setIsDrawingPattern] = useState(false);
  const [customPatternPoints, setCustomPatternPoints] = useState<{ x: number; y: number }[]>([]);
  const [showBacktestResults, setShowBacktestResults] = useState(false);

  // Add new state for pattern management
  const [showAdvancedMetrics, setShowAdvancedMetrics] = useState(false);

  const [showPatternCustomization, setShowPatternCustomization] = useState(false);
  const [showCombinationAnalysis, setShowCombinationAnalysis] = useState(false);
  const [showProbabilityForecast, setShowProbabilityForecast] = useState(false);
  const [userPreferences, setUserPreferences] = useState(PatternRecognition.getUserPreferences());

  // WebSocket integration
  const handleWebSocketMessage = useCallback((message: any) => {
    if (message.type === 'marketData') {
      setData(prevData => [...prevData.slice(1), message.data]);
    }
  }, []);

  const { isConnected, error } = useWebSocket(symbol || '', handleWebSocketMessage);

  // Pattern Recognition
  const patterns = useMemo(() => {
    if (!showPatterns || !config.patterns.enabled.length) return [];
    const candleData = data.map(d => ({
      date: d.date,
      open: d.actual,
      high: d.actual,
      low: d.actual,
      close: d.actual,
      volume: d.volume || 0
    }));
    return PatternRecognition.detectPatterns(candleData);
  }, [data, showPatterns, config.patterns.enabled]);

  // Config Management
  const handleConfigChange = useCallback((newConfig: Partial<typeof config>) => {
    setConfig(prev => {
      const updated = { ...prev, ...newConfig };
      ChartConfigManager.saveConfig(updated);
      return updated;
    });
  }, []);

  const handleExportConfig = useCallback(() => {
    const encodedConfig = ChartConfigManager.exportConfig();
    const link = ChartConfigManager.getShareableLink(config);
    
    // Create temporary input for copying
    const input = document.createElement('input');
    input.value = link;
    document.body.appendChild(input);
    input.select();
    document.execCommand('copy');
    document.body.removeChild(input);

    alert('Configuration link copied to clipboard!');
  }, [config]);

  const handleImportConfig = useCallback((encodedConfig: string) => {
    if (ChartConfigManager.importConfig(encodedConfig)) {
      setConfig(ChartConfigManager.getConfig());
    } else {
      alert('Invalid configuration format');
    }
  }, []);

  // Load config from URL if available
  useEffect(() => {
    const loadUrlConfig = async () => {
      const urlConfig = await ChartConfigManager.loadConfigFromUrl();
      if (urlConfig) {
        setConfig(urlConfig);
      }
    };
    loadUrlConfig();
  }, []);

  const handleZoomStart = useCallback((e: any) => {
    if (e) {
      setZoomState((prev) => ({ ...prev, refAreaLeft: e.activeLabel }));
    }
  }, []);

  const handleZoomMove = useCallback(
    (e: any) => {
      if (zoomState.refAreaLeft && e) {
        setZoomState((prev) => ({ ...prev, refAreaRight: e.activeLabel }));
      }
    },
    [zoomState.refAreaLeft]
  );

  const handleZoomEnd = useCallback(() => {
    if (zoomState.refAreaLeft && zoomState.refAreaRight) {
      let left = zoomState.refAreaLeft;
      let right = zoomState.refAreaRight;

      if (left > right) {
        [left, right] = [right, left];
      }

      setZoomState({
        refAreaLeft: null,
        refAreaRight: null,
        left,
        right,
      });
    }
  }, [zoomState.refAreaLeft, zoomState.refAreaRight]);

  const handleZoomOut = useCallback(() => {
    setZoomState({
      refAreaLeft: null,
      refAreaRight: null,
      left: 'dataMin',
      right: 'dataMax',
    });
  }, []);

  // Pattern Recognition Markers
  const renderPatternMarkers = () => {
    return data.map((point, index) => {
      if (!point.patterns?.length) return null;

      return point.patterns.map((pattern: Pattern, pIndex: number) => (
        <ReferenceLine
          key={`${index}-${pIndex}`}
          x={point.date}
          stroke={pattern.type === 'bullish' ? '#22C55E' : '#EF4444'}
          strokeDasharray="3 3"
          onClick={() => {
            if (point.patterns?.[pIndex]) {
              setSelectedPattern({ 
                date: point.date, 
                pattern: point.patterns[pIndex] 
              });
            }
          }}
        />
      ));
    });
  };

  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload;
      return (
        <div className="rounded-lg border border-gray-200 bg-white p-4 shadow-sm dark:border-gray-800 dark:bg-gray-900">
          <p className="mb-2 font-medium">{label}</p>
          <div className="space-y-1">
            <p className="text-sm text-gray-600 dark:text-gray-400">
              Price: ${data.actual.toLocaleString()}
            </p>
            {showPrediction && data.predicted && (
              <p className="text-sm text-blue-600 dark:text-blue-400">
                Predicted: ${data.predicted.toLocaleString()}
              </p>
            )}
            {showIndicators && (
              <>
                {data.ma20 && (
                  <p className="text-sm text-green-600 dark:text-green-400">
                    MA20: ${data.ma20.toLocaleString()}
                  </p>
                )}
                {data.ma50 && (
                  <p className="text-sm text-purple-600 dark:text-purple-400">
                    MA50: ${data.ma50.toLocaleString()}
                  </p>
                )}
                {data.ma200 && (
                  <p className="text-sm text-yellow-600 dark:text-yellow-400">
                    MA200: ${data.ma200.toLocaleString()}
                  </p>
                )}
                {data.rsi && (
                  <p className="text-sm text-orange-600 dark:text-orange-400">
                    RSI: {data.rsi.toFixed(2)}
                  </p>
                )}
                {data.macd && (
                  <p className="text-sm text-indigo-600 dark:text-indigo-400">
                    MACD: {data.macd.toFixed(2)}
                  </p>
                )}
                {data.bollingerUpper && (
                  <p className="text-sm text-pink-600 dark:text-pink-400">
                    Bollinger Upper: ${data.bollingerUpper.toLocaleString()}
                  </p>
                )}
                {data.bollingerLower && (
                  <p className="text-sm text-pink-600 dark:text-pink-400">
                    Bollinger Lower: ${data.bollingerLower.toLocaleString()}
                  </p>
                )}
              </>
            )}
            {showVolume && data.volume && (
              <p className="text-sm text-gray-600 dark:text-gray-400">
                Volume: {data.volume.toLocaleString()}
              </p>
            )}
            {showSentiment && data.sentiment && (
              <p className="text-sm text-cyan-600 dark:text-cyan-400">
                Sentiment: {(data.sentiment * 100).toFixed(1)}%
              </p>
            )}
            {showPatterns && data.patterns && data.patterns.length > 0 && (
              <div className="mt-2 border-t border-gray-200 pt-2 dark:border-gray-700">
                <p className="text-sm font-medium">Patterns Detected:</p>
                {data.patterns.map((pattern: Pattern, index: number) => (
                  <p
                    key={index}
                    className={`text-sm ${
                      pattern.type === 'bullish'
                        ? 'text-green-600 dark:text-green-400'
                        : 'text-red-600 dark:text-red-400'
                    }`}
                  >
                    {pattern.name}
                  </p>
                ))}
              </div>
            )}
          </div>
        </div>
      );
    }
    return null;
  };

  // Handle custom pattern drawing
  const handleChartClick = useCallback((e: any) => {
    if (!isDrawingPattern) return;

    const point = {
      x: e.activeLabel,
      y: e.activePayload?.[0]?.value
    };

    setCustomPatternPoints(prev => [...prev, point]);
  }, [isDrawingPattern]);

  const handleSaveCustomPattern = useCallback(() => {
    if (customPatternPoints.length < 3) {
      alert('Please draw at least 3 points for the pattern');
      return;
    }

    const patternName = prompt('Enter a name for this pattern:');
    if (!patternName) return;

    const patternType = confirm('Is this a bullish pattern?') ? 'bullish' : 'bearish';
    const description = prompt('Enter a description for this pattern:') || '';

    PatternRecognition.addCustomPattern({
      name: patternName,
      points: customPatternPoints,
      tolerance: 0.2, // Default tolerance
      type: patternType,
      description
    });

    setCustomPatternPoints([]);
    setIsDrawingPattern(false);
  }, [customPatternPoints]);

  // Handle user preference updates
  const handlePreferenceChange = useCallback((prefs: Partial<UserPreferences>) => {
    PatternRecognition.setUserPreferences(prefs);
    setUserPreferences(PatternRecognition.getUserPreferences());
  }, []);

  return (
    <div className="space-y-4 rounded-lg border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-800 dark:bg-gray-900">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-xl font-semibold">{title}</h2>
          <div className="flex items-center space-x-2">
            {symbol && (
              <span className="text-sm text-gray-600 dark:text-gray-400">
                {isConnected ? (
                  <span className="flex items-center">
                    <span className="mr-1 h-2 w-2 rounded-full bg-green-500"></span>
                    Live
                  </span>
                ) : (
                  <span className="flex items-center">
                    <span className="mr-1 h-2 w-2 rounded-full bg-red-500"></span>
                    Disconnected
                  </span>
                )}
              </span>
            )}
            {error && (
              <span className="text-sm text-red-500">{error}</span>
            )}
          </div>
        </div>
        <div className="flex items-center space-x-2">
          <button
            onClick={handleExportConfig}
            className="rounded-md bg-blue-100 px-3 py-1 text-sm font-medium text-blue-700 hover:bg-blue-200 dark:bg-blue-900 dark:text-blue-300"
          >
            Share Config
          </button>
          <button
            onClick={() => {
              const encoded = prompt('Enter configuration code:');
              if (encoded) handleImportConfig(encoded);
            }}
            className="rounded-md bg-purple-100 px-3 py-1 text-sm font-medium text-purple-700 hover:bg-purple-200 dark:bg-purple-900 dark:text-purple-300"
          >
            Import Config
          </button>
          <button
            onClick={handleZoomOut}
            className="rounded-md bg-gray-100 px-3 py-1 text-sm font-medium text-gray-700 hover:bg-gray-200 dark:bg-gray-800 dark:text-gray-300"
          >
            Reset Zoom
          </button>
        </div>
      </div>

      <div className="flex items-center space-x-2 mb-4">
        <button
          onClick={() => setIsDrawingPattern(!isDrawingPattern)}
          className={`rounded-md px-3 py-1 text-sm font-medium ${
            isDrawingPattern
              ? 'bg-blue-600 text-white'
              : 'bg-blue-100 text-blue-700 hover:bg-blue-200 dark:bg-blue-900 dark:text-blue-300'
          }`}
        >
          {isDrawingPattern ? 'Drawing Pattern...' : 'Draw Custom Pattern'}
        </button>
        {isDrawingPattern && customPatternPoints.length > 0 && (
          <button
            onClick={handleSaveCustomPattern}
            className="rounded-md bg-green-100 px-3 py-1 text-sm font-medium text-green-700 hover:bg-green-200 dark:bg-green-900 dark:text-green-300"
          >
            Save Pattern
          </button>
        )}
        <button
          onClick={() => setShowBacktestResults(!showBacktestResults)}
          className={`rounded-md px-3 py-1 text-sm font-medium ${
            showBacktestResults
              ? 'bg-purple-600 text-white'
              : 'bg-purple-100 text-purple-700 hover:bg-purple-200 dark:bg-purple-900 dark:text-purple-300'
          }`}
        >
          {showBacktestResults ? 'Hide Backtesting' : 'Show Backtesting'}
        </button>
        <button
          onClick={() => setShowPatternCustomization(!showPatternCustomization)}
          className={`rounded-md px-3 py-1 text-sm font-medium ${
            showPatternCustomization
              ? 'bg-indigo-600 text-white'
              : 'bg-indigo-100 text-indigo-700 hover:bg-indigo-200 dark:bg-indigo-900 dark:text-indigo-300'
          }`}
        >
          Customize Patterns
        </button>
        <button
          onClick={() => setShowCombinationAnalysis(!showCombinationAnalysis)}
          className={`rounded-md px-3 py-1 text-sm font-medium ${
            showCombinationAnalysis
              ? 'bg-amber-600 text-white'
              : 'bg-amber-100 text-amber-700 hover:bg-amber-200 dark:bg-amber-900 dark:text-amber-300'
          }`}
        >
          Pattern Combinations
        </button>
        <button
          onClick={() => setShowProbabilityForecast(!showProbabilityForecast)}
          className={`rounded-md px-3 py-1 text-sm font-medium ${
            showProbabilityForecast
              ? 'bg-emerald-600 text-white'
              : 'bg-emerald-100 text-emerald-700 hover:bg-emerald-200 dark:bg-emerald-900 dark:text-emerald-300'
          }`}
        >
          Probability Forecast
        </button>
      </div>

      <div style={{ width: '100%', height }}>
        <ResponsiveContainer>
          <LineChart
            data={data}
            onMouseDown={handleZoomStart}
            onMouseMove={handleZoomMove}
            onMouseUp={handleZoomEnd}
            onClick={handleChartClick}
          >
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis
              dataKey="date"
              domain={[zoomState.left || 'dataMin', zoomState.right || 'dataMax']}
              type="category"
            />
            <YAxis yAxisId="price" domain={['auto', 'auto']} />
            {showVolume && <YAxis yAxisId="volume" orientation="right" />}
            <Tooltip content={<CustomTooltip />} />
            <Legend />

            <Line
              yAxisId="price"
              type="monotone"
              dataKey="actual"
              stroke="#10B981"
              dot={false}
              name="Actual Price"
            />

            {showPrediction && (
              <Line
                yAxisId="price"
                type="monotone"
                dataKey="predicted"
                stroke="#3B82F6"
                strokeDasharray="5 5"
                dot={false}
                name="Predicted Price"
              />
            )}

            {showIndicators && (
              <>
                <Line
                  yAxisId="price"
                  type="monotone"
                  dataKey="ma20"
                  stroke="#22C55E"
                  dot={false}
                  name="MA20"
                />
                <Line
                  yAxisId="price"
                  type="monotone"
                  dataKey="ma50"
                  stroke="#A855F7"
                  dot={false}
                  name="MA50"
                />
                <Line
                  yAxisId="price"
                  type="monotone"
                  dataKey="ma200"
                  stroke="#EAB308"
                  dot={false}
                  name="MA200"
                />
                <Line
                  yAxisId="price"
                  type="monotone"
                  dataKey="bollingerUpper"
                  stroke="#EC4899"
                  strokeDasharray="3 3"
                  dot={false}
                  name="Bollinger Upper"
                />
                <Line
                  yAxisId="price"
                  type="monotone"
                  dataKey="bollingerLower"
                  stroke="#EC4899"
                  strokeDasharray="3 3"
                  dot={false}
                  name="Bollinger Lower"
                />
              </>
            )}

            {showVolume && (
              <Line
                yAxisId="volume"
                type="monotone"
                dataKey="volume"
                stroke="#64748B"
                dot={false}
                name="Volume"
              />
            )}

            {showSentiment && (
              <Line
                yAxisId="price"
                type="monotone"
                dataKey="sentiment"
                stroke="#06B6D4"
                dot={false}
                name="Sentiment Score"
              />
            )}

            {showPatterns && renderPatternMarkers()}

            {zoomState.refAreaLeft && zoomState.refAreaRight && (
              <ReferenceArea
                yAxisId="price"
                x1={zoomState.refAreaLeft}
                x2={zoomState.refAreaRight}
                strokeOpacity={0.3}
              />
            )}

            <Brush dataKey="date" height={30} stroke="#8884d8" />

            {/* Custom Pattern Drawing */}
            {isDrawingPattern && customPatternPoints.length > 0 && (
              <Line
                type="linear"
                data={customPatternPoints}
                dataKey="y"
                stroke="#EC4899"
                strokeWidth={2}
                strokeDasharray="5 5"
                dot={{ r: 4, fill: '#EC4899' }}
              />
            )}
          </LineChart>
        </ResponsiveContainer>
      </div>

      <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
        <div className="rounded-lg bg-green-50 p-3 dark:bg-green-900/20">
          <p className="text-sm text-gray-600 dark:text-gray-400">Current Price</p>
          <p className="text-lg font-semibold text-green-600 dark:text-green-400">
            ${data[data.length - 1]?.actual?.toLocaleString() ?? 'N/A'}
          </p>
        </div>
        {showPrediction && data[data.length - 1]?.predicted && (
          <div className="rounded-lg bg-blue-50 p-3 dark:bg-blue-900/20">
            <p className="text-sm text-gray-600 dark:text-gray-400">Predicted</p>
            <p className="text-lg font-semibold text-blue-600 dark:text-blue-400">
              ${data[data.length - 1]?.predicted?.toLocaleString() ?? 'N/A'}
            </p>
          </div>
        )}
        {showVolume && (
          <div className="rounded-lg bg-gray-50 p-3 dark:bg-gray-800">
            <p className="text-sm text-gray-600 dark:text-gray-400">Volume</p>
            <p className="text-lg font-semibold text-gray-600 dark:text-gray-400">
              {data[data.length - 1]?.volume?.toLocaleString()}
            </p>
          </div>
        )}
        {showSentiment && (
          <div className="rounded-lg bg-cyan-50 p-3 dark:bg-cyan-900/20">
            <p className="text-sm text-gray-600 dark:text-gray-400">Sentiment</p>
            <p className="text-lg font-semibold text-cyan-600 dark:text-cyan-400">
              {(data[data.length - 1]?.sentiment ?? 0 * 100).toFixed(1)}%
            </p>
          </div>
        )}
      </div>

      {/* Pattern Details with Enhanced Backtesting Results */}
      {selectedPattern && (
        <div className="mt-4 rounded-lg border border-gray-200 bg-gray-50 p-4 dark:border-gray-700 dark:bg-gray-800">
          <div className="flex items-center justify-between mb-4">
            <h3 className="font-medium">Pattern Details</h3>
            <div className="space-x-2">
              <button
                onClick={() => setShowAdvancedMetrics(!showAdvancedMetrics)}
                className={`rounded-md px-3 py-1 text-sm font-medium ${
                  showAdvancedMetrics
                    ? 'bg-purple-600 text-white'
                    : 'bg-purple-100 text-purple-700 hover:bg-purple-200 dark:bg-purple-900 dark:text-purple-300'
                }`}
              >
                {showAdvancedMetrics ? 'Show Basic Metrics' : 'Show Advanced Metrics'}
              </button>
              <button
                onClick={() => {
                  const encoded = PatternRecognition.exportPatterns();
                  navigator.clipboard.writeText(encoded);
                  alert('Pattern configuration copied to clipboard!');
                }}
                className="rounded-md bg-blue-100 px-3 py-1 text-sm font-medium text-blue-700 hover:bg-blue-200 dark:bg-blue-900 dark:text-blue-300"
              >
                Export Patterns
              </button>
              <button
                onClick={() => {
                  const encoded = prompt('Enter pattern configuration:');
                  if (encoded && PatternRecognition.importPatterns(encoded)) {
                    alert('Patterns imported successfully!');
                  } else {
                    alert('Failed to import patterns. Invalid format.');
                  }
                }}
                className="rounded-md bg-green-100 px-3 py-1 text-sm font-medium text-green-700 hover:bg-green-200 dark:bg-green-900 dark:text-green-300"
              >
                Import Patterns
              </button>
            </div>
          </div>

          <div className="space-y-2">
            <p className="text-sm text-gray-600 dark:text-gray-400">
              Date: {selectedPattern.date}
            </p>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              Pattern: {selectedPattern.pattern.name}
            </p>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              {selectedPattern.pattern.description}
            </p>

            {showBacktestResults && selectedPattern.pattern.performance && (
              <div className="mt-3">
                {!showAdvancedMetrics ? (
                  // Basic Metrics
                  <div className="grid grid-cols-2 gap-4">
                    <div className="rounded-lg bg-green-50 p-3 dark:bg-green-900/20">
                      <p className="text-sm text-gray-600 dark:text-gray-400">Win Rate</p>
                      <p className="text-lg font-semibold text-green-600 dark:text-green-400">
                        {(selectedPattern.pattern.performance.winRate * 100).toFixed(1)}%
                      </p>
                    </div>
                    <div className="rounded-lg bg-blue-50 p-3 dark:bg-blue-900/20">
                      <p className="text-sm text-gray-600 dark:text-gray-400">Avg Profit</p>
                      <p className="text-lg font-semibold text-blue-600 dark:text-blue-400">
                        {(selectedPattern.pattern.performance.avgProfit * 100).toFixed(1)}%
                      </p>
                    </div>
                    <div className="rounded-lg bg-red-50 p-3 dark:bg-red-900/20">
                      <p className="text-sm text-gray-600 dark:text-gray-400">Avg Loss</p>
                      <p className="text-lg font-semibold text-red-600 dark:text-red-400">
                        {(selectedPattern.pattern.performance.avgLoss * 100).toFixed(1)}%
                      </p>
                    </div>
                    <div className="rounded-lg bg-purple-50 p-3 dark:bg-purple-900/20">
                      <p className="text-sm text-gray-600 dark:text-gray-400">Risk/Reward</p>
                      <p className="text-lg font-semibold text-purple-600 dark:text-purple-400">
                        {selectedPattern.pattern.performance.riskRewardRatio.toFixed(2)}
                      </p>
                    </div>
                  </div>
                ) : (
                  // Advanced Metrics
                  <div className="grid grid-cols-3 gap-4">
                    <div className="rounded-lg bg-orange-50 p-3 dark:bg-orange-900/20">
                      <p className="text-sm text-gray-600 dark:text-gray-400">Max Drawdown</p>
                      <p className="text-lg font-semibold text-orange-600 dark:text-orange-400">
                        {((selectedPattern?.pattern?.performance?.maxDrawdown ?? 0) * 100).toFixed(1)}%
                      </p>
                    </div>
                    <div className="rounded-lg bg-cyan-50 p-3 dark:bg-cyan-900/20">
                      <p className="text-sm text-gray-600 dark:text-gray-400">Profit Factor</p>
                      <p className="text-lg font-semibold text-cyan-600 dark:text-cyan-400">
                        {selectedPattern.pattern.performance.profitFactor?.toFixed(2) ?? 'N/A'}
                      </p>
                    </div>
                    <div className="rounded-lg bg-pink-50 p-3 dark:bg-pink-900/20">
                      <p className="text-sm text-gray-600 dark:text-gray-400">Sharpe Ratio</p>
                      <p className="text-lg font-semibold text-pink-600 dark:text-pink-400">
                        {selectedPattern.pattern.performance.sharpeRatio?.toFixed(2) ?? 'N/A'}
                      </p>
                    </div>
                    <div className="rounded-lg bg-indigo-50 p-3 dark:bg-indigo-900/20">
                      <p className="text-sm text-gray-600 dark:text-gray-400">Recovery Factor</p>
                      <p className="text-lg font-semibold text-indigo-600 dark:text-indigo-400">
                        {selectedPattern.pattern.performance.recoveryFactor?.toFixed(2) ?? 'N/A'}
                      </p>
                    </div>
                    <div className="rounded-lg bg-emerald-50 p-3 dark:bg-emerald-900/20">
                      <p className="text-sm text-gray-600 dark:text-gray-400">Consecutive Wins</p>
                      <p className="text-lg font-semibold text-emerald-600 dark:text-emerald-400">
                        {selectedPattern.pattern.performance.consecutiveWins ?? 'N/A'}
                      </p>
                    </div>
                    <div className="rounded-lg bg-rose-50 p-3 dark:bg-rose-900/20">
                      <p className="text-sm text-gray-600 dark:text-gray-400">Consecutive Losses</p>
                      <p className="text-lg font-semibold text-rose-600 dark:text-rose-400">
                        {selectedPattern.pattern.performance.consecutiveLosses ?? 'N/A'}
                      </p>
                    </div>
                    <div className="rounded-lg bg-amber-50 p-3 dark:bg-amber-900/20">
                      <p className="text-sm text-gray-600 dark:text-gray-400">Largest Win</p>
                      <p className="text-lg font-semibold text-amber-600 dark:text-amber-400">
                        {((selectedPattern?.pattern?.performance?.largestWin ?? 0) * 100).toFixed(1)}%
                      </p>
                    </div>
                    <div className="rounded-lg bg-violet-50 p-3 dark:bg-violet-900/20">
                      <p className="text-sm text-gray-600 dark:text-gray-400">Largest Loss</p>
                      <p className="text-lg font-semibold text-violet-600 dark:text-violet-400">
                        {((selectedPattern?.pattern?.performance?.largestLoss ?? 0) * 100).toFixed(1)}%
                      </p>
                    </div>
                    <div className="rounded-lg bg-teal-50 p-3 dark:bg-teal-900/20">
                      <p className="text-sm text-gray-600 dark:text-gray-400">Avg Holding Period</p>
                      <p className="text-lg font-semibold text-teal-600 dark:text-teal-400">
                        {selectedPattern.pattern.performance.averageHoldingPeriod?.toFixed(1) ?? 'N/A'} days
                      </p>
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      )}

      {/* Pattern Configuration */}
      <div className="mt-4 space-y-2">
        <h3 className="text-sm font-medium">Pattern Recognition Settings</h3>
        <div className="flex flex-wrap gap-2">
          {Object.keys(PatternRecognition).map(pattern => (
            <label
              key={pattern}
              className="flex items-center space-x-2"
            >
              <input
                type="checkbox"
                checked={config.patterns.enabled.includes(pattern)}
                onChange={(e) => {
                  const enabled = e.target.checked
                    ? [...config.patterns.enabled, pattern]
                    : config.patterns.enabled.filter(p => p !== pattern);
                  handleConfigChange({
                    patterns: { ...config.patterns, enabled }
                  });
                }}
                className="rounded border-gray-300"
              />
              <span className="text-sm text-gray-700 dark:text-gray-300">
                {pattern}
              </span>
            </label>
          ))}
        </div>
        <div className="flex items-center space-x-4">
          <label className="flex items-center space-x-2">
            <span className="text-sm text-gray-700 dark:text-gray-300">
              Confidence Threshold
            </span>
            <input
              type="range"
              min="0"
              max="100"
              value={config.patterns.confidence}
              onChange={(e) => {
                handleConfigChange({
                  patterns: {
                    ...config.patterns,
                    confidence: Number(e.target.value)
                  }
                });
              }}
              className="w-32"
            />
            <span className="text-sm text-gray-700 dark:text-gray-300">
              {config.patterns.confidence}%
            </span>
          </label>
        </div>
      </div>

      {/* Pattern Customization Panel */}
      {showPatternCustomization && (
        <div className="mt-4 rounded-lg border border-gray-200 bg-gray-50 p-4 dark:border-gray-700 dark:bg-gray-800">
          <div className="mb-6">
            <h3 className="text-lg font-medium mb-4">Candle Funneling Settings</h3>
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium mb-1">Volume Threshold</label>
                <input
                  type="range"
                  min="1"
                  max="3"
                  step="0.1"
                  value={userPreferences.candleFunneling.volumeThreshold}
                  onChange={(e) => handlePreferenceChange({
                    candleFunneling: {
                      ...userPreferences.candleFunneling,
                      volumeThreshold: parseFloat(e.target.value)
                    }
                  })}
                  className="w-full"
                />
                <div className="flex justify-between text-xs">
                  <span>1x</span>
                  <span>{userPreferences.candleFunneling.volumeThreshold}x</span>
                  <span>3x</span>
                </div>
              </div>

              <div>
                <label className="block text-sm font-medium mb-1">Body/Wick Ratio</label>
                <input
                  type="range"
                  min="0.2"
                  max="0.8"
                  step="0.1"
                  value={userPreferences.candleFunneling.bodyToWickRatio}
                  onChange={(e) => handlePreferenceChange({
                    candleFunneling: {
                      ...userPreferences.candleFunneling,
                      bodyToWickRatio: parseFloat(e.target.value)
                    }
                  })}
                  className="w-full"
                />
                <div className="flex justify-between text-xs">
                  <span>20%</span>
                  <span>{(userPreferences.candleFunneling.bodyToWickRatio * 100).toFixed(0)}%</span>
                  <span>80%</span>
                </div>
              </div>

              <div>
                <label className="block text-sm font-medium mb-1">Momentum Threshold</label>
                <input
                  type="range"
                  min="0.01"
                  max="0.05"
                  step="0.01"
                  value={userPreferences.candleFunneling.momentumThreshold}
                  onChange={(e) => handlePreferenceChange({
                    candleFunneling: {
                      ...userPreferences.candleFunneling,
                      momentumThreshold: parseFloat(e.target.value)
                    }
                  })}
                  className="w-full"
                />
                <div className="flex justify-between text-xs">
                  <span>1%</span>
                  <span>{(userPreferences.candleFunneling.momentumThreshold * 100).toFixed(0)}%</span>
                  <span>5%</span>
                </div>
              </div>

              <div>
                <label className="block text-sm font-medium mb-1">Trend Strength</label>
                <input
                  type="range"
                  min="0.3"
                  max="0.9"
                  step="0.1"
                  value={userPreferences.candleFunneling.trendStrength}
                  onChange={(e) => handlePreferenceChange({
                    candleFunneling: {
                      ...userPreferences.candleFunneling,
                      trendStrength: parseFloat(e.target.value)
                    }
                  })}
                  className="w-full"
                />
                <div className="flex justify-between text-xs">
                  <span>Weak</span>
                  <span>{(userPreferences.candleFunneling.trendStrength * 100).toFixed(0)}%</span>
                  <span>Strong</span>
                </div>
              </div>
            </div>
          </div>

          <div className="mb-6">
            <h3 className="text-lg font-medium mb-4">Chart Appearance</h3>
            <div className="grid grid-cols-3 gap-4">
              <div>
                <label className="block text-sm font-medium mb-1">Bullish Color</label>
                <input
                  type="color"
                  value={userPreferences.chartSettings.candleColors.bullish}
                  onChange={(e) => handlePreferenceChange({
                    chartSettings: {
                      ...userPreferences.chartSettings,
                      candleColors: {
                        ...userPreferences.chartSettings.candleColors,
                        bullish: e.target.value
                      }
                    }
                  })}
                  className="w-full h-8 rounded"
                />
              </div>

              <div>
                <label className="block text-sm font-medium mb-1">Bearish Color</label>
                <input
                  type="color"
                  value={userPreferences.chartSettings.candleColors.bearish}
                  onChange={(e) => handlePreferenceChange({
                    chartSettings: {
                      ...userPreferences.chartSettings,
                      candleColors: {
                        ...userPreferences.chartSettings.candleColors,
                        bearish: e.target.value
                      }
                    }
                  })}
                  className="w-full h-8 rounded"
                />
              </div>

              <div>
                <label className="block text-sm font-medium mb-1">Grid Opacity</label>
                <input
                  type="range"
                  min="0"
                  max="1"
                  step="0.1"
                  value={userPreferences.chartSettings.gridLines.opacity}
                  onChange={(e) => handlePreferenceChange({
                    chartSettings: {
                      ...userPreferences.chartSettings,
                      gridLines: {
                        ...userPreferences.chartSettings.gridLines,
                        opacity: parseFloat(e.target.value)
                      }
                    }
                  })}
                  className="w-full"
                />
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Enhanced Pattern Combination Analysis */}
      {showCombinationAnalysis && selectedPattern && selectedPattern.pattern.patternCombinations && (
        <div className="mt-4 rounded-lg border border-gray-200 bg-gray-50 p-4 dark:border-gray-700 dark:bg-gray-800">
          <h3 className="text-lg font-medium mb-4">Pattern Combinations</h3>
          
          <div className="space-y-6">
            {selectedPattern.pattern.patternCombinations.map((combo: PatternCombination, index: number) => (
              <div
                key={index}
                className="rounded-lg border border-gray-200 p-4 dark:border-gray-700"
              >
                <div className="flex justify-between items-center mb-4">
                  <h4 className="font-medium text-lg">
                    {combo.patterns.map(p => p.name).join(' + ')}
                  </h4>
                  <div className="flex items-center space-x-4">
                    <span className={`text-sm font-medium px-2 py-1 rounded ${
                      combo.synergy > 0.7 ? 'bg-green-100 text-green-800' :
                      combo.synergy > 0.4 ? 'bg-amber-100 text-amber-800' : 'bg-red-100 text-red-800'
                    }`}>
                      {(combo.synergy * 100).toFixed(1)}% Synergy
                    </span>
                    <span className="text-sm font-medium px-2 py-1 rounded bg-blue-100 text-blue-800">
                      {(combo.reliability * 100).toFixed(1)}% Reliable
                    </span>
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-6">
                  <div className="space-y-2">
                    <h5 className="font-medium mb-2">Technical Metrics</h5>
                    {Object.entries(combo.metrics).slice(0, 4).map(([key, value]: [string, number]) => (
                      <div key={key} className="flex items-center space-x-2">
                        <span className="text-sm capitalize">{key.replace(/([A-Z])/g, ' $1').trim()}:</span>
                        <div className="flex-1 h-2 bg-gray-200 rounded-full">
                          <div
                            className={`h-2 rounded-full ${
                              value > 0.7 ? 'bg-green-500' :
                              value > 0.4 ? 'bg-amber-500' : 'bg-red-500'
                            }`}
                            style={{ width: `${value * 100}%` }}
                          />
                        </div>
                        <span className="text-sm">{(value * 100).toFixed(0)}%</span>
                      </div>
                    ))}
                  </div>

                  <div className="space-y-2">
                    <h5 className="font-medium mb-2">Market Context</h5>
                    {Object.entries(combo.metrics).slice(4).map(([key, value]: [string, number]) => (
                      <div key={key} className="flex items-center space-x-2">
                        <span className="text-sm capitalize">{key.replace(/([A-Z])/g, ' $1').trim()}:</span>
                        <div className="flex-1 h-2 bg-gray-200 rounded-full">
                          <div
                            className={`h-2 rounded-full ${
                              value > 0.7 ? 'bg-green-500' :
                              value > 0.4 ? 'bg-amber-500' : 'bg-red-500'
                            }`}
                            style={{ width: `${value * 100}%` }}
                          />
                        </div>
                        <span className="text-sm">{(value * 100).toFixed(0)}%</span>
                      </div>
                    ))}
                  </div>
                </div>

                <div className="mt-4 pt-4 border-t border-gray-200 dark:border-gray-700">
                  <div className="flex justify-between items-center">
                    <span className="text-sm font-medium">Timeframe: {combo.timeframe} periods</span>
                    <button
                      onClick={() => {/* Add logic to save combination */}}
                      className="px-3 py-1 text-sm font-medium text-blue-700 bg-blue-100 rounded-md hover:bg-blue-200"
                    >
                      Save Combination
                    </button>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Probability Forecast */}
      {showProbabilityForecast && selectedPattern && selectedPattern.pattern.forecast && (
        <div className="mt-4 rounded-lg border border-gray-200 bg-gray-50 p-4 dark:border-gray-700 dark:bg-gray-800">
          <h3 className="mb-4 font-medium">Pattern Forecast</h3>
          
          <div className="grid grid-cols-2 gap-4 mb-4">
            <div className="rounded-lg bg-blue-50 p-3 dark:bg-blue-900/20">
              <p className="text-sm text-gray-600 dark:text-gray-400">Success Probability</p>
              <p className="text-lg font-semibold text-blue-600 dark:text-blue-400">
                {(selectedPattern.pattern.forecast.probability * 100).toFixed(1)}%
              </p>
            </div>
            
            <div className="rounded-lg bg-green-50 p-3 dark:bg-green-900/20">
              <p className="text-sm text-gray-600 dark:text-gray-400">Expected Return</p>
              <p className="text-lg font-semibold text-green-600 dark:text-green-400">
                {(selectedPattern.pattern.forecast.expectedReturn * 100).toFixed(1)}%
              </p>
            </div>
          </div>

          <div className="space-y-3">
            <h4 className="text-sm font-medium">Contributing Factors</h4>
            
            {Object.entries(selectedPattern.pattern.forecast.factors).map(([factor, value]) => {
              // Ensure value is a number and provide type safety
              const numericValue = typeof value === 'number' ? value : 0;
              return (
                <div key={factor} className="flex items-center space-x-2">
                  <span className="text-sm capitalize">{factor}:</span>
                  <div className="flex-1 h-2 bg-gray-200 rounded-full dark:bg-gray-700">
                    <div
                      className={`h-2 rounded-full ${
                        numericValue > 0.7 ? 'bg-green-500' :
                        numericValue > 0.4 ? 'bg-amber-500' : 'bg-red-500'
                      }`}
                      style={{ width: `${numericValue * 100}%` }}
                    />
                  </div>
                  <span className="text-sm">{(numericValue * 100).toFixed(0)}%</span>
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
};

export default InteractiveChart; 