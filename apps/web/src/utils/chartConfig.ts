interface ChartConfig {
  indicators: {
    showMA20: boolean;
    showMA50: boolean;
    showMA200: boolean;
    showRSI: boolean;
    showMACD: boolean;
    showBollingerBands: boolean;
  };
  display: {
    showVolume: boolean;
    showPredictions: boolean;
    showSentiment: boolean;
    showPatterns: boolean;
    darkMode: boolean;
    height: number;
  };
  patterns: {
    enabled: string[];
    confidence: number;
  };
  realtime: {
    enabled: boolean;
    interval: number;
  };
  appearance: {
    colors: {
      price: string;
      volume: string;
      ma20: string;
      ma50: string;
      ma200: string;
      bollingerBands: string;
      bullish: string;
      bearish: string;
      sentiment: string;
      prediction: string;
    };
    opacity: {
      volume: number;
      patterns: number;
      predictions: number;
    };
  };
}

export class ChartConfigManager {
  private static readonly CONFIG_KEY = 'chart_config';
  private static readonly DEFAULT_CONFIG: ChartConfig = {
    indicators: {
      showMA20: true,
      showMA50: true,
      showMA200: true,
      showRSI: true,
      showMACD: true,
      showBollingerBands: true,
    },
    display: {
      showVolume: true,
      showPredictions: true,
      showSentiment: true,
      showPatterns: true,
      darkMode: false,
      height: 500,
    },
    patterns: {
      enabled: [
        'Double Bottom',
        'Double Top',
        'Head and Shoulders',
        'Bullish Flag',
        'Triangle',
        'Wedge',
      ],
      confidence: 70,
    },
    realtime: {
      enabled: true,
      interval: 60000,
    },
    appearance: {
      colors: {
        price: '#10B981',
        volume: '#64748B',
        ma20: '#22C55E',
        ma50: '#A855F7',
        ma200: '#EAB308',
        bollingerBands: '#EC4899',
        bullish: '#22C55E',
        bearish: '#EF4444',
        sentiment: '#06B6D4',
        prediction: '#3B82F6',
      },
      opacity: {
        volume: 0.7,
        patterns: 0.3,
        predictions: 0.8,
      },
    },
  };

  static getConfig(): ChartConfig {
    try {
      const stored = localStorage.getItem(this.CONFIG_KEY);
      if (!stored) return this.DEFAULT_CONFIG;

      const config = JSON.parse(stored);
      return { ...this.DEFAULT_CONFIG, ...config };
    } catch (error) {
      console.error('Error loading chart config:', error);
      return this.DEFAULT_CONFIG;
    }
  }

  static saveConfig(config: Partial<ChartConfig>): void {
    try {
      const currentConfig = this.getConfig();
      const newConfig = { ...currentConfig, ...config };
      localStorage.setItem(this.CONFIG_KEY, JSON.stringify(newConfig));
    } catch (error) {
      console.error('Error saving chart config:', error);
    }
  }

  static exportConfig(): string {
    const config = this.getConfig();
    return btoa(JSON.stringify(config));
  }

  static importConfig(encodedConfig: string): boolean {
    try {
      const config = JSON.parse(atob(encodedConfig));
      this.validateConfig(config);
      this.saveConfig(config);
      return true;
    } catch (error) {
      console.error('Error importing chart config:', error);
      return false;
    }
  }

  static resetConfig(): void {
    localStorage.removeItem(this.CONFIG_KEY);
  }

  private static validateConfig(config: any): void {
    const requiredKeys = [
      'indicators',
      'display',
      'patterns',
      'realtime',
      'appearance',
    ];

    if (!config || typeof config !== 'object') {
      throw new Error('Invalid config format');
    }

    requiredKeys.forEach(key => {
      if (!(key in config)) {
        throw new Error(`Missing required key: ${key}`);
      }
    });

    // Validate color values
    Object.values(config.appearance.colors).forEach((color: any) => {
      if (typeof color !== 'string' || !color.match(/^#[0-9A-Fa-f]{6}$/)) {
        throw new Error('Invalid color format');
      }
    });

    // Validate numeric values
    Object.values(config.appearance.opacity).forEach((opacity: any) => {
      if (typeof opacity !== 'number' || opacity < 0 || opacity > 1) {
        throw new Error('Invalid opacity value');
      }
    });
  }

  static getShareableLink(config: ChartConfig): string {
    const encodedConfig = this.exportConfig();
    return `${window.location.origin}/chart?config=${encodedConfig}`;
  }

  static async loadConfigFromUrl(): Promise<ChartConfig | null> {
    try {
      const params = new URLSearchParams(window.location.search);
      const encodedConfig = params.get('config');
      if (!encodedConfig) return null;

      const config = JSON.parse(atob(encodedConfig));
      this.validateConfig(config);
      return config;
    } catch (error) {
      console.error('Error loading config from URL:', error);
      return null;
    }
  }
} 