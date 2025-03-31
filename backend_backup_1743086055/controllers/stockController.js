const Stock = require('../models/Stock');

// @desc    Get all stocks
// @route   GET /api/stocks
// @access  Private
exports.getStocks = async (req, res) => {
  try {
    const stocks = await Stock.find();
    res.json(stocks);
  } catch (err) {
    console.error(err);
    res.status(500).json({ message: 'Server error' });
  }
};

// @desc    Get single stock
// @route   GET /api/stocks/:id
// @access  Private
exports.getStockById = async (req, res) => {
  try {
    const stock = await Stock.findById(req.params.id);
    
    if (!stock) {
      return res.status(404).json({ message: 'Stock not found' });
    }
    
    res.json(stock);
  } catch (err) {
    console.error(err);
    if (err.kind === 'ObjectId') {
      return res.status(404).json({ message: 'Stock not found' });
    }
    res.status(500).json({ message: 'Server error' });
  }
};

// @desc    Create sample stocks for testing
// @route   POST /api/stocks/sample
// @access  Private
exports.createSampleStocks = async (req, res) => {
  try {
    // Check if stocks already exist
    const stockCount = await Stock.countDocuments();
    
    if (stockCount > 0) {
      return res.status(400).json({ message: 'Sample stocks already exist' });
    }
    
    // Sample stock data
    const stocks = [
      {
        symbol: 'AAPL',
        name: 'Apple Inc.',
        sector: 'Technology',
        industry: 'Consumer Electronics',
        price: {
          currentPrice: 176.50,
          previousClose: 174.20,
          change: 2.30,
          changePercent: 1.32
        },
        marketCap: 2850000000000,
        metrics: {
          pe: 28.5,
          eps: 6.14,
          dividend: 0.92,
          dividendYield: 0.52,
          beta: 1.28
        },
        analysis: {
          riskLevel: 'medium',
          sentiment: 'bullish',
          recommendation: 'buy'
        }
      },
      {
        symbol: 'MSFT',
        name: 'Microsoft Corporation',
        sector: 'Technology',
        industry: 'Software',
        price: {
          currentPrice: 305.80,
          previousClose: 303.50,
          change: 2.30,
          changePercent: 0.76
        },
        marketCap: 2270000000000,
        metrics: {
          pe: 32.7,
          eps: 9.28,
          dividend: 2.48,
          dividendYield: 0.82,
          beta: 0.93
        },
        analysis: {
          riskLevel: 'low',
          sentiment: 'bullish',
          recommendation: 'strong buy'
        }
      },
      {
        symbol: 'AMZN',
        name: 'Amazon.com Inc.',
        sector: 'Consumer Cyclical',
        industry: 'Internet Retail',
        price: {
          currentPrice: 128.90,
          previousClose: 127.80,
          change: 1.10,
          changePercent: 0.86
        },
        marketCap: 1320000000000,
        metrics: {
          pe: 58.6,
          eps: 2.17,
          dividend: 0,
          dividendYield: 0,
          beta: 1.19
        },
        analysis: {
          riskLevel: 'medium',
          sentiment: 'neutral',
          recommendation: 'buy'
        }
      },
      {
        symbol: 'GOOGL',
        name: 'Alphabet Inc.',
        sector: 'Communication Services',
        industry: 'Internet Content & Information',
        price: {
          currentPrice: 134.60,
          previousClose: 133.20,
          change: 1.40,
          changePercent: 1.05
        },
        marketCap: 1720000000000,
        metrics: {
          pe: 25.5,
          eps: 5.28,
          dividend: 0,
          dividendYield: 0,
          beta: 1.05
        },
        analysis: {
          riskLevel: 'medium',
          sentiment: 'bullish',
          recommendation: 'buy'
        }
      },
      {
        symbol: 'META',
        name: 'Meta Platforms, Inc.',
        sector: 'Communication Services',
        industry: 'Internet Content & Information',
        price: {
          currentPrice: 292.50,
          previousClose: 289.70,
          change: 2.80,
          changePercent: 0.97
        },
        marketCap: 750000000000,
        metrics: {
          pe: 24.8,
          eps: 11.77,
          dividend: 0,
          dividendYield: 0,
          beta: 1.37
        },
        analysis: {
          riskLevel: 'medium',
          sentiment: 'neutral',
          recommendation: 'hold'
        }
      },
      {
        symbol: 'TSLA',
        name: 'Tesla, Inc.',
        sector: 'Consumer Cyclical',
        industry: 'Auto Manufacturers',
        price: {
          currentPrice: 180.20,
          previousClose: 183.40,
          change: -3.20,
          changePercent: -1.75
        },
        marketCap: 570000000000,
        metrics: {
          pe: 48.3,
          eps: 3.63,
          dividend: 0,
          dividendYield: 0,
          beta: 2.04
        },
        analysis: {
          riskLevel: 'high',
          sentiment: 'neutral',
          recommendation: 'hold'
        }
      },
      {
        symbol: 'JNJ',
        name: 'Johnson & Johnson',
        sector: 'Healthcare',
        industry: 'Drug Manufacturers',
        price: {
          currentPrice: 155.30,
          previousClose: 155.80,
          change: -0.50,
          changePercent: -0.32
        },
        marketCap: 405000000000,
        metrics: {
          pe: 17.8,
          eps: 8.73,
          dividend: 4.52,
          dividendYield: 2.91,
          beta: 0.54
        },
        analysis: {
          riskLevel: 'low',
          sentiment: 'neutral',
          recommendation: 'buy'
        }
      }
    ];
    
    // Insert sample stocks
    await Stock.insertMany(stocks);
    
    res.status(201).json({ message: 'Sample stocks created successfully', count: stocks.length });
  } catch (err) {
    console.error(err);
    res.status(500).json({ message: 'Server error' });
  }
}; 