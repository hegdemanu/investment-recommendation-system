import { NextRequest, NextResponse } from 'next/server';

export async function GET(request: NextRequest) {
  // Mock portfolio data
  const portfolioData = {
    totalValue: 125760.42,
    totalGain: 18432.21,
    totalGainPercent: 17.2,
    lastUpdated: new Date().toISOString(),
    positions: [
      {
        symbol: 'AAPL',
        name: 'Apple Inc.',
        shares: 150,
        avgCost: 145.32,
        currentPrice: 175.84,
        value: 26376.00,
        gain: 4578.00,
        gainPercent: 21.02,
        allocation: 20.97,
        sector: 'Technology'
      },
      {
        symbol: 'MSFT',
        name: 'Microsoft Corporation',
        shares: 85,
        avgCost: 285.43,
        currentPrice: 389.56,
        value: 33112.60,
        gain: 8850.45,
        gainPercent: 36.48,
        allocation: 26.33,
        sector: 'Technology'
      },
      {
        symbol: 'AMZN',
        name: 'Amazon.com Inc.',
        shares: 100,
        avgCost: 120.54,
        currentPrice: 180.12,
        value: 18012.00,
        gain: 5958.00,
        gainPercent: 49.43,
        allocation: 14.32,
        sector: 'Consumer Cyclical'
      },
      {
        symbol: 'JNJ',
        name: 'Johnson & Johnson',
        shares: 120,
        avgCost: 164.32,
        currentPrice: 152.56,
        value: 18307.20,
        gain: -1411.20,
        gainPercent: -7.16,
        allocation: 14.56,
        sector: 'Healthcare'
      },
      {
        symbol: 'JPM',
        name: 'JPMorgan Chase & Co.',
        shares: 110,
        avgCost: 145.76,
        currentPrice: 189.43,
        value: 20837.30,
        gain: 4801.30,
        gainPercent: 29.96,
        allocation: 16.57,
        sector: 'Financial Services'
      },
      {
        symbol: 'PG',
        name: 'Procter & Gamble Co.',
        shares: 78,
        avgCost: 142.35,
        currentPrice: 155.25,
        value: 12109.50,
        gain: 1006.20,
        gainPercent: 9.06,
        allocation: 9.63,
        sector: 'Consumer Defensive'
      }
    ],
    sectorAllocation: [
      { sector: 'Technology', allocation: 46.68 },
      { sector: 'Financial Services', allocation: 16.57 },
      { sector: 'Healthcare', allocation: 14.56 },
      { sector: 'Consumer Cyclical', allocation: 14.32 },
      { sector: 'Consumer Defensive', allocation: 9.63 }
    ],
    performance: [
      { date: '2023-04-01', value: 100000.00 },
      { date: '2023-05-01', value: 102500.00 },
      { date: '2023-06-01', value: 105400.00 },
      { date: '2023-07-01', value: 103200.00 },
      { date: '2023-08-01', value: 108600.00 },
      { date: '2023-09-01', value: 112400.00 },
      { date: '2023-10-01', value: 110300.00 },
      { date: '2023-11-01', value: 114800.00 },
      { date: '2023-12-01', value: 118500.00 },
      { date: '2024-01-01', value: 121200.00 },
      { date: '2024-02-01', value: 123800.00 },
      { date: '2024-03-01', value: 125760.42 }
    ],
    recommendations: [
      {
        type: 'Rebalance',
        description: 'Your technology allocation exceeds target. Consider rebalancing to reduce risk.',
        impact: 'Reduce portfolio volatility'
      },
      {
        type: 'Diversify',
        description: 'Add exposure to emerging markets for better diversification.',
        impact: 'Improve risk-adjusted returns'
      },
      {
        type: 'Tax Optimization',
        description: 'Consider tax-loss harvesting with JNJ position.',
        impact: 'Potential tax savings'
      }
    ]
  };

  return NextResponse.json(portfolioData);
} 