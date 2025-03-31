import { z } from 'zod';

// Stock data schema
export const StockDataSchema = z.object({
  symbol: z.string(),
  price: z.number(),
  change: z.number(),
  changePercent: z.number(),
  volume: z.number(),
  marketCap: z.number().optional(),
});

export type StockData = z.infer<typeof StockDataSchema>;

// Investment recommendation schema
export const RecommendationSchema = z.object({
  symbol: z.string(),
  recommendationType: z.enum(['BUY', 'SELL', 'HOLD']),
  confidence: z.number(),
  targetPrice: z.number(),
  timeHorizon: z.string(),
  analysis: z.string(),
  riskLevel: z.enum(['LOW', 'MEDIUM', 'HIGH']),
});

export type Recommendation = z.infer<typeof RecommendationSchema>;

// Portfolio schema
export const PortfolioSchema = z.object({
  userId: z.string(),
  holdings: z.array(
    z.object({
      symbol: z.string(),
      shares: z.number(),
      averagePrice: z.number(),
      currentValue: z.number(),
    })
  ),
  totalValue: z.number(),
  cashBalance: z.number(),
});

export type Portfolio = z.infer<typeof PortfolioSchema>;

// Market sentiment schema
export const MarketSentimentSchema = z.object({
  symbol: z.string(),
  sentiment: z.enum(['BULLISH', 'BEARISH', 'NEUTRAL']),
  score: z.number(),
  newsCount: z.number(),
  latestNews: z.array(
    z.object({
      title: z.string(),
      sentiment: z.number(),
      source: z.string(),
      url: z.string(),
    })
  ),
});

export type MarketSentiment = z.infer<typeof MarketSentimentSchema>; 