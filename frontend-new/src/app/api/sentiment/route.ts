import { NextRequest, NextResponse } from 'next/server';

export async function GET(request: NextRequest) {
  const searchParams = request.nextUrl.searchParams;
  const symbol = searchParams.get('symbol') || '';
  const timeframe = searchParams.get('timeframe') || '7d';

  // Mock sentiment data
  const sentimentData = {
    symbol,
    timeframe,
    bullish: Math.round(Math.random() * 40 + 50), // Random value between 50-90
    bearish: Math.round(Math.random() * 30 + 10), // Random value between 10-40
    newsScore: Math.random() * 0.4 + 0.5, // Random value between 0.5-0.9
    socialScore: Math.random() * 0.4 + 0.4, // Random value between 0.4-0.8
    analystScore: Math.random() * 0.3 + 0.6, // Random value between 0.6-0.9
    newsSummary: `Recent news articles show a generally positive outlook for ${symbol}, with strong quarterly earnings and new product announcements driving investor interest.`,
    socialSentiment: `Social media sentiment for ${symbol} shows increasing retail investor interest, with ${Math.round(Math.random() * 40 + 60)}% positive mentions over the last ${timeframe === '1d' ? 'day' : timeframe === '7d' ? 'week' : 'month'}.`,
    analystInsights: `${Math.round(Math.random() * 4 + 6)}/10 analysts have a "Buy" rating for ${symbol}, with an average price target ${Math.round(Math.random() * 10 + 5)}% above current price.`,
    aiInsights: `Based on the analysis of market sentiment, technical indicators, and fundamental data, ${symbol} appears to be in a ${Math.random() > 0.5 ? 'bullish' : 'neutral'} trend. Key factors include ${Math.random() > 0.5 ? 'positive earnings surprise' : 'sector rotation'} and ${Math.random() > 0.5 ? 'increasing institutional ownership' : 'strong technical support'}.`
  };

  return NextResponse.json(sentimentData);
} 