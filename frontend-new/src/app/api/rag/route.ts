import { NextRequest, NextResponse } from 'next/server';

export async function POST(request: NextRequest) {
  const body = await request.json();
  const { query } = body;

  // Mock RAG response data
  const ragResponse = {
    query,
    answer: `Here's information about your question: "${query}"\n\nBased on current market data and financial analysis, this is a comprehensive answer to your query. The financial markets have been showing significant volatility recently, with major indices experiencing fluctuations due to economic indicators and geopolitical events.\n\nExperts suggest maintaining a diversified portfolio and considering long-term investment strategies rather than reacting to short-term market movements. For this specific query, it's worth noting that historical data shows patterns of recovery following similar market conditions.`,
    sources: [
      'https://www.bloomberg.com/markets',
      'https://www.wsj.com/market-data',
      'https://www.ft.com/markets',
      'https://www.reuters.com/markets'
    ],
    generatedAt: new Date().toISOString(),
    aiInsights: `Based on current market trends and historical data, investors should consider the broader economic indicators including interest rates, inflation forecasts, and sector performance when making investment decisions. The current market sentiment appears cautiously optimistic despite ongoing concerns about inflation and potential economic slowdown in certain regions.`
  };

  return NextResponse.json(ragResponse);
} 