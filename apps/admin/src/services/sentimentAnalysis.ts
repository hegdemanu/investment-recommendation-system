import axios from 'axios';
import { FinBERT } from '../models/finbert';
import { NewsAPI } from '../integrations/newsapi';
import { RedisCache } from '../utils/cache';

interface SentimentResult {
  score: number;
  label: 'positive' | 'negative' | 'neutral';
  confidence: number;
}

interface SectorSentiment {
  sector: string;
  sentiment: SentimentResult;
  newsCount: number;
  trendDirection: 'up' | 'down' | 'neutral';
  topKeywords: string[];
}

class SentimentAnalysisService {
  private finbert: FinBERT;
  private newsApi: NewsAPI;
  private cache: RedisCache;

  constructor() {
    this.finbert = new FinBERT();
    this.newsApi = new NewsAPI();
    this.cache = new RedisCache();
  }

  async analyzeMarketSentiment(): Promise<{
    overall: SentimentResult;
    sectors: SectorSentiment[];
  }> {
    try {
      // Check cache first
      const cachedResult = await this.cache.get('market_sentiment');
      if (cachedResult) {
        return JSON.parse(cachedResult);
      }

      // Fetch news for each sector
      const sectors = ['Technology', 'Finance', 'Healthcare', 'Energy', 'Consumer'];
      const sectorSentiments = await Promise.all(
        sectors.map(async (sector) => {
          const news = await this.newsApi.getNewsBySector(sector);
          const sentiments = await this.finbert.analyzeBatch(
            news.map((article) => article.title + ' ' + article.description)
          );

          // Aggregate sentiments
          const aggregatedSentiment = this.aggregateSentiments(sentiments);
          const keywords = this.extractKeywords(news);

          return {
            sector,
            sentiment: aggregatedSentiment,
            newsCount: news.length,
            trendDirection: this.determineTrend(sentiments),
            topKeywords: keywords.slice(0, 3),
          };
        })
      );

      // Calculate overall market sentiment
      const overall = this.calculateOverallSentiment(sectorSentiments);

      const result = {
        overall,
        sectors: sectorSentiments,
      };

      // Cache the result
      await this.cache.set('market_sentiment', JSON.stringify(result), 3600); // 1 hour

      return result;
    } catch (error) {
      console.error('Error in market sentiment analysis:', error);
      throw new Error('Failed to analyze market sentiment');
    }
  }

  async analyzeStockSentiment(symbol: string): Promise<{
    sentiment: SentimentResult;
    news: Array<{
      title: string;
      sentiment: SentimentResult;
      url: string;
    }>;
  }> {
    try {
      const cacheKey = `stock_sentiment_${symbol}`;
      const cachedResult = await this.cache.get(cacheKey);
      if (cachedResult) {
        return JSON.parse(cachedResult);
      }

      const news = await this.newsApi.getNewsBySymbol(symbol);
      const sentiments = await this.finbert.analyzeBatch(
        news.map((article) => article.title)
      );

      const result = {
        sentiment: this.aggregateSentiments(sentiments),
        news: news.map((article, i) => ({
          title: article.title,
          sentiment: sentiments[i],
          url: article.url,
        })),
      };

      await this.cache.set(cacheKey, JSON.stringify(result), 1800); // 30 minutes
      return result;
    } catch (error) {
      console.error(`Error in stock sentiment analysis for ${symbol}:`, error);
      throw new Error(`Failed to analyze sentiment for ${symbol}`);
    }
  }

  private aggregateSentiments(sentiments: SentimentResult[]): SentimentResult {
    const total = sentiments.reduce(
      (acc, curr) => acc + curr.score * curr.confidence,
      0
    );
    const avgScore = total / sentiments.length;

    return {
      score: avgScore,
      label: this.getLabel(avgScore),
      confidence: sentiments.reduce((acc, curr) => acc + curr.confidence, 0) / sentiments.length,
    };
  }

  private getLabel(score: number): 'positive' | 'negative' | 'neutral' {
    if (score > 0.6) return 'positive';
    if (score < 0.4) return 'negative';
    return 'neutral';
  }

  private determineTrend(
    sentiments: SentimentResult[]
  ): 'up' | 'down' | 'neutral' {
    const recentSentiments = sentiments.slice(-5);
    const avgRecent =
      recentSentiments.reduce((acc, curr) => acc + curr.score, 0) /
      recentSentiments.length;
    const avgAll =
      sentiments.reduce((acc, curr) => acc + curr.score, 0) / sentiments.length;

    if (avgRecent > avgAll + 0.1) return 'up';
    if (avgRecent < avgAll - 0.1) return 'down';
    return 'neutral';
  }

  private extractKeywords(news: any[]): string[] {
    // Implement keyword extraction logic
    // This could use TF-IDF, NER, or other NLP techniques
    return ['AI', 'Cloud', 'Cybersecurity']; // Placeholder
  }

  private calculateOverallSentiment(
    sectorSentiments: SectorSentiment[]
  ): SentimentResult {
    const weights = {
      Technology: 0.25,
      Finance: 0.25,
      Healthcare: 0.2,
      Energy: 0.15,
      Consumer: 0.15,
    };

    const weightedScore = sectorSentiments.reduce((acc, sector) => {
      return acc + sector.sentiment.score * weights[sector.sector as keyof typeof weights];
    }, 0);

    return {
      score: weightedScore,
      label: this.getLabel(weightedScore),
      confidence: sectorSentiments.reduce((acc, s) => acc + s.sentiment.confidence, 0) / sectorSentiments.length,
    };
  }
}

export default new SentimentAnalysisService(); 