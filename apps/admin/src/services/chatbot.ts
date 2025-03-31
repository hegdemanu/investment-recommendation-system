import { OpenAI } from '../integrations/openai';
import { VectorDB } from '../utils/vectordb';
import { MarketData } from '../integrations/marketdata';
import { RedisCache } from '../utils/cache';

interface ChatMessage {
  role: 'user' | 'assistant' | 'system';
  content: string;
}

interface ChatResponse {
  message: string;
  context?: {
    marketData?: any;
    relatedInsights?: any[];
    confidence: number;
  };
}

class ChatbotService {
  private openai: OpenAI;
  private vectorDb: VectorDB;
  private marketData: MarketData;
  private cache: RedisCache;

  constructor() {
    this.openai = new OpenAI();
    this.vectorDb = new VectorDB();
    this.marketData = new MarketData();
    this.cache = new RedisCache();
  }

  async processMessage(
    message: string,
    history: ChatMessage[],
    mode: 'beginner' | 'expert' = 'beginner'
  ): Promise<ChatResponse> {
    try {
      // Extract entities and intent
      const { entities, intent } = await this.analyzeMessage(message);

      // Fetch relevant context
      const context = await this.getContext(entities, intent, mode);

      // Generate system message based on mode
      const systemMessage = this.getSystemMessage(mode);

      // Prepare messages for LLM
      const messages: ChatMessage[] = [
        { role: 'system', content: systemMessage },
        ...history,
        { role: 'user', content: message },
      ];

      // If we have context, add it to the messages
      if (context) {
        messages.push({
          role: 'system',
          content: `Here's some relevant information: ${JSON.stringify(context)}`,
        });
      }

      // Generate response
      const response = await this.openai.chat(messages);

      return {
        message: response,
        context: {
          marketData: context.marketData,
          relatedInsights: context.insights,
          confidence: context.confidence,
        },
      };
    } catch (error) {
      console.error('Error in chatbot service:', error);
      throw new Error('Failed to process message');
    }
  }

  private async analyzeMessage(message: string): Promise<{
    entities: string[];
    intent: string;
  }> {
    // Use NLP to extract entities (stock symbols, sectors, etc.) and intent
    const analysis = await this.openai.analyze(message);
    return {
      entities: analysis.entities,
      intent: analysis.intent,
    };
  }

  private async getContext(
    entities: string[],
    intent: string,
    mode: 'beginner' | 'expert'
  ): Promise<{
    marketData: any;
    insights: any[];
    confidence: number;
  }> {
    try {
      // Fetch market data for relevant entities
      const marketData = await Promise.all(
        entities.map((entity) => this.marketData.getData(entity))
      );

      // Search vector database for relevant insights
      const insights = await this.vectorDb.search(
        entities.join(' ') + ' ' + intent,
        mode === 'expert' ? 5 : 3
      );

      // Calculate confidence score based on data freshness and relevance
      const confidence = this.calculateConfidence(marketData, insights);

      return {
        marketData,
        insights,
        confidence,
      };
    } catch (error) {
      console.error('Error fetching context:', error);
      return {
        marketData: [],
        insights: [],
        confidence: 0,
      };
    }
  }

  private getSystemMessage(mode: 'beginner' | 'expert'): string {
    if (mode === 'beginner') {
      return `You are an AI investment assistant helping beginners understand the market. 
        Explain concepts in simple terms and provide educational context. 
        Focus on basic investment principles and risk management.`;
    }

    return `You are an AI investment assistant for experienced traders. 
      Provide detailed technical analysis and advanced market insights. 
      Include specific data points and sophisticated trading strategies.`;
  }

  private calculateConfidence(marketData: any[], insights: any[]): number {
    // Calculate confidence score based on:
    // 1. Data freshness
    // 2. Number of relevant insights
    // 3. Market data completeness
    const dataFreshness = this.calculateDataFreshness(marketData);
    const insightRelevance = insights.length / 5; // Normalize to 0-1
    const dataCompleteness = marketData.filter(Boolean).length / marketData.length;

    return (dataFreshness + insightRelevance + dataCompleteness) / 3;
  }

  private calculateDataFreshness(data: any[]): number {
    const now = Date.now();
    const freshness = data.map((item) => {
      if (!item || !item.timestamp) return 0;
      const age = now - new Date(item.timestamp).getTime();
      const maxAge = 24 * 60 * 60 * 1000; // 24 hours
      return Math.max(0, 1 - age / maxAge);
    });

    return freshness.reduce((acc, val) => acc + val, 0) / freshness.length;
  }
}

export default new ChatbotService(); 