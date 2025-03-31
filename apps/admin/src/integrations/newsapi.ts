import axios from 'axios';

interface NewsArticle {
  title: string;
  description: string;
  url: string;
}

export class NewsAPI {
  private readonly apiKey: string;
  private readonly baseUrl = 'https://newsapi.org/v2';

  constructor(apiKey = process.env.NEWS_API_KEY) {
    this.apiKey = apiKey || '';
  }

  async getNewsBySector(sector: string): Promise<NewsArticle[]> {
    const response = await axios.get(`${this.baseUrl}/everything`, {
      params: {
        q: sector,
        apiKey: this.apiKey,
        language: 'en',
        sortBy: 'publishedAt'
      }
    });
    return response.data.articles;
  }

  async getNewsBySymbol(symbol: string): Promise<NewsArticle[]> {
    const response = await axios.get(`${this.baseUrl}/everything`, {
      params: {
        q: symbol,
        apiKey: this.apiKey,
        language: 'en',
        sortBy: 'publishedAt'
      }
    });
    return response.data.articles;
  }
} 