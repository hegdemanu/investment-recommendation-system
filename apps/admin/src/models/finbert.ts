import axios from 'axios';

export interface FinBERTResponse {
  score: number;
  label: 'positive' | 'negative' | 'neutral';
  confidence: number;
}

export class FinBERT {
  private readonly apiEndpoint: string;

  constructor(endpoint = process.env.FINBERT_API_ENDPOINT || 'http://localhost:8000/analyze') {
    this.apiEndpoint = endpoint;
  }

  async analyzeBatch(texts: string[]): Promise<FinBERTResponse[]> {
    try {
      const response = await axios.post(this.apiEndpoint, { texts });
      return response.data;
    } catch (error) {
      console.error('FinBERT analysis error:', error);
      throw new Error('Failed to analyze text with FinBERT');
    }
  }
} 