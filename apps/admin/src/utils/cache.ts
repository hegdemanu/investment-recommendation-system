import Redis from 'ioredis';

export class RedisCache {
  private client: Redis;

  constructor(url = process.env.REDIS_URL) {
    this.client = new Redis(url || 'redis://localhost:6379');
  }

  async get(key: string): Promise<string | null> {
    return this.client.get(key);
  }

  async set(key: string, value: string, expireSeconds: number): Promise<void> {
    await this.client.setex(key, expireSeconds, value);
  }
} 