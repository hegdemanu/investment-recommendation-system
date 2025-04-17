import { Stock, User } from '../models/index.js';
import bcrypt from 'bcryptjs';

// Seed stocks data
const seedStocks = async () => {
  try {
    const stockCount = await Stock.count();
    
    if (stockCount > 0) {
      console.log('Stocks already exist in the database. Skipping seed.');
      return;
    }
    
    // Sample stocks data
    const stocks = [
      { 
        symbol: 'AAPL', 
        name: 'Apple Inc.', 
        sector: 'Technology',
        industry: 'Consumer Electronics',
        exchange: 'NASDAQ',
        currentPrice: 189.84,
        previousClose: 187.00,
        change: 2.84,
        changePercent: 1.52,
        marketCap: 2950000000000,
        volume: 76540000,
        fiftyTwoWeekHigh: 198.23,
        fiftyTwoWeekLow: 124.17
      },
      { 
        symbol: 'MSFT', 
        name: 'Microsoft Corporation', 
        sector: 'Technology',
        industry: 'Software—Infrastructure',
        exchange: 'NASDAQ',
        currentPrice: 415.50,
        previousClose: 418.32,
        change: -2.82,
        changePercent: -0.67,
        marketCap: 3090000000000,
        volume: 21780000,
        fiftyTwoWeekHigh: 430.82,
        fiftyTwoWeekLow: 299.33
      },
      { 
        symbol: 'GOOGL', 
        name: 'Alphabet Inc.', 
        sector: 'Technology',
        industry: 'Internet Content & Information',
        exchange: 'NASDAQ',
        currentPrice: 173.45,
        previousClose: 170.99,
        change: 2.46,
        changePercent: 1.44,
        marketCap: 2130000000000,
        volume: 19250000,
        fiftyTwoWeekHigh: 174.45,
        fiftyTwoWeekLow: 115.35
      },
      { 
        symbol: 'AMZN', 
        name: 'Amazon.com Inc.', 
        sector: 'Consumer Cyclical',
        industry: 'Internet Retail',
        exchange: 'NASDAQ',
        currentPrice: 178.15,
        previousClose: 176.43,
        change: 1.72,
        changePercent: 0.97,
        marketCap: 1850000000000,
        volume: 32460000,
        fiftyTwoWeekHigh: 180.14,
        fiftyTwoWeekLow: 101.15
      },
      { 
        symbol: 'TSLA', 
        name: 'Tesla, Inc.', 
        sector: 'Consumer Cyclical',
        industry: 'Auto Manufacturers',
        exchange: 'NASDAQ',
        currentPrice: 142.05,
        previousClose: 147.05,
        change: -5.00,
        changePercent: -3.40,
        marketCap: 452000000000,
        volume: 96320000,
        fiftyTwoWeekHigh: 299.29,
        fiftyTwoWeekLow: 138.80
      }
    ];
    
    // Insert stocks
    await Stock.bulkCreate(stocks);
    console.log(`Successfully seeded ${stocks.length} stocks.`);
    
    return true;
  } catch (error) {
    console.error('Error seeding stocks:', error);
    return false;
  }
};

// Seed users data
const seedUsers = async () => {
  try {
    const userCount = await User.count();
    
    if (userCount > 0) {
      console.log('Users already exist in the database. Skipping seed.');
      return;
    }
    
    // Create admin user
    const adminPassword = 'admin123';
    const salt = await bcrypt.genSalt(10);
    const hashedPassword = await bcrypt.hash(adminPassword, salt);
    
    await User.create({
      name: 'Admin User',
      email: 'admin@example.com',
      password: hashedPassword,
      riskProfile: 'moderate',
      isActive: true
    });
    
    console.log('Successfully created admin user.');
    
    return true;
  } catch (error) {
    console.error('Error seeding users:', error);
    return false;
  }
};

// Initialize database with seed data
export const initializeDatabase = async () => {
  try {
    await seedStocks();
    await seedUsers();
    console.log('Database initialization complete.');
    return true;
  } catch (error) {
    console.error('Error initializing database:', error);
    return false;
  }
};

export default initializeDatabase; 