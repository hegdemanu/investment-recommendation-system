import sequelize from '../config/database.js';
import { initializeTimescaleDB } from '../config/database.js';

// Import models
import User from './user.model.js';
import Stock from './stock.model.js';
import StockPrice, { createStockPriceHypertable } from './stockPrice.model.js';
import Portfolio from './portfolio.model.js';
import PortfolioItem from './portfolioItem.model.js';
import Recommendation from './recommendation.model.js';
import NewsSentiment, { createNewsSentimentHypertable } from './newsSentiment.model.js';

// Define model relationships

// User - Portfolio relationship (one-to-many)
User.hasMany(Portfolio, {
  foreignKey: 'userId',
  as: 'portfolios',
  onDelete: 'CASCADE'
});
Portfolio.belongsTo(User, {
  foreignKey: 'userId',
  as: 'user'
});

// Portfolio - PortfolioItem relationship (one-to-many)
Portfolio.hasMany(PortfolioItem, {
  foreignKey: 'portfolioId',
  as: 'items',
  onDelete: 'CASCADE'
});
PortfolioItem.belongsTo(Portfolio, {
  foreignKey: 'portfolioId',
  as: 'portfolio'
});

// Stock - PortfolioItem relationship (one-to-many)
Stock.hasMany(PortfolioItem, {
  foreignKey: 'stockId',
  as: 'portfolioItems',
  onDelete: 'CASCADE'
});
PortfolioItem.belongsTo(Stock, {
  foreignKey: 'stockId',
  as: 'stock'
});

// Stock - StockPrice relationship (one-to-many)
Stock.hasMany(StockPrice, {
  foreignKey: 'stockId',
  as: 'prices',
  onDelete: 'CASCADE'
});
StockPrice.belongsTo(Stock, {
  foreignKey: 'stockId',
  as: 'stock'
});

// Stock - Recommendation relationship (one-to-many)
Stock.hasMany(Recommendation, {
  foreignKey: 'stockId',
  as: 'recommendations',
  onDelete: 'CASCADE'
});
Recommendation.belongsTo(Stock, {
  foreignKey: 'stockId',
  as: 'stock'
});

// Stock - NewsSentiment relationship (one-to-many)
Stock.hasMany(NewsSentiment, {
  foreignKey: 'stockId',
  as: 'newsSentiments',
  onDelete: 'CASCADE'
});
NewsSentiment.belongsTo(Stock, {
  foreignKey: 'stockId',
  as: 'stock'
});

// Initialize TimescaleDB and hypertables
export const initializeModels = async () => {
  try {
    // Initialize TimescaleDB extension
    await initializeTimescaleDB();
    
    // After database sync, create hypertables
    await createStockPriceHypertable(sequelize);
    await createNewsSentimentHypertable(sequelize);
    
    return true;
  } catch (error) {
    console.error('Error initializing models:', error);
    return false;
  }
};

// Export all models
export {
  User,
  Stock,
  StockPrice,
  Portfolio,
  PortfolioItem,
  Recommendation,
  NewsSentiment
};

export default {
  User,
  Stock,
  StockPrice,
  Portfolio,
  PortfolioItem,
  Recommendation,
  NewsSentiment,
  sequelize
}; 