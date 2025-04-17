import { DataTypes } from 'sequelize';
import sequelize from '../config/database.js';
import Stock from './stock.model.js';

const NewsSentiment = sequelize.define('NewsSentiment', {
  id: {
    type: DataTypes.INTEGER,
    primaryKey: true,
    autoIncrement: true
  },
  stockId: {
    type: DataTypes.INTEGER,
    allowNull: false,
    references: {
      model: 'stocks',
      key: 'id'
    }
  },
  headline: {
    type: DataTypes.TEXT,
    allowNull: false
  },
  summary: {
    type: DataTypes.TEXT,
    allowNull: true
  },
  url: {
    type: DataTypes.STRING(2048),
    allowNull: true,
    validate: {
      isUrl: true
    }
  },
  publishedAt: {
    type: DataTypes.DATE,
    allowNull: false
  },
  source: {
    type: DataTypes.STRING(255),
    allowNull: true
  },
  sentiment: {
    type: DataTypes.ENUM('positive', 'negative', 'neutral'),
    allowNull: false
  },
  sentimentScore: {
    type: DataTypes.DECIMAL(5, 2),
    allowNull: false,
    validate: {
      min: -1,
      max: 1
    },
    comment: 'Score from -1 to 1 where -1 is very negative, 0 is neutral, 1 is very positive'
  },
  impactScore: {
    type: DataTypes.DECIMAL(5, 2),
    allowNull: true,
    validate: {
      min: 0,
      max: 10
    },
    comment: 'Estimated impact on stock price (0-10 scale)'
  },
  keywords: {
    type: DataTypes.ARRAY(DataTypes.STRING),
    allowNull: true
  },
  categories: {
    type: DataTypes.ARRAY(DataTypes.STRING),
    allowNull: true,
    comment: 'News categories like "earnings", "acquisition", "market", etc.'
  }
}, {
  tableName: 'news_sentiments',
  timestamps: true,
  indexes: [
    {
      fields: ['stockId']
    },
    {
      fields: ['publishedAt']
    },
    {
      fields: ['sentiment']
    },
    {
      fields: ['stockId', 'publishedAt']
    }
  ]
});

// Define associations
NewsSentiment.belongsTo(Stock, {
  foreignKey: 'stockId',
  as: 'stock'
});

Stock.hasMany(NewsSentiment, {
  foreignKey: 'stockId',
  as: 'newsSentiments'
});

/**
 * After model is initialized, we need to create a hypertable for it
 * This should be called after the table is created
 * @param {Object} sequelizeInstance - The Sequelize instance
 */
export const createNewsSentimentHypertable = async (sequelizeInstance) => {
  try {
    await sequelizeInstance.query(`
      SELECT create_hypertable('news_sentiments', 'published_at', 
        if_not_exists => TRUE,
        chunk_time_interval => INTERVAL '7 days'
      );
    `);
    console.log('Created hypertable for news_sentiments on published_at column.');
    
    // Create time-based compression policy
    await sequelizeInstance.query(`
      ALTER TABLE news_sentiments SET (
        timescaledb.compress,
        timescaledb.compress_segmentby = 'stock_id,sentiment'
      );
    `);
    
    // Create compression policy - compress data older than 30 days
    await sequelizeInstance.query(`
      SELECT add_compression_policy('news_sentiments', INTERVAL '30 days');
    `);
    
    console.log('Compression policy added for news_sentiments table.');
    return true;
  } catch (error) {
    console.error('Error creating hypertable for news_sentiments:', error);
    // This is not a fatal error, so we'll just log it and continue
    return false;
  }
};

export default NewsSentiment; 