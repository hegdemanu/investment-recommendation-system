import { DataTypes } from 'sequelize';
import sequelize from '../config/database.js';
import Stock from './stock.model.js';

const StockPrice = sequelize.define('StockPrice', {
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
  timestamp: {
    type: DataTypes.DATE,
    allowNull: false
  },
  open: {
    type: DataTypes.DECIMAL(12, 4),
    allowNull: false
  },
  high: {
    type: DataTypes.DECIMAL(12, 4),
    allowNull: false
  },
  low: {
    type: DataTypes.DECIMAL(12, 4),
    allowNull: false
  },
  close: {
    type: DataTypes.DECIMAL(12, 4),
    allowNull: false
  },
  adjustedClose: {
    type: DataTypes.DECIMAL(12, 4),
    allowNull: true
  },
  volume: {
    type: DataTypes.BIGINT,
    allowNull: true
  },
  interval: {
    type: DataTypes.ENUM('1min', '5min', '15min', '30min', '1hour', 'daily', 'weekly', 'monthly'),
    allowNull: false,
    defaultValue: 'daily'
  },
  source: {
    type: DataTypes.STRING(50),
    allowNull: true,
    defaultValue: 'API'
  }
}, {
  tableName: 'stock_prices',
  timestamps: true,
  indexes: [
    {
      fields: ['stockId']
    },
    {
      fields: ['timestamp']
    },
    {
      fields: ['interval']
    },
    {
      fields: ['stockId', 'timestamp', 'interval'],
      unique: true
    }
  ]
});

// Define association with Stock model
StockPrice.belongsTo(Stock, {
  foreignKey: 'stockId',
  as: 'stock'
});

Stock.hasMany(StockPrice, {
  foreignKey: 'stockId',
  as: 'prices'
});

/**
 * After model is initialized, we need to create a hypertable for it
 * This should be called after the table is created
 * @param {Object} sequelizeInstance - The Sequelize instance
 */
export const createStockPriceHypertable = async (sequelizeInstance) => {
  try {
    await sequelizeInstance.query(`
      SELECT create_hypertable('stock_prices', 'timestamp', 
        if_not_exists => TRUE,
        chunk_time_interval => INTERVAL '7 days'
      );
    `);
    console.log('Created hypertable for stock_prices on timestamp column.');
    
    // Create time-based compression policy
    await sequelizeInstance.query(`
      ALTER TABLE stock_prices SET (
        timescaledb.compress,
        timescaledb.compress_segmentby = 'stock_id,interval'
      );
    `);
    
    // Create compression policy - compress data older than 7 days
    await sequelizeInstance.query(`
      SELECT add_compression_policy('stock_prices', INTERVAL '7 days');
    `);
    
    console.log('Compression policy added for stock_prices table.');
    return true;
  } catch (error) {
    console.error('Error creating hypertable for stock_prices:', error);
    // This is not a fatal error, so we'll just log it and continue
    return false;
  }
};

export default StockPrice; 