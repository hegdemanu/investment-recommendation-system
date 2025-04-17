import { DataTypes } from 'sequelize';
import sequelize from '../config/database.js';

const Stock = sequelize.define('Stock', {
  id: {
    type: DataTypes.INTEGER,
    primaryKey: true,
    autoIncrement: true
  },
  symbol: {
    type: DataTypes.STRING(10),
    allowNull: false,
    unique: true,
    validate: {
      notEmpty: true
    }
  },
  name: {
    type: DataTypes.STRING(100),
    allowNull: false,
    validate: {
      notEmpty: true
    }
  },
  sector: {
    type: DataTypes.STRING(50),
    allowNull: true
  },
  industry: {
    type: DataTypes.STRING(100),
    allowNull: true
  },
  exchange: {
    type: DataTypes.STRING(20),
    allowNull: true
  },
  currentPrice: {
    type: DataTypes.DECIMAL(12, 2),
    allowNull: false,
    validate: {
      min: 0
    }
  },
  previousClose: {
    type: DataTypes.DECIMAL(12, 2),
    allowNull: true
  },
  change: {
    type: DataTypes.DECIMAL(12, 2),
    allowNull: true
  },
  changePercent: {
    type: DataTypes.DECIMAL(8, 2),
    allowNull: true
  },
  marketCap: {
    type: DataTypes.BIGINT,
    allowNull: true
  },
  volume: {
    type: DataTypes.BIGINT,
    allowNull: true
  },
  fiftyTwoWeekHigh: {
    type: DataTypes.DECIMAL(12, 2),
    allowNull: true
  },
  fiftyTwoWeekLow: {
    type: DataTypes.DECIMAL(12, 2),
    allowNull: true
  },
  dividendYield: {
    type: DataTypes.DECIMAL(5, 2),
    allowNull: true
  },
  beta: {
    type: DataTypes.DECIMAL(5, 2),
    allowNull: true
  },
  peRatio: {
    type: DataTypes.DECIMAL(10, 2),
    allowNull: true
  },
  description: {
    type: DataTypes.TEXT,
    allowNull: true
  },
  logoUrl: {
    type: DataTypes.STRING(255),
    allowNull: true
  },
  websiteUrl: {
    type: DataTypes.STRING(255),
    allowNull: true
  },
  isActive: {
    type: DataTypes.BOOLEAN,
    allowNull: false,
    defaultValue: true
  }
}, {
  tableName: 'stocks',
  timestamps: true,
  indexes: [
    {
      unique: true,
      fields: ['symbol']
    },
    {
      fields: ['sector']
    },
    {
      fields: ['industry']
    }
  ]
});

export default Stock; 