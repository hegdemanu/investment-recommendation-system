import { DataTypes } from 'sequelize';
import sequelize from '../config/database.js';
import Stock from './stock.model.js';
import User from './user.model.js';

const Recommendation = sequelize.define('Recommendation', {
  id: {
    type: DataTypes.INTEGER,
    primaryKey: true,
    autoIncrement: true
  },
  userId: {
    type: DataTypes.INTEGER,
    allowNull: false,
    references: {
      model: 'users',
      key: 'id'
    }
  },
  stockId: {
    type: DataTypes.INTEGER,
    allowNull: false,
    references: {
      model: 'stocks',
      key: 'id'
    }
  },
  type: {
    type: DataTypes.ENUM('buy', 'sell', 'hold'),
    allowNull: false
  },
  confidence: {
    type: DataTypes.DECIMAL(5, 2),
    allowNull: false,
    validate: {
      min: 0,
      max: 100
    }
  },
  targetPrice: {
    type: DataTypes.DECIMAL(12, 2),
    allowNull: true
  },
  stopLoss: {
    type: DataTypes.DECIMAL(12, 2),
    allowNull: true
  },
  timeHorizon: {
    type: DataTypes.ENUM('short_term', 'medium_term', 'long_term'),
    allowNull: false,
    defaultValue: 'medium_term'
  },
  reasoning: {
    type: DataTypes.TEXT,
    allowNull: true
  },
  riskLevel: {
    type: DataTypes.ENUM('low', 'moderate', 'high'),
    allowNull: false,
    defaultValue: 'moderate'
  },
  expectedReturn: {
    type: DataTypes.DECIMAL(6, 2),
    allowNull: true,
    comment: 'Expected return percentage'
  },
  status: {
    type: DataTypes.ENUM('active', 'expired', 'successful', 'unsuccessful'),
    allowNull: false,
    defaultValue: 'active'
  },
  modelVersion: {
    type: DataTypes.STRING(50),
    allowNull: true,
    comment: 'Version of the ML model used for this recommendation'
  },
  expiryDate: {
    type: DataTypes.DATE,
    allowNull: true
  }
}, {
  tableName: 'recommendations',
  timestamps: true,
  indexes: [
    {
      fields: ['userId']
    },
    {
      fields: ['stockId']
    },
    {
      fields: ['type']
    },
    {
      fields: ['timeHorizon']
    },
    {
      fields: ['status']
    },
    {
      fields: ['createdAt']
    }
  ]
});

// Define associations
Recommendation.belongsTo(User, {
  foreignKey: 'userId',
  as: 'user'
});

Recommendation.belongsTo(Stock, {
  foreignKey: 'stockId',
  as: 'stock'
});

User.hasMany(Recommendation, {
  foreignKey: 'userId',
  as: 'recommendations'
});

Stock.hasMany(Recommendation, {
  foreignKey: 'stockId',
  as: 'recommendations'
});

export default Recommendation; 