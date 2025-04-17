import { DataTypes } from 'sequelize';
import sequelize from '../config/database.js';
import User from './user.model.js';

const Portfolio = sequelize.define('Portfolio', {
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
  name: {
    type: DataTypes.STRING(100),
    allowNull: false,
    validate: {
      notEmpty: true
    }
  },
  description: {
    type: DataTypes.TEXT,
    allowNull: true
  },
  isDefault: {
    type: DataTypes.BOOLEAN,
    allowNull: false,
    defaultValue: false
  },
  riskProfile: {
    type: DataTypes.ENUM('conservative', 'moderate', 'aggressive'),
    allowNull: false,
    defaultValue: 'moderate'
  },
  targetReturn: {
    type: DataTypes.DECIMAL(5, 2),
    allowNull: true,
    comment: 'Target return percentage'
  },
  investmentHorizon: {
    type: DataTypes.ENUM('short_term', 'medium_term', 'long_term'),
    allowNull: false,
    defaultValue: 'medium_term'
  },
  totalValue: {
    type: DataTypes.DECIMAL(15, 2),
    allowNull: false,
    defaultValue: 0.00
  },
  totalCost: {
    type: DataTypes.DECIMAL(15, 2),
    allowNull: false,
    defaultValue: 0.00
  },
  cash: {
    type: DataTypes.DECIMAL(15, 2),
    allowNull: false,
    defaultValue: 0.00
  },
  currency: {
    type: DataTypes.STRING(3),
    allowNull: false,
    defaultValue: 'USD'
  },
  isActive: {
    type: DataTypes.BOOLEAN,
    allowNull: false,
    defaultValue: true
  }
}, {
  tableName: 'portfolios',
  timestamps: true,
  indexes: [
    {
      fields: ['userId']
    },
    {
      fields: ['isDefault']
    },
    {
      fields: ['riskProfile']
    },
    {
      unique: true,
      fields: ['userId', 'name']
    }
  ]
});

// Define associations
Portfolio.belongsTo(User, {
  foreignKey: 'userId',
  as: 'user'
});

User.hasMany(Portfolio, {
  foreignKey: 'userId',
  as: 'portfolios'
});

export default Portfolio; 