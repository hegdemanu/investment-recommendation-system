import { DataTypes } from 'sequelize';
import sequelize from '../config/database.js';
import Portfolio from './portfolio.model.js';
import Stock from './stock.model.js';

const PortfolioItem = sequelize.define('PortfolioItem', {
  id: {
    type: DataTypes.INTEGER,
    primaryKey: true,
    autoIncrement: true
  },
  portfolioId: {
    type: DataTypes.INTEGER,
    allowNull: false,
    references: {
      model: 'portfolios',
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
  quantity: {
    type: DataTypes.DECIMAL(15, 6),
    allowNull: false,
    validate: {
      min: 0
    }
  },
  averageCostPerShare: {
    type: DataTypes.DECIMAL(15, 6),
    allowNull: false,
    validate: {
      min: 0
    }
  },
  currentValue: {
    type: DataTypes.DECIMAL(15, 2),
    allowNull: true
  },
  totalCost: {
    type: DataTypes.DECIMAL(15, 2),
    allowNull: true
  },
  unrealizedGainLoss: {
    type: DataTypes.DECIMAL(15, 2),
    allowNull: true
  },
  unrealizedGainLossPercent: {
    type: DataTypes.DECIMAL(8, 2),
    allowNull: true
  },
  weight: {
    type: DataTypes.DECIMAL(5, 2),
    allowNull: true,
    comment: 'Percentage weight in the portfolio'
  },
  targetWeight: {
    type: DataTypes.DECIMAL(5, 2),
    allowNull: true,
    comment: 'Target percentage weight in the portfolio'
  },
  notes: {
    type: DataTypes.TEXT,
    allowNull: true
  }
}, {
  tableName: 'portfolio_items',
  timestamps: true,
  indexes: [
    {
      fields: ['portfolioId']
    },
    {
      fields: ['stockId']
    },
    {
      unique: true,
      fields: ['portfolioId', 'stockId']
    }
  ],
  hooks: {
    afterCreate: async (item) => {
      await updatePortfolioValues(item.portfolioId);
    },
    afterUpdate: async (item) => {
      await updatePortfolioValues(item.portfolioId);
    },
    afterDestroy: async (item) => {
      await updatePortfolioValues(item.portfolioId);
    }
  }
});

// Define associations
PortfolioItem.belongsTo(Portfolio, {
  foreignKey: 'portfolioId',
  as: 'portfolio'
});

PortfolioItem.belongsTo(Stock, {
  foreignKey: 'stockId',
  as: 'stock'
});

Portfolio.hasMany(PortfolioItem, {
  foreignKey: 'portfolioId',
  as: 'items'
});

Stock.hasMany(PortfolioItem, {
  foreignKey: 'stockId',
  as: 'portfolioItems'
});

// Function to update portfolio values
const updatePortfolioValues = async (portfolioId) => {
  try {
    // Get all items for this portfolio
    const items = await PortfolioItem.findAll({
      where: { portfolioId },
      include: [{
        model: Stock,
        as: 'stock',
        attributes: ['currentPrice']
      }]
    });

    // Calculate totals
    let totalValue = 0;
    let totalCost = 0;

    items.forEach(item => {
      // Calculate current value
      const currentValue = parseFloat(item.quantity) * parseFloat(item.stock.currentPrice);
      
      // Calculate total cost
      const totalItemCost = parseFloat(item.quantity) * parseFloat(item.averageCostPerShare);
      
      // Update item
      item.currentValue = currentValue;
      item.totalCost = totalItemCost;
      item.unrealizedGainLoss = currentValue - totalItemCost;
      item.unrealizedGainLossPercent = totalItemCost > 0 
        ? ((currentValue - totalItemCost) / totalItemCost) * 100 
        : 0;
      
      // Add to totals
      totalValue += currentValue;
      totalCost += totalItemCost;
    });

    // Update all items with weights
    const updatePromises = items.map(item => {
      item.weight = totalValue > 0 ? (parseFloat(item.currentValue) / totalValue) * 100 : 0;
      return item.save();
    });

    // Wait for all item updates
    await Promise.all(updatePromises);

    // Update portfolio
    await Portfolio.update({
      totalValue,
      totalCost
    }, {
      where: { id: portfolioId }
    });

    return true;
  } catch (error) {
    console.error('Error updating portfolio values:', error);
    return false;
  }
};

export default PortfolioItem; 