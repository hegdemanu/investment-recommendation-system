import { DataTypes } from 'sequelize';
import bcrypt from 'bcryptjs';
import sequelize from '../config/database.js';

const User = sequelize.define('User', {
  id: {
    type: DataTypes.INTEGER,
    primaryKey: true,
    autoIncrement: true
  },
  name: {
    type: DataTypes.STRING(100),
    allowNull: false
  },
  email: {
    type: DataTypes.STRING(100),
    allowNull: false,
    unique: true,
    validate: {
      isEmail: true
    }
  },
  password: {
    type: DataTypes.STRING(100),
    allowNull: false
  },
  role: {
    type: DataTypes.ENUM('user', 'admin'),
    allowNull: false,
    defaultValue: 'user'
  },
  profilePicture: {
    type: DataTypes.STRING(255),
    allowNull: true
  },
  riskProfile: {
    type: DataTypes.ENUM('conservative', 'moderate', 'aggressive'),
    allowNull: false,
    defaultValue: 'moderate'
  },
  investmentGoals: {
    type: DataTypes.TEXT,
    allowNull: true
  },
  investmentHorizon: {
    type: DataTypes.ENUM('short_term', 'medium_term', 'long_term'),
    allowNull: true,
    defaultValue: 'medium_term'
  },
  notificationPreferences: {
    type: DataTypes.JSONB,
    allowNull: true,
    defaultValue: {
      email: true,
      push: false,
      priceAlerts: true,
      newsAlerts: true,
      recommendationAlerts: true
    }
  },
  lastLogin: {
    type: DataTypes.DATE,
    allowNull: true
  },
  isActive: {
    type: DataTypes.BOOLEAN,
    allowNull: false,
    defaultValue: true
  }
}, {
  tableName: 'users',
  timestamps: true,
  indexes: [
    {
      unique: true,
      fields: ['email']
    }
  ],
  hooks: {
    beforeCreate: async (user) => {
      if (user.password) {
        const salt = await bcrypt.genSalt(10);
        user.password = await bcrypt.hash(user.password, salt);
      }
    },
    beforeUpdate: async (user) => {
      if (user.changed('password')) {
        const salt = await bcrypt.genSalt(10);
        user.password = await bcrypt.hash(user.password, salt);
      }
    }
  }
});

// Instance method to check password
User.prototype.matchPassword = async function(enteredPassword) {
  return await bcrypt.compare(enteredPassword, this.password);
};

export default User; 