import { Sequelize } from 'sequelize';
import dotenv from 'dotenv';

// Load environment variables
dotenv.config();

// Database configuration
const DB_NAME = process.env.DB_NAME || 'investment_system';
const DB_USER = process.env.DB_USER || 'postgres';
const DB_PASSWORD = process.env.DB_PASSWORD || 'postgres';
const DB_HOST = process.env.DB_HOST || 'localhost';
const DB_PORT = process.env.DB_PORT || 5432;
const DB_DIALECT = 'postgres';

// Create Sequelize instance
export const sequelize = new Sequelize(DB_NAME, DB_USER, DB_PASSWORD, {
  host: DB_HOST,
  port: DB_PORT,
  dialect: DB_DIALECT,
  logging: process.env.NODE_ENV === 'development' ? console.log : false,
  pool: {
    max: 10,
    min: 0,
    acquire: 30000,
    idle: 10000
  },
  define: {
    timestamps: true,
    underscored: true
  }
});

/**
 * Initialize TimescaleDB extension for time-series data
 * This function should be called after the database connection is established
 */
export const initializeTimescaleDB = async () => {
  try {
    // Check if TimescaleDB extension exists
    const [result] = await sequelize.query(`
      SELECT EXISTS (
        SELECT 1 FROM pg_extension WHERE extname = 'timescaledb'
      );
    `);
    
    const extensionExists = result[0].exists;
    
    if (!extensionExists) {
      console.log('TimescaleDB extension not found. Attempting to create...');
      // Create TimescaleDB extension
      await sequelize.query('CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;');
      console.log('TimescaleDB extension created successfully.');
    } else {
      console.log('TimescaleDB extension already exists.');
    }
    
    return true;
  } catch (error) {
    console.error('Error initializing TimescaleDB:', error);
    // This is not a fatal error, so we'll just log it and continue
    return false;
  }
};

/**
 * Create a hypertable for time-series data
 * @param {string} tableName - The name of the table to convert to a hypertable
 * @param {string} timeColumn - The name of the timestamp column
 * @param {number} chunkTimeInterval - The time interval for chunks in days
 */
export const createHypertable = async (tableName, timeColumn, chunkTimeInterval = 7) => {
  try {
    // Convert the table to a hypertable
    await sequelize.query(`
      SELECT create_hypertable('${tableName}', '${timeColumn}', 
        if_not_exists => TRUE,
        chunk_time_interval => INTERVAL '${chunkTimeInterval} days'
      );
    `);
    console.log(`Created hypertable for ${tableName} on ${timeColumn} column.`);
    return true;
  } catch (error) {
    console.error(`Error creating hypertable for ${tableName}:`, error);
    return false;
  }
};

export default sequelize; 