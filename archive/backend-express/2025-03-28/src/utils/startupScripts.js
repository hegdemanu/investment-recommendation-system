import { initializeDatabase } from './dbInit.js';
import { sequelize } from '../config/database.js';
import { initializeModels } from '../models/index.js';

/**
 * Run all startup scripts in sequence
 */
export const runStartupScripts = async () => {
  try {
    // 1. Check database connection
    console.log('Checking database connection...');
    await sequelize.authenticate();
    console.log('Database connection established successfully.');
    
    // 2. Sync database models (in production, use migrations instead)
    console.log('Syncing database models...');
    if (process.env.NODE_ENV === 'development') {
      // In development, we can use force:true to reset the database
      // WARNING: This will drop all tables and recreate them
      // Only use in development environment
      const forceSync = process.env.FORCE_DB_SYNC === 'true';
      await sequelize.sync({ force: forceSync });
      console.log(`Database models synchronized ${forceSync ? 'with force' : 'successfully'}.`);
    } else {
      // In production, just sync without force
      await sequelize.sync();
      console.log('Database models synchronized successfully.');
    }
    
    // 3. Initialize TimescaleDB and create hypertables
    console.log('Initializing TimescaleDB and creating hypertables...');
    await initializeModels();
    
    // 4. Initialize database with seed data
    console.log('Initializing database with seed data...');
    await initializeDatabase();
    
    // 5. Any other startup tasks
    console.log('Running additional startup tasks...');
    // Add any other startup tasks here
    
    console.log('All startup scripts completed successfully.');
    return true;
  } catch (error) {
    console.error('Error running startup scripts:', error);
    return false;
  }
};

export default runStartupScripts; 