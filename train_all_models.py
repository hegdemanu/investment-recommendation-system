#!/usr/bin/env python3
"""
Train models for all available stock and mutual fund files.
This script:
1. Finds all stock CSV files in data/stocks/
2. Trains LSTM models for each stock
3. Stores models and metadata in the organized directory structure
4. Generates a training summary report
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import json
import time
import logging
from pathlib import Path
from src.model_trainer import ModelTrainer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("ModelTrainer")

# Configure directories
STOCKS_DIR = "./data/stocks"
MF_DIR = "./data/mutual_funds"
MODELS_DIR = "./models"
RESULTS_DIR = "./results"

# Create directories if they don't exist
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(os.path.join(MODELS_DIR, "stocks"), exist_ok=True)
os.makedirs(os.path.join(MODELS_DIR, "mutual_funds"), exist_ok=True)
os.makedirs(os.path.join(RESULTS_DIR, "training"), exist_ok=True)

def load_stock_data(filename):
    """Load and preprocess stock data from a CSV file."""
    logger.info(f"Loading stock data from {filename}")
    
    try:
        # Load data
        data = pd.read_csv(filename)
        
        # Convert date to datetime
        data['Date'] = pd.to_datetime(data['Date'])
        
        # Extract ticker from filename if not in data
        if 'ticker' not in data.columns:
            ticker = Path(filename).stem.split('_')[0]
            data['ticker'] = ticker
            
        # Process numeric columns with commas
        for col in ['Price', 'Open', 'High', 'Low']:
            if col in data.columns:
                if data[col].dtype == 'object':
                    data[col] = data[col].str.replace(',', '').astype(float)
                # Ensure column is float type
                data[col] = data[col].astype(float)
                
        # Process Vol. column with K, M suffixes
        if 'Vol.' in data.columns:
            if data['Vol.'].dtype == 'object':
                # Replace missing values with NaN
                data['Vol.'] = data['Vol.'].replace('-', np.nan).replace('', np.nan)
                
                # Process K and M suffixes and convert to numeric
                data.loc[data['Vol.'].notna(), 'Vol.'] = data.loc[data['Vol.'].notna(), 'Vol.'].apply(
                    lambda x: float(str(x).replace('K', '')) * 1000 if isinstance(x, str) and 'K' in x 
                    else float(str(x).replace('M', '')) * 1000000 if isinstance(x, str) and 'M' in x
                    else float(x) if x and pd.notna(x) else np.nan
                )
            
            # Ensure column is float type
            data['Vol.'] = pd.to_numeric(data['Vol.'], errors='coerce')
            
        # Process Change % column
        if 'Change %' in data.columns:
            if data['Change %'].dtype == 'object':
                data['Change %'] = data['Change %'].str.replace('%', '').astype(float) / 100
            # Ensure column is float type
            data['Change %'] = data['Change %'].astype(float)
            
        # Check for NaN values and handle them
        for col in data.columns:
            if data[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
                if data[col].isna().any():
                    logger.info(f"Found {data[col].isna().sum()} NaN values in column {col}, filling with mean")
                    data[col] = data[col].fillna(data[col].mean())
            
        logger.info(f"Loaded {len(data)} rows with columns: {data.columns.tolist()}")
        return data
    
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        return None

def load_mf_data(filename):
    """Load and preprocess mutual fund data from an Excel file."""
    logger.info(f"Loading mutual fund data from {filename}")
    
    try:
        # Load data based on file extension
        if filename.endswith('.xlsx'):
            data = pd.read_excel(filename)
        elif filename.endswith('.csv'):
            data = pd.read_csv(filename)
        else:
            logger.error(f"Unsupported file format: {filename}")
            return None
        
        # Extract fund name from filename
        fund_name = Path(filename).stem.split('_')[0]
        
        # Basic validation of data format
        required_columns = ['Date', 'NAV']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            return None
            
        # Convert date to datetime
        data['Date'] = pd.to_datetime(data['Date'])
        
        # Ensure NAV is numeric
        if data['NAV'].dtype == 'object':
            data['NAV'] = pd.to_numeric(data['NAV'], errors='coerce')
            
        # Add fund_name column
        data['fund_name'] = fund_name
        
        # Check for NaN values and handle them
        if data['NAV'].isna().any():
            logger.info(f"Found {data['NAV'].isna().sum()} NaN values in NAV, filling with mean")
            data['NAV'] = data['NAV'].fillna(data['NAV'].mean())
            
        logger.info(f"Loaded {len(data)} rows with columns: {data.columns.tolist()}")
        return data
    
    except Exception as e:
        logger.error(f"Error loading mutual fund data: {str(e)}")
        return None

def train_stock_models():
    """Train models for all stock files in the stocks directory."""
    # Get all CSV files in the stocks directory
    stock_files = [f for f in os.listdir(STOCKS_DIR) if f.endswith('.csv')]
    
    if not stock_files:
        logger.warning("No stock files found in directory")
        return {}
    
    logger.info(f"Found {len(stock_files)} stock files to process")
    
    # Create model trainer instance
    model_trainer = ModelTrainer(models_dir=os.path.join(MODELS_DIR, "stocks"))
    
    # Overall results
    training_results = {}
    
    # Process each file
    for stock_file in stock_files:
        ticker = stock_file.split('_')[0]
        logger.info(f"\n=== Training model for {ticker} ===")
        
        # Load data
        data = load_stock_data(os.path.join(STOCKS_DIR, stock_file))
        if data is None:
            continue
            
        # Check minimum required data points
        if len(data) < 100:  # Minimum data requirement
            logger.warning(f"Insufficient data for {ticker}: {len(data)} points (minimum 100 required)")
            continue
            
        try:
            # Train the model
            start_time = time.time()
            models = model_trainer.train_lstm_models(data)
            training_time = time.time() - start_time
            
            if ticker in models['models']:
                logger.info(f"Successfully trained model for {ticker}")
                
                # Get model metrics
                model_info = models['models'][ticker]
                
                # Save results
                training_results[ticker] = {
                    'ticker': ticker,
                    'data_points': len(data),
                    'training_time': training_time,
                    'loss': model_info.get('loss', 'N/A'),
                    'features_used': model_info.get('features', []),
                    'model_saved': True,
                    'model_path': os.path.join(MODELS_DIR, "stocks", f"{ticker}_lstm.h5"),
                    'status': 'SUCCESS'
                }
            else:
                logger.error(f"Failed to train model for {ticker}")
                training_results[ticker] = {
                    'ticker': ticker,
                    'status': 'FAILED',
                    'reason': 'Model training failed'
                }
                
        except Exception as e:
            logger.error(f"Error training model for {ticker}: {str(e)}")
            training_results[ticker] = {
                'ticker': ticker,
                'status': 'ERROR',
                'reason': str(e)
            }
    
    return training_results

def train_mf_models():
    """Train models for all mutual fund files in the mutual_funds directory."""
    # Get all MF files in the mutual_funds directory
    mf_files = [f for f in os.listdir(MF_DIR) if f.endswith(('.xlsx', '.csv'))]
    
    if not mf_files:
        logger.warning("No mutual fund files found in directory")
        return {}
    
    logger.info(f"Found {len(mf_files)} mutual fund files to process")
    
    # Create model trainer instance for mutual funds
    model_trainer = ModelTrainer(models_dir=os.path.join(MODELS_DIR, "mutual_funds"))
    
    # Overall results
    training_results = {}
    
    # Process each file
    for mf_file in mf_files:
        fund_name = mf_file.split('_')[0]
        logger.info(f"\n=== Training model for {fund_name} mutual fund ===")
        
        # Load data
        data = load_mf_data(os.path.join(MF_DIR, mf_file))
        if data is None:
            continue
            
        # Check minimum required data points
        if len(data) < 100:  # Minimum data requirement
            logger.warning(f"Insufficient data for {fund_name}: {len(data)} points (minimum 100 required)")
            continue
            
        try:
            # Prepare data for training
            # For mutual funds, we'll use NAV as the primary target
            train_data = data.copy()
            train_data['Price'] = train_data['NAV']  # Use NAV as price for consistency with stock models
            
            # Train the model
            start_time = time.time()
            models = model_trainer.train_lstm_models(train_data)
            training_time = time.time() - start_time
            
            model_id = fund_name
            if model_id in models['models']:
                logger.info(f"Successfully trained model for {fund_name} mutual fund")
                
                # Get model metrics
                model_info = models['models'][model_id]
                
                # Save results
                training_results[fund_name] = {
                    'fund_name': fund_name,
                    'data_points': len(data),
                    'training_time': training_time,
                    'loss': model_info.get('loss', 'N/A'),
                    'features_used': model_info.get('features', []),
                    'model_saved': True,
                    'model_path': os.path.join(MODELS_DIR, "mutual_funds", f"{fund_name}_lstm.h5"),
                    'status': 'SUCCESS'
                }
            else:
                logger.error(f"Failed to train model for {fund_name} mutual fund")
                training_results[fund_name] = {
                    'fund_name': fund_name,
                    'status': 'FAILED',
                    'reason': 'Model training failed'
                }
                
        except Exception as e:
            logger.error(f"Error training model for {fund_name} mutual fund: {str(e)}")
            training_results[fund_name] = {
                'fund_name': fund_name,
                'status': 'ERROR',
                'reason': str(e)
            }
    
    return training_results

def generate_training_report(stock_results, mf_results):
    """Generate a comprehensive training report."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Count successes and failures
    stock_successes = sum(1 for r in stock_results.values() if r.get('status') == 'SUCCESS')
    stock_failures = len(stock_results) - stock_successes
    
    mf_successes = sum(1 for r in mf_results.values() if r.get('status') == 'SUCCESS')
    mf_failures = len(mf_results) - mf_successes
    
    # Create summary report
    report = {
        'timestamp': now,
        'stocks': {
            'total': len(stock_results),
            'success': stock_successes,
            'failed': stock_failures,
            'details': stock_results
        },
        'mutual_funds': {
            'total': len(mf_results),
            'success': mf_successes,
            'failed': mf_failures,
            'details': mf_results
        },
        'overall': {
            'total_trained': stock_successes + mf_successes,
            'total_failed': stock_failures + mf_failures
        }
    }
    
    # Save report as JSON
    report_path = os.path.join(RESULTS_DIR, "training", "training_summary.json")
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"Training report saved to {report_path}")
    
    # Generate visual report
    plt.figure(figsize=(12, 8))
    
    # Plot training success rates
    labels = ['Stocks', 'Mutual Funds']
    success_rates = [
        stock_successes / len(stock_results) * 100 if stock_results else 0,
        mf_successes / len(mf_results) * 100 if mf_results else 0
    ]
    
    plt.bar(labels, success_rates, color=['blue', 'green'])
    plt.title('Model Training Success Rate')
    plt.ylabel('Success Rate (%)')
    plt.ylim(0, 100)
    
    # Add text labels on bars
    for i, rate in enumerate(success_rates):
        plt.text(i, rate + 2, f"{rate:.1f}%", ha='center')
        
    # Add counts below bars
    for i, counts in enumerate([
        f"{stock_successes}/{len(stock_results)}",
        f"{mf_successes}/{len(mf_results)}"
    ]):
        plt.text(i, -5, counts, ha='center')
    
    # Save chart
    chart_path = os.path.join(RESULTS_DIR, "training", "training_success_rate.png")
    plt.tight_layout()
    plt.savefig(chart_path)
    plt.close()
    
    logger.info(f"Training success chart saved to {chart_path}")
    
    return report

def main():
    logger.info("Starting model training for all available files")
    
    # Train stock models
    logger.info("=== Training Stock Models ===")
    stock_results = train_stock_models()
    
    # Train mutual fund models
    logger.info("=== Training Mutual Fund Models ===")
    mf_results = train_mf_models()
    
    # Generate training report
    logger.info("=== Generating Training Report ===")
    report = generate_training_report(stock_results, mf_results)
    
    # Print summary
    total_trained = report['overall']['total_trained']
    total_failed = report['overall']['total_failed']
    total_files = total_trained + total_failed
    
    logger.info(f"\n=== Training Summary ===")
    logger.info(f"Total files processed: {total_files}")
    logger.info(f"Models successfully trained: {total_trained} ({total_trained/total_files*100:.1f}% success rate)")
    logger.info(f"Training failures: {total_failed}")
    logger.info(f"Full report saved to {os.path.join(RESULTS_DIR, 'training', 'training_summary.json')}")
    
    # Return success rate
    return total_trained / total_files if total_files > 0 else 0

if __name__ == "__main__":
    main() 