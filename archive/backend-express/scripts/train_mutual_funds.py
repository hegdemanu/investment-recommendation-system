#!/usr/bin/env python3
"""
Process mutual fund Excel files and train models on them.
This script:
1. Converts Excel files to CSV format
2. Trains LSTM, ARIMA-GARCH, and Prophet models for each mutual fund
3. Implements model weighting based on time horizon performance
4. Stores models and metadata in the organized directory structure
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
        logging.FileHandler("mf_training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("MutualFundTrainer")

# Configure directories
MF_DIR = "./data/mutual_funds"
MF_CSV_DIR = "./data/mutual_funds/csv"
MODELS_DIR = "./models/mutual_funds"
RESULTS_DIR = "./results/training"
REPORTS_DIR = "./results/reports/expert"

# Create directories if they don't exist
os.makedirs(MF_CSV_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# Time horizon definitions
TIME_HORIZONS = {
    'short': [1, 3, 5],      # Short-term: 1-5 days
    'medium': [7, 14, 21],   # Medium-term: 7-21 days
    'long': [30, 60, 90]     # Long-term: 30-90 days
}

# Model weights by horizon (for ensemble predictions)
MODEL_WEIGHTS = {
    'short': {'lstm': 0.5, 'arima_garch': 0.3, 'prophet': 0.2},
    'medium': {'lstm': 0.4, 'arima_garch': 0.3, 'prophet': 0.3},
    'long': {'lstm': 0.2, 'arima_garch': 0.4, 'prophet': 0.4}  # More weight to ARIMA-GARCH and Prophet for long-term
}

def convert_excel_to_csv(excel_file):
    """Convert Excel file to CSV format."""
    logger.info(f"Converting {excel_file} to CSV")
    
    try:
        # Extract fund name from filename
        fund_name = Path(excel_file).stem.split('_')[0]
        
        # Read Excel file
        df = pd.read_excel(excel_file)
        
        # Check if 'Date' and 'NAV' columns exist
        if 'Date' not in df.columns:
            # Look for date-like column
            date_cols = [col for col in df.columns if 'date' in col.lower()]
            if date_cols:
                logger.info(f"Renaming {date_cols[0]} to Date")
                df.rename(columns={date_cols[0]: 'Date'}, inplace=True)
            else:
                logger.error(f"No date column found in {excel_file}")
                return None
                
        if 'NAV' not in df.columns:
            # Look for NAV-like column
            nav_cols = [col for col in df.columns if 'nav' in col.lower() or 'price' in col.lower() or 'value' in col.lower()]
            if nav_cols:
                logger.info(f"Renaming {nav_cols[0]} to NAV")
                df.rename(columns={nav_cols[0]: 'NAV'}, inplace=True)
            else:
                logger.error(f"No NAV column found in {excel_file}")
                return None
        
        # Ensure Date is datetime
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        
        # Ensure NAV is numeric
        df['NAV'] = pd.to_numeric(df['NAV'], errors='coerce')
        
        # Drop rows with NaN in Date or NAV
        df = df.dropna(subset=['Date', 'NAV'])
        
        # Add fund_name column
        df['fund_name'] = fund_name
        
        # Save to CSV
        csv_file = os.path.join(MF_CSV_DIR, f"{fund_name}.csv")
        df.to_csv(csv_file, index=False)
        
        logger.info(f"Saved {len(df)} rows to {csv_file}")
        return csv_file
        
    except Exception as e:
        logger.error(f"Error converting {excel_file} to CSV: {str(e)}")
        return None

def load_mf_data(filename):
    """Load and preprocess mutual fund data."""
    logger.info(f"Loading mutual fund data from {filename}")
    
    try:
        # Load data
        data = pd.read_csv(filename)
        
        # Extract fund name from filename
        fund_name = Path(filename).stem
        
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
            
        # Add fund_name column if not present
        if 'fund_name' not in data.columns:
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

def prepare_training_data(data, fund_name):
    """Prepare mutual fund data for model training."""
    train_data = data.copy()
    
    # For mutual funds, we'll use NAV as the primary target
    train_data['Price'] = train_data['NAV']  # Use NAV as price for consistency with stock models
    train_data['Open'] = train_data['NAV']   # Fill required columns with NAV
    train_data['High'] = train_data['NAV']
    train_data['Low'] = train_data['NAV']
    train_data['Vol.'] = 0  # Fill with 0 as we don't have volume data
    train_data['Change %'] = train_data['NAV'].pct_change()  # Calculate percent change
    
    # Fill NaN values in Change %
    train_data['Change %'] = train_data['Change %'].fillna(0)
    
    # Add ticker column for model training
    train_data['ticker'] = fund_name
    
    # Sort by date
    train_data = train_data.sort_values('Date')
    
    return train_data

def train_lstm_model(data, fund_name, model_trainer):
    """Train LSTM model for a mutual fund."""
    logger.info(f"Training LSTM model for {fund_name} mutual fund")
    
    try:
        # Prepare data for training
        train_data = prepare_training_data(data, fund_name)
        
        # Check if we have enough data
        if len(train_data) < 100:  # Minimum data requirement
            logger.warning(f"Insufficient data for LSTM on {fund_name}: {len(train_data)} points (minimum 100 required)")
            return None
            
        # Train the model
        start_time = time.time()
        models = model_trainer.train_lstm_models(train_data)
        training_time = time.time() - start_time
        
        if fund_name in models['models']:
            logger.info(f"Successfully trained LSTM model for {fund_name} mutual fund")
            
            # Get model metrics
            model_info = models['models'][fund_name]
            
            # Return result
            return {
                'model_type': 'lstm',
                'data_points': len(data),
                'training_time': training_time,
                'loss': model_info.get('loss', 'N/A'),
                'features_used': model_info.get('features', []),
                'model_saved': True,
                'model_path': os.path.join(MODELS_DIR, f"{fund_name}_lstm.h5"),
                'status': 'SUCCESS'
            }
        else:
            logger.error(f"Failed to train LSTM model for {fund_name} mutual fund")
            return {
                'model_type': 'lstm',
                'status': 'FAILED',
                'reason': 'Model training failed'
            }
            
    except Exception as e:
        logger.error(f"Error training LSTM model for {fund_name} mutual fund: {str(e)}")
        return {
            'model_type': 'lstm',
            'status': 'ERROR',
            'reason': str(e)
        }

def train_arima_garch_model(data, fund_name, model_trainer):
    """Train ARIMA-GARCH model for a mutual fund."""
    logger.info(f"Training ARIMA-GARCH model for {fund_name} mutual fund")
    
    try:
        # Prepare data for training
        train_data = prepare_training_data(data, fund_name)
        
        # Check if we have enough data
        if len(train_data) < 50:  # ARIMA requires less data than LSTM
            logger.warning(f"Insufficient data for ARIMA-GARCH on {fund_name}: {len(train_data)} points (minimum 50 required)")
            return None
            
        # Train the model - focusing on the NAV/Price column
        start_time = time.time()
        model_result = model_trainer.train_arima_garch_model(
            train_data, 
            target_column='Price', 
            ticker=fund_name
        )
        training_time = time.time() - start_time
        
        if model_result and model_result.get('status') == 'success':
            logger.info(f"Successfully trained ARIMA-GARCH model for {fund_name} mutual fund")
            
            return {
                'model_type': 'arima_garch',
                'data_points': len(data),
                'training_time': training_time,
                'model_order': model_result.get('model_order', 'N/A'),
                'aic': model_result.get('aic', 'N/A'),
                'model_saved': True,
                'model_path': os.path.join(MODELS_DIR, f"{fund_name}_arima_garch.pkl"),
                'status': 'SUCCESS'
            }
        else:
            logger.error(f"Failed to train ARIMA-GARCH model for {fund_name} mutual fund")
            return {
                'model_type': 'arima_garch',
                'status': 'FAILED',
                'reason': 'Model training failed or did not converge'
            }
            
    except Exception as e:
        logger.error(f"Error training ARIMA-GARCH model for {fund_name} mutual fund: {str(e)}")
        return {
            'model_type': 'arima_garch',
            'status': 'ERROR',
            'reason': str(e)
        }

def train_prophet_model(data, fund_name, model_trainer):
    """Train Prophet model for a mutual fund."""
    logger.info(f"Training Prophet model for {fund_name} mutual fund")
    
    try:
        # Prophet requires a specific data format with 'ds' (date) and 'y' (target) columns
        prophet_data = data.copy()
        prophet_data = prophet_data.rename(columns={'Date': 'ds', 'NAV': 'y'})
        
        # Check if we have enough data
        if len(prophet_data) < 30:  # Prophet can work with less data
            logger.warning(f"Insufficient data for Prophet on {fund_name}: {len(prophet_data)} points (minimum 30 required)")
            return None
            
        # Train the model
        start_time = time.time()
        model_result = model_trainer.train_prophet_model(
            prophet_data, 
            ticker=fund_name,
            future_periods=max(TIME_HORIZONS['long'])  # Train for the longest horizon
        )
        training_time = time.time() - start_time
        
        if model_result and model_result.get('status') == 'success':
            logger.info(f"Successfully trained Prophet model for {fund_name} mutual fund")
            
            return {
                'model_type': 'prophet',
                'data_points': len(data),
                'training_time': training_time,
                'metrics': model_result.get('metrics', {}),
                'model_saved': True,
                'model_path': os.path.join(MODELS_DIR, f"{fund_name}_prophet.pkl"),
                'status': 'SUCCESS'
            }
        else:
            logger.error(f"Failed to train Prophet model for {fund_name} mutual fund")
            return {
                'model_type': 'prophet',
                'status': 'FAILED',
                'reason': 'Model training failed'
            }
            
    except Exception as e:
        logger.error(f"Error training Prophet model for {fund_name} mutual fund: {str(e)}")
        return {
            'model_type': 'prophet',
            'status': 'ERROR',
            'reason': str(e)
        }

def generate_expert_report(fund_name, model_results, data):
    """Generate expert-level PDF report for mutual fund models."""
    try:
        # Create a dictionary to store all report data
        report_data = {
            'fund_name': fund_name,
            'generated_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'data_size': len(data),
            'date_range': f"{data['Date'].min().strftime('%Y-%m-%d')} to {data['Date'].max().strftime('%Y-%m-%d')}",
            'models': model_results,
            'model_weights': MODEL_WEIGHTS,
            'time_horizons': TIME_HORIZONS
        }
        
        # Save the full report data as JSON for expert analysis
        report_path = os.path.join(REPORTS_DIR, f"{fund_name}_expert_report.json")
        with open(report_path, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        logger.info(f"Expert report data saved to {report_path}")
        
        # TODO: Generate PDF report visualization with matplotlib or a PDF library
        # This would be an enhancement for the future
        
        return report_path
        
    except Exception as e:
        logger.error(f"Error generating expert report for {fund_name}: {str(e)}")
        return None

def train_mf_models(data, fund_name):
    """Train all model types for a mutual fund."""
    logger.info(f"\n=== Training models for {fund_name} mutual fund ===")
    
    # Create model trainer instance
    model_trainer = ModelTrainer(models_dir=MODELS_DIR)
    
    # Dictionary to store all model results
    model_results = {}
    
    # Check if we have enough data overall
    if len(data) < 30:  # Minimum requirement for any model
        logger.warning(f"Insufficient data for {fund_name}: {len(data)} points (minimum 30 required)")
        return {
            'fund_name': fund_name,
            'status': 'FAILED',
            'reason': 'Insufficient data points'
        }
    
    # Train LSTM model
    lstm_result = train_lstm_model(data, fund_name, model_trainer)
    if lstm_result:
        model_results['lstm'] = lstm_result
    
    # Train ARIMA-GARCH model
    arima_result = train_arima_garch_model(data, fund_name, model_trainer)
    if arima_result:
        model_results['arima_garch'] = arima_result
    
    # Train Prophet model
    prophet_result = train_prophet_model(data, fund_name, model_trainer)
    if prophet_result:
        model_results['prophet'] = prophet_result
    
    # Generate expert report
    if model_results:
        report_path = generate_expert_report(fund_name, model_results, data)
        
        # Return overall result
        successful_models = sum(1 for m in model_results.values() if m.get('status') == 'SUCCESS')
        
        if successful_models > 0:
            return {
                'fund_name': fund_name,
                'data_points': len(data),
                'models_trained': list(model_results.keys()),
                'successful_models': successful_models,
                'total_models': len(model_results),
                'expert_report': report_path,
                'model_results': model_results,
                'status': 'SUCCESS'
            }
        else:
            return {
                'fund_name': fund_name,
                'status': 'PARTIAL_FAILURE',
                'reason': 'No models trained successfully'
            }
    else:
        return {
            'fund_name': fund_name,
            'status': 'FAILED',
            'reason': 'Failed to train any models'
        }

def main():
    logger.info("=== Processing Mutual Fund Files ===")
    
    # Get all Excel files in the mutual funds directory
    excel_files = [os.path.join(MF_DIR, f) for f in os.listdir(MF_DIR) if f.endswith(('.xlsx', '.xls'))]
    
    if not excel_files:
        logger.warning("No mutual fund Excel files found")
        return {}
    
    logger.info(f"Found {len(excel_files)} mutual fund Excel files")
    
    # Convert Excel files to CSV
    csv_files = []
    for excel_file in excel_files:
        csv_file = convert_excel_to_csv(excel_file)
        if csv_file:
            csv_files.append(csv_file)
    
    logger.info(f"Converted {len(csv_files)} Excel files to CSV")
    
    # Train models for each CSV file
    training_results = {}
    
    for csv_file in csv_files:
        fund_name = Path(csv_file).stem
        
        # Load data
        data = load_mf_data(csv_file)
        if data is None:
            continue
            
        # Train all models
        result = train_mf_models(data, fund_name)
        if result:
            training_results[fund_name] = result
    
    # Count successes by model type
    model_success_counts = {
        'lstm': 0,
        'arima_garch': 0,
        'prophet': 0
    }
    
    for fund_result in training_results.values():
        if fund_result.get('status') == 'SUCCESS' and 'model_results' in fund_result:
            for model_type, model_result in fund_result['model_results'].items():
                if model_result.get('status') == 'SUCCESS':
                    model_success_counts[model_type] += 1
    
    # Generate summary report
    summary = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'total': len(excel_files),
        'processed': len(csv_files),
        'success': sum(1 for r in training_results.values() if r.get('status') == 'SUCCESS'),
        'failed': sum(1 for r in training_results.values() if r.get('status') != 'SUCCESS'),
        'model_successes': model_success_counts,
        'weights': MODEL_WEIGHTS,
        'time_horizons': TIME_HORIZONS,
        'details': training_results
    }
    
    # Save summary
    summary_path = os.path.join(RESULTS_DIR, "mutual_funds_training_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Mutual fund training summary saved to {summary_path}")
    
    # Print summary
    logger.info(f"\n=== Mutual Fund Training Summary ===")
    logger.info(f"Total Excel files: {len(excel_files)}")
    logger.info(f"Files processed: {len(csv_files)}")
    logger.info(f"Models successfully trained: {summary['success']}")
    logger.info(f"Training failures: {summary['failed']}")
    logger.info(f"Model success by type: LSTM: {model_success_counts['lstm']}, ARIMA-GARCH: {model_success_counts['arima_garch']}, Prophet: {model_success_counts['prophet']}")
    
    return summary

if __name__ == "__main__":
    main() 