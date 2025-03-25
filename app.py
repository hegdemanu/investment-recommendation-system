"""
Investment Recommendation System Flask API
==========================================
A Flask-based REST API for stock price prediction, risk analysis, 
and investment recommendations tailored for Indian markets.
"""

import os
import json
import pandas as pd
from flask import Flask, request, jsonify, send_file, render_template
from werkzeug.utils import secure_filename
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('InvestmentRecommendationAPI')

# Import our modules
from src.data_processor import DataProcessor
from src.model_trainer import ModelTrainer
from src.risk_analyzer import RiskAnalyzer
from src.recommendation_engine import RecommendationEngine
from src.long_term_models import LongTermPredictor
from src.report_generator import ReportGenerator

# Create Flask application
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './data/uploads'
app.config['MODELS_DIR'] = './models'
app.config['REPORTS_DIR'] = './reports'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max upload size

# Create directories if they don't exist
for directory in [app.config['UPLOAD_FOLDER'], app.config['MODELS_DIR'], app.config['REPORTS_DIR']]:
    if not os.path.exists(directory):
        os.makedirs(directory)

# Initialize components
data_processor = DataProcessor()
model_trainer = ModelTrainer(models_dir=app.config['MODELS_DIR'])
risk_analyzer = RiskAnalyzer()
recommendation_engine = RecommendationEngine()
long_term_predictor = LongTermPredictor(models_dir=app.config['MODELS_DIR'])
report_generator = ReportGenerator(reports_dir=app.config['REPORTS_DIR'])

# Validate directory structure
data_processor.validate_directory_structure()

# Global variables for cached data
cached_data = None
cached_risk_profiles = None
cached_predictions = None
cached_long_term_predictions = None

# Helper function to check for valid file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'csv'}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/upload-csv', methods=['POST'])
def upload_csv():
    """API endpoint for uploading CSV data."""
    global cached_data, cached_risk_profiles, cached_predictions, cached_long_term_predictions
    
    # Check if a file was included in the request
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    # Check if a file was selected
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # Check if the file is a CSV
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Load and preprocess data
        try:
            data = data_processor.load_file(filepath)
            
            if data is None or data.empty:
                return jsonify({'error': 'Failed to load data from uploaded file. The file may be empty or in an incorrect format.'}), 400
            
            processed_data = data_processor.preprocess(data)
            cached_data = processed_data
            
            # Reset other cached data
            cached_risk_profiles = None
            cached_predictions = None
            cached_long_term_predictions = None
            
            # Extract tickers if available
            tickers = list(processed_data['ticker'].unique()) if 'ticker' in processed_data.columns else []
            
            # Get sample data structure for info
            data_sample = {}
            sample_row = processed_data.iloc[0].to_dict() if not processed_data.empty else {}
            for key, value in sample_row.items():
                if isinstance(value, (int, float)):
                    data_sample[key] = f"{value:.2f}" if isinstance(value, float) else value
                else:
                    data_sample[key] = str(value)
            
            return jsonify({
                'success': True,
                'message': f'Successfully loaded and preprocessed {len(processed_data)} records from {filename}',
                'data_info': {
                    'rows': len(processed_data),
                    'columns': list(processed_data.columns),
                    'tickers': tickers,
                    'date_range': {
                        'start': processed_data['Date'].min().strftime('%Y-%m-%d') if 'Date' in processed_data.columns else None,
                        'end': processed_data['Date'].max().strftime('%Y-%m-%d') if 'Date' in processed_data.columns else None
                    },
                    'sample_data': data_sample
                }
            })
        except Exception as e:
            logger.error(f"Error processing uploaded file: {str(e)}")
            return jsonify({'error': f'Error processing data: {str(e)}'}), 500
    
    return jsonify({'error': 'File type not allowed. Please upload a CSV file.'}), 400

@app.route('/api/batch-upload', methods=['POST'])
def batch_upload():
    """API endpoint for uploading multiple CSV files."""
    global cached_data, cached_risk_profiles, cached_predictions, cached_long_term_predictions
    
    # Check if files were included in the request
    if 'files[]' not in request.files:
        return jsonify({'error': 'No files included in request'}), 400
    
    files = request.files.getlist('files[]')
    
    if not files or len(files) == 0:
        return jsonify({'error': 'No files selected'}), 400
    
    # Process each file
    successful_files = []
    failed_files = []
    
    for file in files:
        if file and file.filename and allowed_file(file.filename):
            try:
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                successful_files.append(filename)
            except Exception as e:
                logger.error(f"Error saving file {file.filename}: {str(e)}")
                failed_files.append({'name': file.filename, 'error': str(e)})
        else:
            if file and file.filename:
                failed_files.append({'name': file.filename, 'error': 'Invalid file type'})
    
    if not successful_files:
        return jsonify({
            'error': 'All files failed to upload',
            'failed_files': failed_files
        }), 400
    
    # Combine and process all successful files
    try:
        combined_data = data_processor.load_and_combine_files(f"{app.config['UPLOAD_FOLDER']}/*.csv")
        
        if combined_data.empty:
            return jsonify({
                'warning': 'Files were uploaded but no valid data could be extracted',
                'successful_files': successful_files,
                'failed_files': failed_files
            }), 200
        
        processed_data = data_processor.preprocess(combined_data)
        cached_data = processed_data
        
        # Reset other cached data
        cached_risk_profiles = None
        cached_predictions = None 
        cached_long_term_predictions = None
        
        return jsonify({
            'success': True,
            'message': f'Successfully processed {len(successful_files)} files with {len(processed_data)} total rows',
            'successful_files': successful_files,
            'failed_files': failed_files,
            'data_info': {
                'rows': len(processed_data),
                'columns': list(processed_data.columns),
                'tickers': list(processed_data['ticker'].unique()) if 'ticker' in processed_data.columns else []
            }
        })
    except Exception as e:
        logger.error(f"Error combining and processing files: {str(e)}")
        return jsonify({
            'error': f'Error processing data: {str(e)}',
            'successful_files': successful_files,
            'failed_files': failed_files
        }), 500

@app.route('/api/fetch-data', methods=['POST'])
def fetch_data():
    """API endpoint for fetching data from RapidAPI."""
    global cached_data, cached_risk_profiles, cached_predictions, cached_long_term_predictions
    
    # Get API parameters from request
    data = request.get_json()
    
    if not data:
        return jsonify({'error': 'No data provided'}), 400
    
    ticker_list = data.get('ticker_list')
    start_date = data.get('start_date')
    end_date = data.get('end_date')
    api_key = data.get('api_key')
    
    if not ticker_list or not api_key:
        return jsonify({'error': 'Missing required parameters: ticker_list and api_key are required'}), 400
    
    # Set default dates if not provided
    if not start_date:
        start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        logger.info(f"No start_date provided, using default: {start_date}")
        
    if not end_date:
        end_date = datetime.now().strftime('%Y-%m-%d')
        logger.info(f"No end_date provided, using default: {end_date}")
    
    # Fetch data from API
    try:
        data = data_processor.load_from_api(ticker_list, start_date, end_date, api_key)
        
        if data is None or data.empty:
            return jsonify({
                'error': 'Failed to fetch any valid data from API. Please check your ticker symbols or try again later.'
            }), 400
            
        processed_data = data_processor.preprocess(data)
        cached_data = processed_data
        
        # Reset other cached data
        cached_risk_profiles = None
        cached_predictions = None
        cached_long_term_predictions = None
        
        # Check if we got data for all requested tickers
        retrieved_tickers = list(processed_data['ticker'].unique()) if 'ticker' in processed_data.columns else []
        missing_tickers = [ticker for ticker in ticker_list if ticker not in retrieved_tickers]
        
        return jsonify({
            'success': True,
            'message': f'Successfully fetched and preprocessed {len(processed_data)} records for {len(retrieved_tickers)} tickers',
            'data_info': {
                'rows': len(processed_data),
                'columns': list(processed_data.columns),
                'tickers': retrieved_tickers,
                'missing_tickers': missing_tickers
            }
        })
    except Exception as e:
        logger.error(f"Error fetching data from API: {str(e)}")
        return jsonify({'error': f'Error fetching data: {str(e)}'}), 500

@app.route('/api/train-models', methods=['POST'])
def train_models():
    """API endpoint for training prediction models."""
    global cached_data, cached_predictions, cached_long_term_predictions
    
    if cached_data is None:
        return jsonify({'error': 'No data available. Please upload or fetch data first.'}), 400
    
    # Get parameters from request
    data = request.get_json() or {}
    model_type = data.get('model_type', 'lstm')  # 'lstm', 'long_term', or 'all'
    force_retrain = data.get('force_retrain', False)
    
    # Check for data drift if not forcing retrain
    if not force_retrain and model_type in ['lstm', 'all']:
        data_drift = model_trainer.check_data_drift(cached_data)
        if not data_drift:
            # Try to load existing models
            try:
                lstm_models = model_trainer.load_lstm_models()
                if lstm_models and lstm_models.get('models'):
                    cached_predictions = model_trainer.predict_lstm(lstm_models, cached_data)
                    return jsonify({
                        'success': True,
                        'message': f'Using existing LSTM models for {len(lstm_models["models"])} tickers. No significant data drift detected.',
                        'predictions_info': {
                            'rows': len(cached_predictions),
                            'tickers': list(cached_predictions['ticker'].unique()) if not cached_predictions.empty else []
                        }
                    })
            except Exception as e:
                logger.warning(f"Error loading existing models: {str(e)}")
    
    try:
        results = {}
        
        # Train LSTM models
        if model_type in ['lstm', 'all']:
            try:
                lstm_models = model_trainer.train_lstm_models(cached_data)
                if lstm_models:
                    cached_predictions = model_trainer.predict_lstm(lstm_models, cached_data)
                    results['lstm'] = {
                        'models_trained': len(lstm_models.get('models', [])),
                        'predictions_generated': len(cached_predictions) if cached_predictions is not None else 0
                    }
            except Exception as e:
                logger.error(f"Error training LSTM models: {str(e)}")
                results['lstm'] = {'error': str(e)}
        
        # Train long-term models
        if model_type in ['long_term', 'all']:
            try:
                long_term_models = long_term_predictor.fit_long_term_models(cached_data)
                if long_term_models:
                    cached_long_term_predictions = long_term_predictor.predict_ensemble(long_term_models, cached_data)
                    results['long_term'] = {
                        'arima_garch_models_trained': len(long_term_models.get('arima_garch', [])),
                        'prophet_models_trained': len(long_term_models.get('prophet', [])),
                        'predictions_generated': len(cached_long_term_predictions) if cached_long_term_predictions is not None else 0
                    }
            except Exception as e:
                logger.error(f"Error training long-term models: {str(e)}")
                results['long_term'] = {'error': str(e)}
        
        if not results:
            return jsonify({'error': 'No models were trained successfully'}), 500
            
        return jsonify({
            'success': True,
            'message': 'Models trained successfully',
            'results': results
        })
    except Exception as e:
        logger.error(f"Error training models: {str(e)}")
        return jsonify({'error': f'Error training models: {str(e)}'}), 500

@app.route('/api/multi-timeframe-predictions', methods=['POST'])
def multi_timeframe_predictions():
    """API endpoint for generating predictions across multiple time frames."""
    global cached_data, cached_predictions
    
    if cached_data is None:
        return jsonify({'error': 'No data available. Please upload or fetch data first.'}), 400
    
    # Get parameters from request
    data = request.get_json() or {}
    ticker = data.get('ticker')  # Optional - filter by ticker
    
    try:
        # Get ticker-specific data if specified
        if ticker and 'ticker' in cached_data.columns:
            ticker_data = cached_data[cached_data['ticker'] == ticker].copy()
            if ticker_data.empty:
                return jsonify({'error': f'No data available for ticker: {ticker}'}), 404
        else:
            ticker_data = cached_data.copy()
        
        # Load models
        lstm_models = model_trainer.load_lstm_models()
        if not lstm_models or not lstm_models.get('models'):
            # Try training models first
            lstm_models = model_trainer.train_lstm_models(cached_data)
            if not lstm_models or not lstm_models.get('models'):
                return jsonify({'error': 'Failed to load or train models'}), 500
        
        # Get predictions for different time frames
        timeframes = {
            'short_term': 7,    # 1 week
            'medium_term': 30,  # 1 month
            'long_term': 90     # 3 months
        }
        
        multi_tf_results = {}
        
        for ticker_name, ticker_data in ticker_data.groupby('ticker'):
            # Skip if we don't have enough data
            if len(ticker_data) < 60:  # Need at least 60 data points for prediction
                continue
                
            # Get ticker model
            if ticker_name not in lstm_models['models']:
                continue
                
            ticker_model = lstm_models['models'][ticker_name]
            
            # Prepare data for prediction
            last_sequence, available_features = model_trainer.prepare_sequence_for_prediction(
                ticker_data, 
                ticker_model['features'], 
                ticker_model['scaler']
            )
            
            if last_sequence is None:
                continue
                
            # Generate predictions for each timeframe
            tf_predictions = {}
            last_price = ticker_data['Price'].iloc[-1]
            
            for tf_name, days in timeframes.items():
                predictions = model_trainer.predict_future(
                    ticker_model['model'],
                    last_sequence,
                    ticker_model['scaler'],
                    available_features,
                    days=days
                )
                
                # Calculate metrics
                tf_predictions[tf_name] = {
                    'days': days,
                    'final_price': float(predictions[-1]),
                    'change_pct': float(((predictions[-1] / last_price) - 1) * 100),
                    'predictions': [float(p) for p in predictions]
                }
            
            multi_tf_results[ticker_name] = {
                'timeframes': tf_predictions,
                'last_price': float(last_price),
                'last_date': ticker_data['Date'].iloc[-1].strftime('%Y-%m-%d')
            }
        
        if not multi_tf_results:
            return jsonify({'error': 'No valid predictions generated for any ticker'}), 500
            
        return jsonify({
            'success': True,
            'message': 'Multi-timeframe predictions generated successfully',
            'ticker_count': len(multi_tf_results),
            'predictions': multi_tf_results
        })
    except Exception as e:
        logger.error(f"Error generating multi-timeframe predictions: {str(e)}")
        return jsonify({'error': f'Error generating predictions: {str(e)}'}), 500

@app.route('/api/backtracking-analysis', methods=['POST'])
def backtracking_analysis():
    """API endpoint for running backtracking analysis on historical data."""
    global cached_data
    
    if cached_data is None:
        return jsonify({'error': 'No data available. Please upload or fetch data first.'}), 400
    
    # Get parameters from request
    data = request.get_json() or {}
    ticker = data.get('ticker')  # Optional - filter by ticker
    custom_periods = data.get('periods')  # Optional - custom periods for analysis
    
    try:
        # Get ticker-specific data if specified
        if ticker and 'ticker' in cached_data.columns:
            ticker_data = cached_data[cached_data['ticker'] == ticker].copy()
            if ticker_data.empty:
                return jsonify({'error': f'No data available for ticker: {ticker}'}), 404
        else:
            ticker_data = cached_data.copy()
        
        # Need at least 180 days of data for meaningful backtracking
        if len(ticker_data) < 180:
            return jsonify({'error': 'Not enough historical data for backtracking analysis. Need at least 180 days.'}), 400
        
        # Load models
        lstm_models = model_trainer.load_lstm_models()
        if not lstm_models or not lstm_models.get('models'):
            # Try training models first
            lstm_models = model_trainer.train_lstm_models(cached_data)
            if not lstm_models or not lstm_models.get('models'):
                return jsonify({'error': 'Failed to load or train models'}), 500
        
        # Determine periods for backtracking
        if custom_periods:
            periods = custom_periods
        else:
            # Using equally spaced periods from the data
            start_date = ticker_data['Date'].min()
            end_date = ticker_data['Date'].max()
            total_days = (end_date - start_date).days
            
            # Create 3 equal periods
            period_length = total_days // 3
            
            periods = {
                'early_period': (start_date.strftime('%Y-%m-%d'), 
                              (start_date + pd.Timedelta(days=period_length)).strftime('%Y-%m-%d')),
                'mid_period': ((start_date + pd.Timedelta(days=period_length+1)).strftime('%Y-%m-%d'),
                            (start_date + pd.Timedelta(days=2*period_length)).strftime('%Y-%m-%d')),
                'recent_period': ((start_date + pd.Timedelta(days=2*period_length+1)).strftime('%Y-%m-%d'),
                               end_date.strftime('%Y-%m-%d'))
            }
        
        backtracking_results = {}
        
        for ticker_name, ticker_group in ticker_data.groupby('ticker'):
            # Skip if no model for this ticker
            if ticker_name not in lstm_models['models']:
                continue
                
            ticker_model = lstm_models['models'][ticker_name]
            
            # Perform backtracking for this ticker
            ticker_results = model_trainer.backtracking_analysis(
                ticker_group,
                ticker_model['model'],
                ticker_model['scaler'],
                ticker_model['features'],
                periods
            )
            
            if ticker_results:
                backtracking_results[ticker_name] = ticker_results
        
        if not backtracking_results:
            return jsonify({'error': 'No valid backtracking results generated for any ticker'}), 500
            
        return jsonify({
            'success': True,
            'message': 'Backtracking analysis completed successfully',
            'ticker_count': len(backtracking_results),
            'periods': periods,
            'results': backtracking_results
        })
    except Exception as e:
        logger.error(f"Error performing backtracking analysis: {str(e)}")
        return jsonify({'error': f'Error during backtracking: {str(e)}'}), 500

@app.route('/api/risk-based-recommendations', methods=['POST'])
def risk_based_recommendations():
    """API endpoint for generating risk-based investment recommendations."""
    global cached_data, cached_predictions
    
    if cached_data is None:
        return jsonify({'error': 'No data available. Please upload or fetch data first.'}), 400
    
    # Get parameters from request
    data = request.get_json() or {}
    ticker = data.get('ticker')  # Optional - filter by ticker
    risk_appetite = data.get('risk_appetite', 'moderate')  # 'conservative', 'moderate', or 'aggressive'
    
    # Validate risk appetite
    if risk_appetite not in ['conservative', 'moderate', 'aggressive']:
        risk_appetite = 'moderate'  # Default to moderate
    
    try:
        # Get ticker-specific data if specified
        if ticker and 'ticker' in cached_data.columns:
            ticker_data = cached_data[cached_data['ticker'] == ticker].copy()
            if ticker_data.empty:
                return jsonify({'error': f'No data available for ticker: {ticker}'}), 404
        else:
            ticker_data = cached_data.copy()
        
        # Load models
        lstm_models = model_trainer.load_lstm_models()
        if not lstm_models or not lstm_models.get('models'):
            # Try training models first
            lstm_models = model_trainer.train_lstm_models(cached_data)
            if not lstm_models or not lstm_models.get('models'):
                return jsonify({'error': 'Failed to load or train models'}), 500
        
        # Define risk thresholds for different risk appetites
        risk_thresholds = {
            'conservative': {'min_return': 3, 'max_loss': 2, 'timeframe': 'short_term'},
            'moderate': {'min_return': 5, 'max_loss': 5, 'timeframe': 'medium_term'},
            'aggressive': {'min_return': 10, 'max_loss': 10, 'timeframe': 'long_term'}
        }
        
        # Define timeframes
        timeframes = {
            'short_term': {'days': 7, 'name': '1 Week'},
            'medium_term': {'days': 30, 'name': '1 Month'},
            'long_term': {'days': 90, 'name': '3 Months'}
        }
        
        recommendations = {}
        
        for ticker_name, ticker_group in ticker_data.groupby('ticker'):
            # Skip if no model for this ticker
            if ticker_name not in lstm_models['models']:
                continue
                
            ticker_model = lstm_models['models'][ticker_name]
            
            # Prepare data for prediction
            last_sequence, available_features = model_trainer.prepare_sequence_for_prediction(
                ticker_group, 
                ticker_model['features'], 
                ticker_model['scaler']
            )
            
            if last_sequence is None:
                continue
            
            # Get predictions for different timeframes
            multi_tf_predictions = {}
            
            for tf_key, tf_data in timeframes.items():
                # Predict for this timeframe
                predictions = model_trainer.predict_future(
                    ticker_model['model'],
                    last_sequence,
                    ticker_model['scaler'],
                    available_features,
                    days=tf_data['days']
                )
                
                # Store predictions
                multi_tf_predictions[tf_key] = {
                    'days': tf_data['days'],
                    'name': tf_data['name'],
                    'predictions': [float(p) for p in predictions],
                    'final_price': float(predictions[-1]),
                    'change_pct': float(((predictions[-1] / predictions[0]) - 1) * 100)
                }
            
            # Last available price
            last_price = ticker_group['Price'].iloc[-1]
            
            # Calculate expected returns for each timeframe
            returns = {}
            for tf, tf_data in multi_tf_predictions.items():
                returns[tf] = float(((tf_data['final_price'] / last_price) - 1) * 100)
            
            # Get threshold for this risk appetite
            threshold = risk_thresholds[risk_appetite]
            
            # Preferred timeframe for this risk appetite
            preferred_tf = threshold['timeframe']
            
            # Make recommendation based on risk appetite
            recommendation = {
                'ticker': ticker_name,
                'risk_appetite': risk_appetite,
                'last_price': float(last_price),
                'last_date': ticker_group['Date'].iloc[-1].strftime('%Y-%m-%d'),
                'preferred_timeframe': preferred_tf,
                'preferred_timeframe_name': timeframes[preferred_tf]['name'],
                'expected_returns': returns,
                'action': 'HOLD'  # Default action
            }
            
            # Decide action based on expected return in preferred timeframe
            expected_return = returns[preferred_tf]
            
            if expected_return >= threshold['min_return']:
                recommendation['action'] = 'BUY'
                recommendation['reason'] = f"Expected return of {expected_return:.2f}% exceeds minimum threshold of {threshold['min_return']}% for {risk_appetite} risk appetite"
            elif expected_return <= -threshold['max_loss']:
                recommendation['action'] = 'SELL'
                recommendation['reason'] = f"Expected loss of {abs(expected_return):.2f}% exceeds maximum threshold of {threshold['max_loss']}% for {risk_appetite} risk appetite"
            else:
                recommendation['reason'] = f"Expected return of {expected_return:.2f}% is within thresholds for {risk_appetite} risk appetite"
            
            recommendations[ticker_name] = recommendation
        
        if not recommendations:
            return jsonify({'error': 'No valid recommendations generated for any ticker'}), 500
        
        # Get actions counts
        action_counts = {}
        for ticker, rec in recommendations.items():
            action = rec['action']
            if action in action_counts:
                action_counts[action] += 1
            else:
                action_counts[action] = 1
            
        return jsonify({
            'success': True,
            'message': 'Risk-based recommendations generated successfully',
            'risk_appetite': risk_appetite,
            'ticker_count': len(recommendations),
            'action_summary': action_counts,
            'recommendations': recommendations
        })
    except Exception as e:
        logger.error(f"Error generating risk-based recommendations: {str(e)}")
        return jsonify({'error': f'Error generating recommendations: {str(e)}'}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """API endpoint to check if the server is running."""
    return jsonify({
        'status': 'ok',
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'has_data': cached_data is not None,
        'has_predictions': cached_predictions is not None,
        'version': '1.0.0'
    })

@app.route('/api/get-predictions', methods=['GET'])
def get_predictions():
    """API endpoint to retrieve cached predictions."""
    global cached_predictions
    
    # Get parameters
    ticker = request.args.get('ticker')
    
    if cached_predictions is None or cached_predictions.empty:
        return jsonify({'error': 'No predictions available. Please train models first.'}), 404
    
    try:
        # Filter by ticker if specified
        if ticker and 'ticker' in cached_predictions.columns:
            ticker_preds = cached_predictions[cached_predictions['ticker'] == ticker]
            if ticker_preds.empty:
                return jsonify({'error': f'No predictions found for ticker: {ticker}'}), 404
            predictions_to_return = ticker_preds
        else:
            predictions_to_return = cached_predictions
        
        # Convert to dict for JSON serialization
        predictions_dict = predictions_to_return.to_dict(orient='records')
        
        return jsonify({
            'success': True,
            'message': 'Predictions retrieved successfully',
            'count': len(predictions_dict),
            'predictions': predictions_dict
        })
    except Exception as e:
        logger.error(f"Error retrieving predictions: {str(e)}")
        return jsonify({'error': f'Error retrieving predictions: {str(e)}'}), 500

@app.route('/api/peg-analysis', methods=['POST'])
def peg_analysis():
    """API endpoint for performing PEG ratio analysis."""
    global cached_data
    
    if cached_data is None:
        return jsonify({'error': 'No data available. Please upload or fetch data first.'}), 400
    
    # Get parameters from request
    data = request.get_json() or {}
    ticker = data.get('ticker')  # Optional - filter by ticker
    eps_data = data.get('eps_data')  # Optional - eps data to add to the dataset
    
    try:
        # Get ticker-specific data if specified
        if ticker and 'ticker' in cached_data.columns:
            ticker_data = cached_data[cached_data['ticker'] == ticker].copy()
            if ticker_data.empty:
                return jsonify({'error': f'No data available for ticker: {ticker}'}), 404
        else:
            ticker_data = cached_data.copy()
        
        # Check if we have or can add EPS data
        has_eps_data = False
        
        # If EPS data provided, add it to the dataset
        if eps_data:
            # Example format: {'TICKER1': {'EPS': 2.5, 'EPS Growth %': 15.2}, ...}
            for ticker_name, eps_info in eps_data.items():
                if ticker_name in ticker_data['ticker'].values:
                    # Add EPS data to rows with this ticker
                    ticker_mask = ticker_data['ticker'] == ticker_name
                    
                    if 'EPS' in eps_info:
                        ticker_data.loc[ticker_mask, 'EPS'] = eps_info['EPS']
                    
                    if 'EPS Growth %' in eps_info:
                        ticker_data.loc[ticker_mask, 'EPS Growth %'] = eps_info['EPS Growth %']
            
            has_eps_data = True
        else:
            # Check if we already have EPS data
            has_eps_data = 'EPS' in ticker_data.columns and 'EPS Growth %' in ticker_data.columns
        
        if not has_eps_data:
            return jsonify({
                'error': 'EPS data not available. Please provide EPS data via the eps_data parameter.',
                'required_format': {
                    'ticker1': {'EPS': 2.5, 'EPS Growth %': 15.2},
                    'ticker2': {'EPS': 3.1, 'EPS Growth %': 8.7}
                }
            }), 400
        
        # Calculate P/E and PEG ratios
        peg_results = {}
        
        for ticker_name, ticker_group in ticker_data.groupby('ticker'):
            # Skip if no EPS data for this ticker
            if ticker_group['EPS'].isnull().all() or ticker_group['EPS Growth %'].isnull().all():
                continue
            
            # Get latest price
            latest_price = ticker_group['Price'].iloc[-1]
            
            # Get EPS and EPS Growth
            eps = ticker_group['EPS'].iloc[-1]
            eps_growth = ticker_group['EPS Growth %'].iloc[-1]
            
            # Calculate P/E ratio
            pe_ratio = latest_price / eps if eps > 0 else None
            
            # Calculate PEG ratio
            peg_ratio = pe_ratio / eps_growth if pe_ratio and eps_growth > 0 else None
            
            # Store results
            peg_results[ticker_name] = {
                'ticker': ticker_name,
                'last_price': float(latest_price),
                'eps': float(eps),
                'eps_growth_pct': float(eps_growth),
                'pe_ratio': float(pe_ratio) if pe_ratio else None,
                'peg_ratio': float(peg_ratio) if peg_ratio else None
            }
            
            # Add analysis
            if peg_ratio:
                if peg_ratio < 1:
                    peg_results[ticker_name]['valuation'] = 'Undervalued'
                    peg_results[ticker_name]['analysis'] = f"PEG ratio of {peg_ratio:.2f} suggests the stock is potentially undervalued relative to its growth rate"
                elif peg_ratio < 1.5:
                    peg_results[ticker_name]['valuation'] = 'Fair Value'
                    peg_results[ticker_name]['analysis'] = f"PEG ratio of {peg_ratio:.2f} suggests the stock is reasonably valued relative to its growth rate"
                else:
                    peg_results[ticker_name]['valuation'] = 'Overvalued'
                    peg_results[ticker_name]['analysis'] = f"PEG ratio of {peg_ratio:.2f} suggests the stock may be overvalued relative to its growth rate"
        
        if not peg_results:
            return jsonify({'error': 'No valid PEG analysis results generated for any ticker'}), 500
            
        # Get valuation counts
        valuation_counts = {}
        for ticker, result in peg_results.items():
            if 'valuation' in result:
                valuation = result['valuation']
                if valuation in valuation_counts:
                    valuation_counts[valuation] += 1
                else:
                    valuation_counts[valuation] = 1
        
        return jsonify({
            'success': True,
            'message': 'PEG ratio analysis completed successfully',
            'ticker_count': len(peg_results),
            'valuation_summary': valuation_counts,
            'results': peg_results
        })
    except Exception as e:
        logger.error(f"Error performing PEG analysis: {str(e)}")
        return jsonify({'error': f'Error during PEG analysis: {str(e)}'}), 500

@app.route('/api/analyze-risk', methods=['POST'])
def analyze_risk():
    """API endpoint for analyzing risk profiles."""
    global cached_data, cached_predictions, cached_risk_profiles
    
    if cached_data is None:
        return jsonify({'error': 'No data available. Please upload or fetch data first.'}), 400
    
    if cached_predictions is None:
        return jsonify({'error': 'No predictions available. Please train models first.'}), 400
    
    try:
        # Analyze risk profiles
        cached_risk_profiles = risk_analyzer.classify(cached_data, cached_predictions)
        
        # Prepare and return results
        risk_counts = cached_risk_profiles['risk_category'].value_counts().to_dict() if not cached_risk_profiles.empty else {}
        
        return jsonify({
            'success': True,
            'message': 'Risk analysis completed successfully',
            'risk_info': {
                'rows': len(cached_risk_profiles),
                'risk_categories': risk_counts
            }
        })
    except Exception as e:
        return jsonify({'error': f'Error analyzing risk: {str(e)}'}), 500

@app.route('/api/generate-recommendations', methods=['POST'])
def generate_recommendations():
    """API endpoint for generating investment recommendations."""
    global cached_risk_profiles, cached_predictions
    
    if cached_risk_profiles is None:
        return jsonify({'error': 'No risk profiles available. Please analyze risk first.'}), 400
    
    if cached_predictions is None:
        return jsonify({'error': 'No predictions available. Please train models first.'}), 400
    
    # Get parameters from request
    data = request.get_json() or {}
    user_profile = data.get('user_profile', 'moderate')  # 'conservative', 'moderate', or 'aggressive'
    max_recommendations = data.get('max_recommendations', 5)
    
    try:
        # Generate recommendations
        recommendations = recommendation_engine.generate_recommendations(
            cached_risk_profiles, 
            cached_predictions, 
            user_profile=user_profile, 
            max_recommendations=max_recommendations
        )
        
        # Create portfolio allocation if requested
        if data.get('create_portfolio', False):
            investment_amount = data.get('investment_amount', 100000)
            max_securities = data.get('max_securities', 5)
            
            portfolio = recommendation_engine.create_portfolio(
                recommendations, 
                investment_amount=investment_amount, 
                max_securities=max_securities
            )
            
            # Generate portfolio report
            portfolio_report = recommendation_engine.generate_report(
                portfolio, 
                cached_risk_profiles, 
                cached_predictions
            )
            
            # Export portfolio report if requested
            if data.get('export_report', False):
                report_format = data.get('report_format', 'json')
                report_path = report_generator.export_portfolio_report(
                    portfolio_report, 
                    format=report_format
                )
                
                # Include report file path in response
                if report_path:
                    return jsonify({
                        'success': True,
                        'message': 'Recommendations and portfolio generated successfully',
                        'recommendations_count': len(recommendations),
                        'portfolio_securities': len(portfolio),
                        'report_path': report_path,
                        'report_format': report_format
                    })
            
            # Return portfolio data without export
            return jsonify({
                'success': True,
                'message': 'Recommendations and portfolio generated successfully',
                'recommendations_count': len(recommendations),
                'portfolio_securities': len(portfolio),
                'portfolio_data': portfolio.to_dict(orient='records') if not portfolio.empty else [],
                'portfolio_summary': portfolio_report['portfolio_summary'] if portfolio_report else {}
            })
        
        # Return just the recommendations
        return jsonify({
            'success': True,
            'message': 'Recommendations generated successfully',
            'recommendations_count': len(recommendations),
            'recommendations': recommendations.to_dict(orient='records') if not recommendations.empty else []
        })
    except Exception as e:
        return jsonify({'error': f'Error generating recommendations: {str(e)}'}), 500

@app.route('/api/generate-technical-report', methods=['POST'])
def generate_technical_report():
    """API endpoint for generating a technical analysis report."""
    global cached_data, cached_risk_profiles, cached_predictions
    
    if cached_data is None:
        return jsonify({'error': 'No data available. Please upload or fetch data first.'}), 400
    
    if cached_risk_profiles is None:
        return jsonify({'error': 'No risk profiles available. Please analyze risk first.'}), 400
    
    if cached_predictions is None:
        return jsonify({'error': 'No predictions available. Please train models first.'}), 400
    
    # Get parameters from request
    data = request.get_json() or {}
    report_format = data.get('report_format', 'excel')  # 'excel' or 'pdf'
    
    try:
        # Generate technical report
        report_path = report_generator.generate_technical_report(
            cached_data, 
            cached_risk_profiles, 
            cached_predictions, 
            format=report_format
        )
        
        if report_path:
            return jsonify({
                'success': True,
                'message': 'Technical report generated successfully',
                'report_path': report_path,
                'report_format': report_format
            })
        else:
            return jsonify({'error': 'Error generating technical report'}), 500
    except Exception as e:
        return jsonify({'error': f'Error generating technical report: {str(e)}'}), 500

@app.route('/api/download-report/<path:report_name>', methods=['GET'])
def download_report(report_name):
    """API endpoint for downloading a generated report."""
    report_path = os.path.join(app.config['REPORTS_DIR'], report_name)
    
    if not os.path.exists(report_path):
        return jsonify({'error': 'Report file not found'}), 404
    
    try:
        return send_file(report_path, as_attachment=True)
    except Exception as e:
        return jsonify({'error': f'Error downloading report: {str(e)}'}), 500

@app.route('/api/results', methods=['GET'])
def get_results():
    results_dir = 'results'
    if not os.path.exists(results_dir):
        return jsonify({'error': 'Results directory not found'}), 404
    
    results = {}
    for file in os.listdir(results_dir):
        if file.endswith('.csv'):
            df = pd.read_csv(os.path.join(results_dir, file))
            results[file] = df.to_dict(orient='records')
    
    return jsonify(results)

@app.route('/api/download/<filename>')
def download_file(filename):
    filepath = os.path.join('results', filename)
    if os.path.exists(filepath):
        return send_file(filepath, as_attachment=True)
    return jsonify({'error': 'File not found'}), 404

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5004) 