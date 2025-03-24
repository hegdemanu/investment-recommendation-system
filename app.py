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
            data = data_processor.load_from_csv(filepath)
            
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
            lstm_models = model_trainer.load_lstm_models()
            if lstm_models['models']:
                cached_predictions = model_trainer.predict_lstm(lstm_models, cached_data)
                return jsonify({
                    'success': True,
                    'message': f'Using existing LSTM models for {len(lstm_models["models"])} tickers. No significant data drift detected.',
                    'predictions_info': {
                        'rows': len(cached_predictions),
                        'tickers': list(cached_predictions['ticker'].unique()) if not cached_predictions.empty else []
                    }
                })
    
    try:
        results = {}
        
        # Train LSTM models
        if model_type in ['lstm', 'all']:
            lstm_models = model_trainer.train_lstm_models(cached_data)
            cached_predictions = model_trainer.predict_lstm(lstm_models, cached_data)
            results['lstm'] = {
                'models_trained': len(lstm_models['models']),
                'predictions_generated': len(cached_predictions)
            }
        
        # Train long-term models
        if model_type in ['long_term', 'all']:
            long_term_models = long_term_predictor.fit_long_term_models(cached_data)
            cached_long_term_predictions = long_term_predictor.predict_ensemble(long_term_models, cached_data)
            results['long_term'] = {
                'arima_garch_models_trained': len(long_term_models['arima_garch']),
                'prophet_models_trained': len(long_term_models['prophet']),
                'predictions_generated': len(cached_long_term_predictions)
            }
        
        return jsonify({
            'success': True,
            'message': 'Models trained successfully',
            'results': results
        })
    except Exception as e:
        return jsonify({'error': f'Error training models: {str(e)}'}), 500

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
    app.run(debug=True, host='0.0.0.0', port=5000) 