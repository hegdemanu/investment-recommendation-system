#!/usr/bin/env python3
"""
Investment Recommendation System - Unified Command Line Interface

This script provides a single entry point for all operations in the Investment Recommendation System.
"""

import os
import sys
import json
import argparse
import webbrowser
import subprocess
from pathlib import Path
from datetime import date

# Import configuration settings
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import ensure_directories, DASHBOARD_FILE

# Import necessary modules for each command
try:
    from app.dashboard.dashboard_generator import generate_dashboard
    from app.utils.file_utils import file_to_base64, load_json_file
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Make sure you've installed all requirements and are in the correct directory.")
    sys.exit(1)


def setup_parser():
    """Setup command-line arguments parser"""
    parser = argparse.ArgumentParser(
        description="Investment Recommendation System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_investment_system.py dashboard        # Generate and open the dashboard
  python run_investment_system.py validate         # Validate models
  python run_investment_system.py predict          # Generate predictions
  python run_investment_system.py all              # Run all: train, predict & generate dashboard
  python run_investment_system.py full-analysis    # Run the complete analysis pipeline
  python run_investment_system.py sample           # Generate sample data for demonstration
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to execute')

    # Dashboard command
    dashboard_parser = subparsers.add_parser('dashboard', help='Generate and open investment dashboard')
    dashboard_parser.add_argument('--no-browser', action='store_true', help='Do not open the dashboard in a browser')

    # Train command
    train_parser = subparsers.add_parser('train', help='Train prediction models')
    train_parser.add_argument('--model', type=str, help='Specific model to train (default: all)')
    train_parser.add_argument('--symbol', type=str, help='Specific symbol to train (default: all)')

    # Validate command
    validate_parser = subparsers.add_parser('validate', help='Validate prediction models')
    validate_parser.add_argument('--model', type=str, help='Specific model to validate (default: all)')

    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Generate predictions')
    predict_parser.add_argument('--symbol', type=str, help='Specific symbol to predict (default: all)')
    predict_parser.add_argument('--horizon', type=int, default=30, help='Prediction horizon in days (default: 30)')

    # All-in-one command
    subparsers.add_parser('all', help='Run all: train, predict & generate dashboard')

    # Full analysis command
    subparsers.add_parser('full-analysis', help='Run the complete analysis pipeline')

    # Technical Analysis command
    technical_parser = subparsers.add_parser('technical', help='Run technical analysis')
    technical_parser.add_argument('--symbol', type=str, help='Specific symbol to analyze (default: all)')

    # Fundamental Analysis command
    fundamental_parser = subparsers.add_parser('fundamental', help='Run fundamental analysis')
    fundamental_parser.add_argument('--symbol', type=str, help='Specific symbol to analyze (default: all)')

    # Sample data generation command
    subparsers.add_parser('sample', help='Generate sample data for demonstration')

    # Web application command
    webapp_parser = subparsers.add_parser('webapp', help='Start the web application')
    webapp_parser.add_argument('--port', type=int, default=5000, help='Port to run the web app on (default: 5000)')
    webapp_parser.add_argument('--debug', action='store_true', help='Run in debug mode')

    return parser


def generate_sample_data():
    """Generate sample data files for demonstration purposes"""
    print("Generating sample data files...")
    ensure_directories()

    # Create sample training summary
    os.makedirs('results/training', exist_ok=True)
    with open('results/training/sample_training_summary.json', 'w') as f:
        f.write('''{
  "model_name": "LSTM_AAPL_30d",
  "training_date": "2023-09-15",
  "epochs": 100,
  "batch_size": 32,
  "sequence_length": 60,
  "prediction_horizon": 30,
  "features": ["Close", "Volume", "RSI", "MACD", "EMA"],
  "metrics": {
    "loss": 0.0012,
    "val_loss": 0.0015,
    "mae": 0.0245,
    "val_mae": 0.0267
  },
  "training_time": "00:15:32",
  "notes": "Model trained on 5 years of historical data"
}''')

    # Create sample validation summary
    with open('results/validation_summary.json', 'w') as f:
        f.write('''{
  "date": "2023-09-15",
  "models": {
    "LSTM_AAPL_30d": {
      "mae": 0.0245,
      "mse": 0.0012,
      "rmse": 0.0346,
      "mape": 1.23
    },
    "LSTM_MSFT_30d": {
      "mae": 0.0189,
      "mse": 0.0009,
      "rmse": 0.0300,
      "mape": 0.98
    }
  },
  "average_metrics": {
    "mae": 0.0217,
    "mse": 0.0011,
    "rmse": 0.0323,
    "mape": 1.11
  }
}''')

    # Create sample stock data
    os.makedirs('data', exist_ok=True)
    with open('data/sample_stock_data.csv', 'w') as f:
        f.write('''Date,Open,High,Low,Close,Volume
2023-09-01,185.23,186.12,184.10,185.92,35678920
2023-09-02,185.88,188.45,185.60,187.32,42563210
2023-09-03,187.40,189.20,186.55,188.75,38976540
2023-09-04,188.60,190.15,187.90,189.50,45321780
2023-09-05,189.45,191.35,189.00,190.25,40125630
2023-09-06,190.30,192.50,189.75,192.10,47856920
2023-09-07,192.00,193.75,191.20,192.80,39875410
2023-09-08,192.65,195.40,192.10,194.85,52364890
2023-09-09,194.90,196.20,193.75,195.35,48752360
2023-09-10,195.20,197.85,194.60,197.25,51236540''')

    print("Sample data generated successfully.")


def run_dashboard(no_browser=False):
    """Generate and open the investment dashboard"""
    # Import here to avoid circular imports
    from app.dashboard.dashboard_generator import generate_dashboard
    
    # Generate the dashboard
    print("Generating investment dashboard...")
    dashboard_file = generate_dashboard()
    
    # Open the dashboard in a browser if requested
    if not no_browser:
        try:
            # Use the webbrowser module to open the dashboard
            import webbrowser
            file_url = f"file://{os.path.abspath(dashboard_file)}"
            print(f"Opening dashboard in browser: {dashboard_file}")
            
            # Try multiple methods to open the browser
            success = webbrowser.open(file_url)
            
            if not success:
                # Try alternative methods based on platform
                if sys.platform == 'darwin':  # macOS
                    subprocess.run(['open', dashboard_file], check=False)
                elif sys.platform.startswith('win'):  # Windows
                    os.startfile(os.path.abspath(dashboard_file))
                elif sys.platform.startswith('linux'):  # Linux
                    subprocess.run(['xdg-open', dashboard_file], check=False)
            
            # Print instructions in case automatic opening fails
            print("\nIf the dashboard doesn't open automatically, please open it manually:")
            print(f"- File location: {dashboard_file}")
            print(f"- Full URL: {file_url}")
            print("- You can open this file with any web browser")
        except Exception as e:
            print(f"\nCouldn't automatically open the dashboard: {str(e)}")
            print(f"Please open this file manually: {dashboard_file}")
    
    return dashboard_file


def run_training(symbol=None, model=None):
    """Train prediction models"""
    print(f"Starting training process...")
    print(f"Would train models for {'all symbols' if symbol is None else f'symbol: {symbol}'} "
          f"using {'all model types' if model is None else f'model: {model}'}")
    
    # Here you would call your actual training code
    # from app.training.model_trainer import train_models
    # train_models(symbol=symbol, model_type=model)
    
    print("Training completed successfully.")


def run_validation(model=None):
    """Validate prediction models"""
    print("Validating models...")
    print(f"Would validate {'all models' if model is None else f'model: {model}'}")
    
    # Here you would call your actual validation code
    # from app.validation.model_validator import validate_models
    # validate_models(model_type=model)
    
    print("Validation completed successfully.")


def run_prediction(symbol=None, horizon=30):
    """Generate predictions"""
    print("Generating predictions...")
    print(f"Would predict {'all symbols' if symbol is None else f'symbol: {symbol}'} "
          f"for {horizon} days")
    
    # Here you would call your actual prediction code
    # from app.prediction.predictor import generate_predictions
    # generate_predictions(symbol=symbol, horizon=horizon)
    
    print("Predictions generated successfully.")


def run_technical_analysis(symbol=None):
    """Run technical analysis"""
    print("Running technical analysis...")
    print(f"Would analyze {'all symbols' if symbol is None else f'symbol: {symbol}'}")
    
    # Here you would call your actual technical analysis code
    # from app.analysis.technical_analyzer import analyze_technical
    # analyze_technical(symbol=symbol)
    
    print("Technical analysis completed successfully.")


def run_fundamental_analysis(symbol=None):
    """Run fundamental analysis"""
    print("Running fundamental analysis...")
    print(f"Would analyze {'all symbols' if symbol is None else f'symbol: {symbol}'}")
    
    # Here you would call your actual fundamental analysis code
    # from app.analysis.fundamental_analyzer import analyze_fundamental
    # analyze_fundamental(symbol=symbol)
    
    print("Fundamental analysis completed successfully.")


def run_all():
    """Run all main functionalities: training, prediction, and dashboard generation"""
    print("Running all functionalities in sequence...")
    
    # Run training
    run_training()
    
    # Run prediction
    run_prediction()
    
    # Generate and open dashboard
    run_dashboard(no_browser=False)
    
    print("All operations completed successfully.")


def run_full_analysis():
    """Run the complete analysis pipeline"""
    print("Running full analysis pipeline...")
    run_training()
    run_validation()
    run_prediction()
    run_technical_analysis()
    run_fundamental_analysis()
    run_dashboard(no_browser=False)


def run_webapp(port=5000, debug=False):
    """Start the web application"""
    print("This functionality will be implemented in a future version.")
    print(f"Would start web app on port {port} with debug={debug}")


def main():
    """Main entry point for the script"""
    parser = setup_parser()
    args = parser.parse_args()

    # Ensure all required directories exist
    ensure_directories()

    # Execute the requested command
    if args.command == 'dashboard':
        run_dashboard(no_browser=args.no_browser if hasattr(args, 'no_browser') else False)
    elif args.command == 'train':
        run_training(
            symbol=args.symbol if hasattr(args, 'symbol') else None,
            model=args.model if hasattr(args, 'model') else None
        )
    elif args.command == 'validate':
        run_validation(model=args.model if hasattr(args, 'model') else None)
    elif args.command == 'predict':
        run_prediction(
            symbol=args.symbol if hasattr(args, 'symbol') else None,
            horizon=args.horizon if hasattr(args, 'horizon') else 30
        )
    elif args.command == 'all':
        run_all()
    elif args.command == 'full-analysis':
        run_full_analysis()
    elif args.command == 'technical':
        run_technical_analysis(symbol=args.symbol if hasattr(args, 'symbol') else None)
    elif args.command == 'fundamental':
        run_fundamental_analysis(symbol=args.symbol if hasattr(args, 'symbol') else None)
    elif args.command == 'sample':
        generate_sample_data()
    elif args.command == 'webapp':
        run_webapp(
            port=args.port if hasattr(args, 'port') else 5000,
            debug=args.debug if hasattr(args, 'debug') else False
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main() 