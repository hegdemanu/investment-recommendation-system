#!/usr/bin/env python3
"""
Investment Recommendation System - Master Runner Script
Provides an easy interface to run various analysis tools
"""

import os
import sys
import argparse
import logging
import datetime
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_dependencies():
    """Check if all required dependencies are installed"""
    
    required_packages = [
        "pandas", "numpy", "matplotlib", "yfinance", 
        "prophet", "transformers", "torch", "requests"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        logger.error(f"Missing required packages: {', '.join(missing_packages)}")
        logger.error("Please install required packages using:")
        logger.error(f"pip install {' '.join(missing_packages)}")
        return False
    
    return True

def download_stock_data(args):
    """Download historical stock data"""
    from trading_engine.get_stock_data import main as get_stock_data_main
    
    # Modify sys.argv to pass arguments to the stock data downloader
    sys.argv = [
        "get_stock_data.py",
        "--tickers", args.tickers,
        "--years", str(args.years),
        "--interval", args.interval,
        "--output", args.output
    ]
    
    if args.indicators:
        sys.argv.append("--indicators")
    
    # Run the stock data downloader
    get_stock_data_main()
    
    logger.info(f"Stock data downloaded for {args.tickers} to directory {args.output}")

def run_sentiment_analysis(args):
    """Run sentiment analysis on news"""
    from trading_engine.analyze_stock_sentiment import main as sentiment_main
    
    # Check if API key is provided
    api_key = args.api_key
    if not api_key:
        api_key = os.environ.get('NEWSAPI_KEY')
        if not api_key:
            logger.error("No NewsAPI key provided. Please provide using --api_key or set NEWSAPI_KEY environment variable")
            logger.error("You can get a free API key at https://newsapi.org/")
            return
    
    # Parse ticker list
    tickers = [ticker.strip() for ticker in args.tickers.split(',')]
    
    # Run sentiment analysis for each ticker
    for ticker in tickers:
        logger.info(f"Running sentiment analysis for {ticker}")
        
        # Modify sys.argv to pass arguments
        sys.argv = [
            "analyze_stock_sentiment.py",
            "--ticker", ticker,
            "--days", str(args.news_days),
            "--api_key", api_key
        ]
        
        # Run sentiment analysis
        sentiment_main()
    
    logger.info(f"Sentiment analysis completed for {len(tickers)} tickers")

def run_integrated_analysis(args):
    """Run integrated analysis with technical, forecast, and sentiment"""
    from trading_engine.integrated_analysis import main as integrated_main
    
    # Get API key for sentiment analysis
    api_key = args.api_key
    if not api_key:
        api_key = os.environ.get('NEWSAPI_KEY')
        if not api_key and not args.skip_sentiment:
            logger.warning("No NewsAPI key provided. Sentiment analysis will be skipped.")
            args.skip_sentiment = True
    
    # Parse ticker list
    tickers = [ticker.strip() for ticker in args.tickers.split(',')]
    
    # Create results directory with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(args.output)
    if len(tickers) > 1:
        results_dir = results_dir / f"analysis_{timestamp}"
        results_dir.mkdir(exist_ok=True, parents=True)
    
    # Run integrated analysis for each ticker
    for ticker in tickers:
        logger.info(f"Running integrated analysis for {ticker}")
        
        # Create ticker-specific directory for multiple tickers
        ticker_dir = results_dir
        if len(tickers) > 1:
            ticker_dir = results_dir / ticker
            ticker_dir.mkdir(exist_ok=True)
        
        # Modify sys.argv to pass arguments
        sys.argv = [
            "integrated_analysis.py",
            "--ticker", ticker,
            "--years", str(args.years),
            "--days", str(args.forecast_days),
            "--news_days", str(args.news_days),
            "--output", str(ticker_dir)
        ]
        
        if api_key and not args.skip_sentiment:
            sys.argv.extend(["--api_key", api_key])
        
        if args.skip_sentiment:
            sys.argv.append("--skip_sentiment")
        
        # Run integrated analysis
        integrated_main()
    
    # Create summary report for multiple tickers
    if len(tickers) > 1:
        create_summary_report(tickers, results_dir)
    
    logger.info(f"Integrated analysis completed for {len(tickers)} tickers")
    logger.info(f"Results saved to {results_dir}")

def create_summary_report(tickers, results_dir):
    """Create a summary report of recommendations for all tickers"""
    import pandas as pd
    import re
    
    # Collect recommendations from individual reports
    recommendations = []
    
    for ticker in tickers:
        report_path = results_dir / ticker / f"{ticker}_analysis_report.txt"
        
        if not report_path.exists():
            logger.warning(f"No report found for {ticker}")
            continue
        
        # Read the report file
        with open(report_path, 'r') as f:
            report_text = f.read()
        
        # Extract key information using regex
        try:
            # Extract recommendation
            rec_match = re.search(r"Recommendation: (\w+(?:\s+\w+)?)", report_text)
            recommendation = rec_match.group(1) if rec_match else "N/A"
            
            # Extract confidence
            conf_match = re.search(r"Confidence: (\w+)", report_text)
            confidence = conf_match.group(1) if conf_match else "N/A"
            
            # Extract overall score
            score_match = re.search(r"Overall Score: ([-+]?\d*\.\d+|\d+)", report_text)
            score = float(score_match.group(1)) if score_match else 0.0
            
            # Extract current price
            price_match = re.search(r"Current Price: \$([-+]?\d*\.\d+|\d+)", report_text)
            price = float(price_match.group(1)) if price_match else 0.0
            
            # Extract forecasted price
            forecast_match = re.search(r"Forecasted Price: \$([-+]?\d*\.\d+|\d+)", report_text)
            forecast_price = float(forecast_match.group(1)) if forecast_match else 0.0
            
            # Extract expected return
            return_match = re.search(r"Expected Return: ([-+]?\d*\.\d+|\d+)", report_text)
            expected_return = float(return_match.group(1)) if return_match else 0.0
            
            # Extract MA signal
            ma_match = re.search(r"Moving Averages: (\w+(?:\s+\w+)?)", report_text)
            ma_signal = ma_match.group(1) if ma_match else "N/A"
            
            # Extract sentiment (if available)
            sentiment_match = re.search(r"Overall Market Sentiment: (\w+(?:\s+\w+)?)", report_text)
            sentiment = sentiment_match.group(1) if sentiment_match else "N/A"
            
            # Add to recommendations list
            recommendations.append({
                'Ticker': ticker,
                'Recommendation': recommendation,
                'Confidence': confidence,
                'Score': score,
                'Current Price': price,
                'Forecast Price': forecast_price,
                'Expected Return (%)': expected_return,
                'MA Signal': ma_signal,
                'Sentiment': sentiment
            })
            
        except Exception as e:
            logger.warning(f"Error parsing report for {ticker}: {e}")
    
    if not recommendations:
        logger.warning("No recommendations found to create summary")
        return
    
    # Create DataFrame and sort by score
    df = pd.DataFrame(recommendations)
    df = df.sort_values('Score', ascending=False)
    
    # Save summary as CSV
    csv_path = results_dir / "recommendations_summary.csv"
    df.to_csv(csv_path, index=False)
    
    # Save summary as text report
    txt_path = results_dir / "recommendations_summary.txt"
    with open(txt_path, 'w') as f:
        f.write("Investment Recommendations Summary\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d')}\n")
        f.write(f"Tickers Analyzed: {len(df)}\n\n")
        
        f.write("TOP RECOMMENDATIONS:\n")
        f.write("-" * 30 + "\n")
        
        # Write top BUY recommendations
        buy_recs = df[df['Recommendation'].isin(['STRONG BUY', 'BUY'])].head(5)
        if not buy_recs.empty:
            f.write("BUY RECOMMENDATIONS:\n")
            for _, row in buy_recs.iterrows():
                f.write(f"  {row['Ticker']}: {row['Recommendation']} (Confidence: {row['Confidence']}, Return: {row['Expected Return (%)']:.2f}%)\n")
            f.write("\n")
        
        # Write top SELL recommendations
        sell_recs = df[df['Recommendation'].isin(['STRONG SELL', 'SELL'])].head(5)
        if not sell_recs.empty:
            f.write("SELL RECOMMENDATIONS:\n")
            for _, row in sell_recs.iterrows():
                f.write(f"  {row['Ticker']}: {row['Recommendation']} (Confidence: {row['Confidence']}, Return: {row['Expected Return (%)']:.2f}%)\n")
            f.write("\n")
        
        # Write all recommendations in a table format
        f.write("ALL RECOMMENDATIONS:\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Ticker':<10} {'Recommendation':<15} {'Confidence':<10} {'Score':<8} {'Current':<10} {'Forecast':<10} {'Return %':<10} {'MA Signal':<15} {'Sentiment':<15}\n")
        f.write("-" * 80 + "\n")
        
        for _, row in df.iterrows():
            f.write(f"{row['Ticker']:<10} {row['Recommendation']:<15} {row['Confidence']:<10} {row['Score']:<8.2f} ")
            f.write(f"${row['Current Price']:<8.2f} ${row['Forecast Price']:<8.2f} {row['Expected Return (%)']:<8.2f}% ")
            f.write(f"{row['MA Signal']:<15} {row['Sentiment']:<15}\n")
    
    logger.info(f"Summary report saved to {txt_path}")
    logger.info(f"Summary CSV saved to {csv_path}")

def main():
    parser = argparse.ArgumentParser(description="Investment Recommendation System")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Common arguments for all commands
    common_parser = argparse.ArgumentParser(add_help=False)
    common_parser.add_argument("--tickers", type=str, default="AAPL", 
                             help="Stock ticker symbols (comma-separated for multiple)")
    common_parser.add_argument("--output", type=str, default="results",
                             help="Output directory for results")
    
    # Download stock data command
    download_parser = subparsers.add_parser("download", parents=[common_parser],
                                         help="Download historical stock data")
    download_parser.add_argument("--years", type=int, default=5,
                               help="Number of years of historical data to download")
    download_parser.add_argument("--interval", type=str, default="1d", choices=["1d", "1wk", "1mo"],
                               help="Data interval (1d=daily, 1wk=weekly, 1mo=monthly)")
    download_parser.add_argument("--indicators", action="store_true",
                               help="Calculate technical indicators")
    
    # Sentiment analysis command
    sentiment_parser = subparsers.add_parser("sentiment", parents=[common_parser],
                                          help="Run sentiment analysis on news")
    sentiment_parser.add_argument("--api_key", type=str,
                                help="NewsAPI API key (or set NEWSAPI_KEY environment variable)")
    sentiment_parser.add_argument("--news_days", type=int, default=7,
                                help="Number of days of news to analyze")
    
    # Integrated analysis command
    integrated_parser = subparsers.add_parser("analyze", parents=[common_parser],
                                           help="Run integrated analysis with technical, forecast, and sentiment")
    integrated_parser.add_argument("--years", type=int, default=5,
                                 help="Years of historical data to analyze")
    integrated_parser.add_argument("--forecast_days", type=int, default=30,
                                 help="Days to forecast")
    integrated_parser.add_argument("--news_days", type=int, default=7,
                                 help="Days of news to analyze")
    integrated_parser.add_argument("--api_key", type=str,
                                 help="NewsAPI API key (or set NEWSAPI_KEY environment variable)")
    integrated_parser.add_argument("--skip_sentiment", action="store_true",
                                 help="Skip sentiment analysis")
    
    args = parser.parse_args()
    
    # Check if dependencies are installed
    if not check_dependencies():
        sys.exit(1)
    
    # Call appropriate function based on command
    if args.command == "download":
        download_stock_data(args)
    elif args.command == "sentiment":
        run_sentiment_analysis(args)
    elif args.command == "analyze":
        run_integrated_analysis(args)
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main() 