#!/usr/bin/env python3
"""
Main entry point for the Investment Recommendation System
"""

import os
import sys
import click
import asyncio
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add project root to Python path
sys.path.append(str(Path(__file__).parent))

# Import configuration
from config.settings import (
    ensure_directories,
    API_CONFIG,
    ENV,
    DEBUG
)

# Make sure required directories exist
ensure_directories()

@click.group()
def cli():
    """Investment Recommendation System CLI"""
    pass

@cli.command()
@click.option('--host', default=API_CONFIG['host'], help='API host')
@click.option('--port', default=API_CONFIG['port'], help='API port')
@click.option('--debug', is_flag=True, help='Enable debug mode')
def api(host, port, debug):
    """Run the Flask server"""
    click.echo(f"Starting API server on {host}:{port}")
    
    # Import and run Flask app
    from app.api.main import app
    app.run(host=host, port=port, debug=debug)

@cli.command()
@click.option('--symbol', help='Stock symbol to analyze')
@click.option('--days', default=30, help='Number of days to predict')
@click.option('--include-sentiment', is_flag=True, help='Include sentiment analysis')
def predict(symbol, days, include_sentiment):
    """Generate predictions for a stock"""
    from app.models.prediction_pipeline import PredictionPipeline
    from app.models.sentiment_pipeline import SentimentAnalyzer
    
    async def run_prediction():
        pipeline = PredictionPipeline()
        analyzer = SentimentAnalyzer() if include_sentiment else None
        
        click.echo(f"Generating predictions for {symbol} ({days} days)")
        predictions = await pipeline.generate_predictions(
            symbol=symbol,
            horizon=days,
            sentiment_analyzer=analyzer
        )
        
        click.echo("Predictions generated successfully")
        return predictions
    
    asyncio.run(run_prediction())

@cli.command()
@click.option('--symbol', help='Stock symbol to train models for')
@click.option('--start-date', help='Training start date (YYYY-MM-DD)')
@click.option('--end-date', help='Training end date (YYYY-MM-DD)')
def train(symbol, start_date, end_date):
    """Train prediction models"""
    from app.models.training_pipeline import ModelTrainingPipeline
    
    pipeline = ModelTrainingPipeline()
    click.echo(f"Training models for {symbol}")
    
    results = pipeline.train_all_models(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date
    )
    
    click.echo("Training completed successfully")
    return results

@cli.command()
@click.option('--symbols', multiple=True, help='Stock symbols to update data for')
@click.option('--force', is_flag=True, help='Force update even if cache is valid')
def update_data(symbols, force):
    """Update market data"""
    from app.data.market_data import MarketDataPipeline
    
    async def run_update():
        pipeline = MarketDataPipeline()
        click.echo(f"Updating market data for {len(symbols)} symbols")
        
        await pipeline.update_market_data(
            symbols=symbols,
            force_update=force
        )
        
        click.echo("Market data updated successfully")
    
    asyncio.run(run_update())

@cli.command()
@click.option('--text', help='Text to analyze sentiment for')
def analyze_sentiment(text):
    """Analyze sentiment of text"""
    from app.models.sentiment_pipeline import SentimentAnalyzer
    
    async def run_analysis():
        analyzer = SentimentAnalyzer()
        click.echo("Analyzing sentiment...")
        
        result = await analyzer.analyze_text(text)
        click.echo(f"Sentiment: {result['sentiment']} (confidence: {result['confidence']:.2f})")
        
        return result
    
    asyncio.run(run_analysis())

@cli.command()
def setup():
    """Setup the investment system"""
    # Create required directories
    ensure_directories()
    click.echo("Created required directories")
    
    # Create .env file if it doesn't exist
    env_template = Path('.env.template')
    env_file = Path('.env')
    
    if not env_file.exists() and env_template.exists():
        env_file.write_text(env_template.read_text())
        click.echo("Created .env file from template")
    
    click.echo("Setup completed successfully")

if __name__ == '__main__':
    cli() 