#!/usr/bin/env python3

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import seaborn as sns

def load_predictions_summary():
    """Load the predictions summary from the JSON file"""
    with open('results/predictions/predictions_summary.json', 'r') as f:
        return json.load(f)

def categorize_by_return(predictions):
    """Categorize stocks based on expected returns"""
    high_return = []
    moderate_return = []
    low_return = []
    negative_return = []
    
    for ticker, data in predictions.items():
        expected_return = data['expected_return_percent']
        if expected_return > 5:
            high_return.append((ticker, expected_return))
        elif expected_return > 2:
            moderate_return.append((ticker, expected_return))
        elif expected_return >= 0:
            low_return.append((ticker, expected_return))
        else:
            negative_return.append((ticker, expected_return))
    
    return {
        'high_return': sorted(high_return, key=lambda x: x[1], reverse=True),
        'moderate_return': sorted(moderate_return, key=lambda x: x[1], reverse=True),
        'low_return': sorted(low_return, key=lambda x: x[1], reverse=True),
        'negative_return': sorted(negative_return, key=lambda x: x[1])
    }

def calculate_risk_metrics(predictions):
    """Calculate risk metrics for each stock using prediction data"""
    risk_metrics = {}
    
    for ticker, data in predictions.items():
        csv_path = data['prediction_data'].replace('./', '')
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            # Calculate volatility (standard deviation of daily returns)
            if 'Predicted Price' in df.columns:
                df['Daily Return'] = df['Predicted Price'].pct_change()
                volatility = df['Daily Return'].std() * 100  # Convert to percentage
                
                # Calculate downside deviation (only negative returns)
                downside_returns = df['Daily Return'][df['Daily Return'] < 0]
                downside_deviation = downside_returns.std() * 100 if len(downside_returns) > 0 else 0
                
                risk_metrics[ticker] = {
                    'volatility': volatility,
                    'downside_deviation': downside_deviation,
                    'expected_return': data['expected_return_percent']
                }
    
    return risk_metrics

def analyze_best_prediction_horizons(predictions):
    """Analyze which prediction horizons worked best for different stocks"""
    horizons = {}
    for ticker, data in predictions.items():
        horizon = data['best_horizon_days']
        if horizon not in horizons:
            horizons[horizon] = []
        horizons[horizon].append(ticker)
    
    return horizons

def create_risk_adjusted_recommendations(risk_metrics):
    """Create investment recommendations based on risk appetite"""
    # Calculate Sharpe-like ratio (expected return / volatility)
    for ticker in risk_metrics:
        if risk_metrics[ticker]['volatility'] > 0:
            risk_metrics[ticker]['risk_adjusted_return'] = risk_metrics[ticker]['expected_return'] / risk_metrics[ticker]['volatility']
        else:
            risk_metrics[ticker]['risk_adjusted_return'] = 0
    
    # Sort by different metrics for different risk profiles
    conservative = sorted(risk_metrics.items(), key=lambda x: (
        0 if x[1]['expected_return'] < 0 else x[1]['risk_adjusted_return']
    ), reverse=True)
    
    moderate = sorted(risk_metrics.items(), key=lambda x: (
        0 if x[1]['expected_return'] < 0 else
        x[1]['expected_return'] * 0.6 + x[1]['risk_adjusted_return'] * 0.4
    ), reverse=True)
    
    aggressive = sorted(risk_metrics.items(), key=lambda x: (
        x[1]['expected_return']
    ), reverse=True)
    
    return {
        'conservative': [t[0] for t in conservative if t[1]['expected_return'] > 0][:3],
        'moderate': [t[0] for t in moderate if t[1]['expected_return'] > 0][:3],
        'aggressive': [t[0] for t in aggressive if t[1]['expected_return'] > 0][:3]
    }

def plot_risk_return_profile(risk_metrics):
    """Create a risk-return scatter plot"""
    tickers = list(risk_metrics.keys())
    volatility = [risk_metrics[t]['volatility'] for t in tickers]
    returns = [risk_metrics[t]['expected_return'] for t in tickers]
    
    plt.figure(figsize=(12, 8))
    
    # Use a color map based on the return to risk ratio
    colors = [max(0, r) / max(v, 0.001) for r, v in zip(returns, volatility)]
    
    scatter = plt.scatter(volatility, returns, c=colors, cmap='viridis', s=100, alpha=0.7)
    plt.colorbar(scatter, label='Return/Risk Ratio')
    
    # Add ticker labels
    for i, ticker in enumerate(tickers):
        plt.annotate(ticker, (volatility[i], returns[i]), 
                    textcoords="offset points", 
                    xytext=(0, 5), 
                    ha='center')
    
    plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
    plt.axvline(x=0, color='r', linestyle='-', alpha=0.3)
    
    plt.title('Risk-Return Profile of Predicted Stocks')
    plt.xlabel('Volatility (Daily Returns Std Dev %)')
    plt.ylabel('Expected Return (%)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the plot
    os.makedirs('results/analysis', exist_ok=True)
    plt.savefig('results/analysis/risk_return_profile.png')
    plt.close()

def plot_return_comparison(predictions):
    """Create a bar chart comparing expected returns"""
    tickers = list(predictions.keys())
    returns = [predictions[t]['expected_return_percent'] for t in tickers]
    
    # Sort by return
    sorted_data = sorted(zip(tickers, returns), key=lambda x: x[1], reverse=True)
    sorted_tickers, sorted_returns = zip(*sorted_data)
    
    # Create a color map based on return values
    colors = ['green' if r > 0 else 'red' for r in sorted_returns]
    
    plt.figure(figsize=(12, 8))
    bars = plt.bar(sorted_tickers, sorted_returns, color=colors, alpha=0.7)
    
    # Add values on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.,
                 height + 0.3,
                 f'{height:.2f}%',
                 ha='center', va='bottom', rotation=0)
    
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.title('Predicted 30-Day Returns by Stock')
    plt.xlabel('Stock Ticker')
    plt.ylabel('Expected Return (%)')
    plt.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    os.makedirs('results/analysis', exist_ok=True)
    plt.savefig('results/analysis/return_comparison.png')
    plt.close()

def generate_html_report(predictions, risk_metrics, categorized_returns, recommendations, horizon_analysis):
    """Generate an HTML report with all the analysis"""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Investment Recommendation Report</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                line-height: 1.6;
                margin: 0;
                padding: 20px;
                color: #333;
            }}
            h1, h2, h3 {{
                color: #2c3e50;
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
            }}
            .header {{
                background: #f8f9fa;
                padding: 20px;
                border-radius: 5px;
                margin-bottom: 20px;
            }}
            .section {{
                margin-bottom: 30px;
                padding: 20px;
                background: #fff;
                border-radius: 5px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin-bottom: 20px;
            }}
            th, td {{
                padding: 12px 15px;
                text-align: left;
                border-bottom: 1px solid #ddd;
            }}
            th {{
                background-color: #f2f2f2;
            }}
            tr:hover {{
                background-color: #f5f5f5;
            }}
            .positive {{
                color: green;
            }}
            .negative {{
                color: red;
            }}
            .recommendation {{
                padding: 15px;
                margin-bottom: 10px;
                border-radius: 5px;
            }}
            .conservative {{
                background-color: #e8f4f8;
                border-left: 4px solid #4b86b4;
            }}
            .moderate {{
                background-color: #edf7ed;
                border-left: 4px solid #63c461;
            }}
            .aggressive {{
                background-color: #fff8e8;
                border-left: 4px solid #ffba49;
            }}
            .img-container {{
                margin: 20px 0;
                text-align: center;
            }}
            img {{
                max-width: 100%;
                height: auto;
                box-shadow: 0 3px 10px rgba(0,0,0,0.1);
                border-radius: 5px;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>Investment Recommendation Report</h1>
                <p>Generated on: {now}</p>
                <p>This report provides an analysis of predicted stock performance over the next 30 days.</p>
            </div>
            
            <div class="section">
                <h2>Executive Summary</h2>
                <p>Based on our predictive models, we have analyzed 12 stocks and categorized them according to expected returns and risk profiles.</p>
                <p>Top performers: {', '.join([f"{t[0]} ({t[1]:.2f}%)" for t in categorized_returns['high_return'][:3]])} show the highest expected returns.</p>
                <p>Stocks with negative expected returns: {', '.join([f"{t[0]} ({t[1]:.2f}%)" for t in categorized_returns['negative_return']])}.</p>
            </div>
            
            <div class="section">
                <h2>Risk-Return Analysis</h2>
                <div class="img-container">
                    <img src="results/analysis/risk_return_profile.png" alt="Risk-Return Profile">
                </div>
                <div class="img-container">
                    <img src="results/analysis/return_comparison.png" alt="Return Comparison">
                </div>
            </div>
            
            <div class="section">
                <h2>Detailed Stock Predictions</h2>
                <table>
                    <tr>
                        <th>Stock</th>
                        <th>Starting Price</th>
                        <th>Predicted Price (30d)</th>
                        <th>Expected Return</th>
                        <th>Volatility</th>
                        <th>Best Prediction Horizon</th>
                    </tr>
    """
    
    # Sort tickers by expected return
    sorted_tickers = sorted(predictions.keys(), 
                          key=lambda t: predictions[t]['expected_return_percent'], 
                          reverse=True)
    
    for ticker in sorted_tickers:
        data = predictions[ticker]
        risk_data = risk_metrics.get(ticker, {'volatility': 'N/A', 'downside_deviation': 'N/A'})
        return_class = "positive" if data['expected_return_percent'] >= 0 else "negative"
        
        html += f"""
                    <tr>
                        <td>{ticker}</td>
                        <td>{data['start_price']:.2f}</td>
                        <td>{data['end_price']:.2f}</td>
                        <td class="{return_class}">{data['expected_return_percent']:.2f}%</td>
                        <td>{risk_data['volatility'] if isinstance(risk_data['volatility'], str) else f"{risk_data['volatility']:.2f}%"}</td>
                        <td>{data['best_horizon_days']} days</td>
                    </tr>
        """
    
    html += """
                </table>
            </div>
            
            <div class="section">
                <h2>Investment Recommendations by Risk Profile</h2>
    """
    
    html += f"""
                <div class="recommendation conservative">
                    <h3>Conservative Investor</h3>
                    <p>For investors with low risk tolerance, focusing on stable returns with lower volatility:</p>
                    <ul>
                        {"".join([f"<li>{ticker} - Expected Return: {predictions[ticker]['expected_return_percent']:.2f}%, Risk-Adjusted Return: {risk_metrics[ticker]['risk_adjusted_return']:.2f}</li>" for ticker in recommendations['conservative']])}
                    </ul>
                </div>
                
                <div class="recommendation moderate">
                    <h3>Moderate Investor</h3>
                    <p>For investors seeking a balance between risk and return:</p>
                    <ul>
                        {"".join([f"<li>{ticker} - Expected Return: {predictions[ticker]['expected_return_percent']:.2f}%, Risk-Adjusted Return: {risk_metrics[ticker]['risk_adjusted_return']:.2f}</li>" for ticker in recommendations['moderate']])}
                    </ul>
                </div>
                
                <div class="recommendation aggressive">
                    <h3>Aggressive Investor</h3>
                    <p>For investors willing to accept higher volatility for potentially higher returns:</p>
                    <ul>
                        {"".join([f"<li>{ticker} - Expected Return: {predictions[ticker]['expected_return_percent']:.2f}%, Risk-Adjusted Return: {risk_metrics[ticker]['risk_adjusted_return']:.2f}</li>" for ticker in recommendations['aggressive']])}
                    </ul>
                </div>
    """
    
    html += """
            </div>
            
            <div class="section">
                <h2>Model Prediction Horizons Analysis</h2>
                <p>Our analysis shows which prediction horizons worked best for different stocks:</p>
    """
    
    for horizon, tickers in horizon_analysis.items():
        html += f"""
                <h3>{horizon}-Day Horizon</h3>
                <p>These stocks performed best with a {horizon}-day prediction horizon: {', '.join(tickers)}</p>
        """
    
    html += """
            </div>
            
            <div class="section">
                <h2>Methodology</h2>
                <p>Predictions were generated using LSTM (Long Short-Term Memory) models trained on historical price data. The models were trained on several months of data and optimized for the best prediction horizon for each stock.</p>
                <p>Risk metrics were calculated based on the volatility of predicted daily returns.</p>
                <p>Recommendations are based on a combination of expected returns, volatility, and risk-adjusted performance metrics.</p>
            </div>
            
            <div class="section">
                <h2>Disclaimer</h2>
                <p>This report is for informational purposes only and does not constitute investment advice. Past performance and predictions are not guarantees of future results. Always conduct your own research before making investment decisions.</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    os.makedirs('results/analysis', exist_ok=True)
    with open('results/analysis/investment_report.html', 'w') as f:
        f.write(html)
    
    print(f"HTML report generated at results/analysis/investment_report.html")

def main():
    # Create analysis directory if it doesn't exist
    os.makedirs('results/analysis', exist_ok=True)
    
    # Load prediction data
    print("Loading prediction data...")
    predictions = load_predictions_summary()
    
    # Calculate risk metrics
    print("Calculating risk metrics...")
    risk_metrics = calculate_risk_metrics(predictions)
    
    # Categorize stocks by expected return
    print("Categorizing stocks by expected return...")
    categorized_returns = categorize_by_return(predictions)
    
    # Analyze best prediction horizons
    print("Analyzing best prediction horizons...")
    horizon_analysis = analyze_best_prediction_horizons(predictions)
    
    # Create risk-adjusted recommendations
    print("Creating risk-adjusted recommendations...")
    recommendations = create_risk_adjusted_recommendations(risk_metrics)
    
    # Generate visualizations
    print("Generating visualizations...")
    plot_risk_return_profile(risk_metrics)
    plot_return_comparison(predictions)
    
    # Generate HTML report
    print("Generating HTML report...")
    generate_html_report(predictions, risk_metrics, categorized_returns, 
                        recommendations, horizon_analysis)
    
    print("Analysis complete! Results saved to results/analysis/")

if __name__ == "__main__":
    main() 