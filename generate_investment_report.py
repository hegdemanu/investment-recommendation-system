#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from datetime import datetime
import logging
from collections import defaultdict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger('InvestmentReport')

# Set up directories
STOCK_MODELS_DIR = './models/stocks'
MF_MODELS_DIR = './models/mutual_funds'
RESULTS_DIR = './results'
REPORT_DIR = os.path.join(RESULTS_DIR, 'reports')
TRAINING_SUMMARY_FILE = os.path.join(RESULTS_DIR, 'training', 'training_summary.json')
MF_TRAINING_SUMMARY_FILE = os.path.join(RESULTS_DIR, 'training', 'mutual_funds_training_summary.json')

# Ensure directories exist
os.makedirs(REPORT_DIR, exist_ok=True)

def load_training_summaries():
    """
    Load the training summaries for stocks and mutual funds
    """
    stock_summary = {}
    mf_summary = {}
    
    # Load stock summary if exists
    if os.path.exists(TRAINING_SUMMARY_FILE):
        with open(TRAINING_SUMMARY_FILE, 'r') as f:
            stock_summary = json.load(f)
        logger.info(f"Loaded stock training summary with {len(stock_summary.get('stocks', {}).get('details', {}))} stocks")
    
    # Load mutual fund summary if exists
    if os.path.exists(MF_TRAINING_SUMMARY_FILE):
        with open(MF_TRAINING_SUMMARY_FILE, 'r') as f:
            mf_summary = json.load(f)
        logger.info(f"Loaded mutual fund training summary with {mf_summary.get('total', 0)} mutual funds")
        
    return stock_summary, mf_summary

def extract_metrics(stock_summary, mf_summary):
    """
    Extract metrics from the training summaries
    """
    metrics = {
        'stocks': {
            'total': stock_summary.get('stocks', {}).get('total', 0),
            'success': stock_summary.get('stocks', {}).get('success', 0),
            'failed': stock_summary.get('stocks', {}).get('failed', 0),
            'avg_training_time': 0,
            'rmse_values': {},
        },
        'mutual_funds': {
            'total': mf_summary.get('total', 0),
            'success': mf_summary.get('success', 0),
            'failed': mf_summary.get('failed', 0),
            'avg_training_time': 0,
            'rmse_values': {},
        }
    }
    
    # Process stock metrics
    stock_training_times = []
    stock_details = stock_summary.get('stocks', {}).get('details', {})
    for ticker, details in stock_details.items():
        if details.get('training_time'):
            stock_training_times.append(details['training_time'])
        
        # Load model metadata to get RMSE
        model_path = details.get('model_path', '')
        if model_path and model_path.endswith('_lstm.h5'):
            metadata_path = model_path.replace('_lstm.h5', '_metadata.json')
            if os.path.exists(metadata_path):
                try:
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                        rmse = metadata.get('rmse', {})
                        if rmse:
                            metrics['stocks']['rmse_values'][ticker] = rmse
                except Exception as e:
                    logger.error(f"Error loading metadata for {ticker}: {e}")
    
    # Process mutual fund metrics
    mf_training_times = []
    mf_details = mf_summary.get('details', {})
    for fund, details in mf_details.items():
        if details.get('training_time'):
            mf_training_times.append(details['training_time'])
        
        # Load model metadata to get RMSE
        model_path = details.get('model_path', '')
        if model_path and model_path.endswith('_lstm.h5'):
            metadata_path = model_path.replace('_lstm.h5', '_metadata.json')
            if os.path.exists(metadata_path):
                try:
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                        rmse = metadata.get('rmse', {})
                        if rmse:
                            metrics['mutual_funds']['rmse_values'][fund] = rmse
                except Exception as e:
                    logger.error(f"Error loading metadata for {fund}: {e}")
    
    # Calculate averages
    if stock_training_times:
        metrics['stocks']['avg_training_time'] = sum(stock_training_times) / len(stock_training_times)
    
    if mf_training_times:
        metrics['mutual_funds']['avg_training_time'] = sum(mf_training_times) / len(mf_training_times)
    
    return metrics

def generate_training_performance_chart(stock_summary, mf_summary):
    """
    Generate a chart showing training performance for stocks and mutual funds
    """
    plt.figure(figsize=(10, 6))
    
    # Extract data
    categories = ['Stocks', 'Mutual Funds']
    success_data = [
        stock_summary.get('stocks', {}).get('success', 0), 
        mf_summary.get('success', 0)
    ]
    failed_data = [
        stock_summary.get('stocks', {}).get('failed', 0), 
        mf_summary.get('failed', 0)
    ]
    
    # Create the bar chart
    x = np.arange(len(categories))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 7))
    rects1 = ax.bar(x - width/2, success_data, width, label='Success', color='#2ecc71')
    rects2 = ax.bar(x + width/2, failed_data, width, label='Failed', color='#e74c3c')
    
    # Add labels and titles
    ax.set_title('Model Training Performance', fontsize=16, fontweight='bold')
    ax.set_xlabel('Investment Type', fontsize=14)
    ax.set_ylabel('Number of Models', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=12)
    ax.legend(fontsize=12)
    
    # Add value labels on bars
    def add_labels(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom',
                        fontsize=12, fontweight='bold')
    
    add_labels(rects1)
    add_labels(rects2)
    
    plt.tight_layout()
    chart_path = os.path.join(REPORT_DIR, 'training_performance.png')
    plt.savefig(chart_path)
    logger.info(f"Training performance chart saved to {chart_path}")
    
    return chart_path

def generate_prediction_accuracy_chart(metrics):
    """
    Generate a chart showing prediction accuracy for stocks and mutual funds
    """
    plt.figure(figsize=(12, 7))
    
    # Extract RMSE data
    stock_rmse = metrics['stocks']['rmse_values']
    mf_rmse = metrics['mutual_funds']['rmse_values']
    
    # Prepare data for the chart
    stock_data = []
    mf_data = []
    
    for ticker, rmse_dict in stock_rmse.items():
        best_horizon = min(rmse_dict.items(), key=lambda x: float(x[1]))[0]
        best_rmse = float(rmse_dict[best_horizon])
        stock_data.append((ticker, best_rmse))
    
    for fund, rmse_dict in mf_rmse.items():
        best_horizon = min(rmse_dict.items(), key=lambda x: float(x[1]))[0]
        best_rmse = float(rmse_dict[best_horizon])
        mf_data.append((fund, best_rmse))
    
    # Sort data by RMSE (lower is better)
    stock_data.sort(key=lambda x: x[1])
    mf_data.sort(key=lambda x: x[1])
    
    # Create the bar chart
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Set up colors
    stock_color = '#3498db'
    mf_color = '#9b59b6'
    
    # Plot stocks
    stock_tickers = [item[0] for item in stock_data]
    stock_rmse_values = [item[1] for item in stock_data]
    stock_bars = ax.bar(range(len(stock_data)), stock_rmse_values, color=stock_color, alpha=0.7, label='Stocks')
    
    # Plot mutual funds (offset by the number of stocks)
    mf_tickers = [item[0] for item in mf_data]
    mf_rmse_values = [item[1] for item in mf_data]
    offset = len(stock_data) + 1  # Add a gap between stocks and mutual funds
    mf_bars = ax.bar(range(offset, offset + len(mf_data)), mf_rmse_values, color=mf_color, alpha=0.7, label='Mutual Funds')
    
    # Add labels and titles
    ax.set_title('Prediction Accuracy (Lower RMSE is Better)', fontsize=16, fontweight='bold')
    ax.set_xlabel('Investment Instrument', fontsize=14)
    ax.set_ylabel('Root Mean Square Error (RMSE)', fontsize=14)
    
    # Set tick positions
    all_positions = list(range(len(stock_data))) + list(range(offset, offset + len(mf_data)))
    all_labels = stock_tickers + mf_tickers
    
    ax.set_xticks(all_positions)
    ax.set_xticklabels(all_labels, rotation=45, ha='right', fontsize=10)
    
    # Add legend
    ax.legend(fontsize=12)
    
    plt.tight_layout()
    chart_path = os.path.join(REPORT_DIR, 'prediction_accuracy.png')
    plt.savefig(chart_path)
    logger.info(f"Prediction accuracy chart saved to {chart_path}")
    
    return chart_path

def generate_training_size_vs_accuracy_chart(stock_summary, mf_summary, metrics):
    """
    Generate a scatter plot showing relationship between training size and model accuracy
    """
    plt.figure(figsize=(12, 8))
    
    # Extract data
    stock_data = []
    mf_data = []
    
    # Process stock data
    stock_details = stock_summary.get('stocks', {}).get('details', {})
    for ticker, details in stock_details.items():
        data_points = details.get('data_points', 0)
        if ticker in metrics['stocks']['rmse_values']:
            best_horizon = min(metrics['stocks']['rmse_values'][ticker].items(), key=lambda x: float(x[1]))[0]
            best_rmse = float(metrics['stocks']['rmse_values'][ticker][best_horizon])
            stock_data.append((ticker, data_points, best_rmse))
    
    # Process mutual fund data
    mf_details = mf_summary.get('details', {})
    for fund, details in mf_details.items():
        data_points = details.get('data_points', 0)
        if fund in metrics['mutual_funds']['rmse_values']:
            best_horizon = min(metrics['mutual_funds']['rmse_values'][fund].items(), key=lambda x: float(x[1]))[0]
            best_rmse = float(metrics['mutual_funds']['rmse_values'][fund][best_horizon])
            mf_data.append((fund, data_points, best_rmse))
    
    # Create the scatter plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot stocks
    stock_x = [item[1] for item in stock_data]
    stock_y = [item[2] for item in stock_data]
    stock_labels = [item[0] for item in stock_data]
    
    ax.scatter(stock_x, stock_y, color='#3498db', alpha=0.7, s=100, label='Stocks')
    
    # Plot mutual funds
    mf_x = [item[1] for item in mf_data]
    mf_y = [item[2] for item in mf_data]
    mf_labels = [item[0] for item in mf_data]
    
    ax.scatter(mf_x, mf_y, color='#9b59b6', alpha=0.7, s=100, label='Mutual Funds')
    
    # Add labels for points
    for i, label in enumerate(stock_labels):
        ax.annotate(label, (stock_x[i], stock_y[i]), fontsize=9)
    
    for i, label in enumerate(mf_labels):
        ax.annotate(label, (mf_x[i], mf_y[i]), fontsize=9)
    
    # Add labels and titles
    ax.set_title('Training Size vs. Model Accuracy', fontsize=16, fontweight='bold')
    ax.set_xlabel('Number of Data Points', fontsize=14)
    ax.set_ylabel('Root Mean Square Error (RMSE)', fontsize=14)
    
    # Add trend lines
    if stock_x and stock_y:
        stock_z = np.polyfit(stock_x, stock_y, 1)
        stock_p = np.poly1d(stock_z)
        ax.plot(stock_x, stock_p(stock_x), color='#3498db', linestyle='--')
    
    if mf_x and mf_y:
        mf_z = np.polyfit(mf_x, mf_y, 1)
        mf_p = np.poly1d(mf_z)
        ax.plot(mf_x, mf_p(mf_x), color='#9b59b6', linestyle='--')
    
    # Add legend
    ax.legend(fontsize=12)
    
    plt.tight_layout()
    chart_path = os.path.join(REPORT_DIR, 'training_size_vs_accuracy.png')
    plt.savefig(chart_path)
    logger.info(f"Training size vs accuracy chart saved to {chart_path}")
    
    return chart_path

def format_training_time(details):
    """Helper function to format training time"""
    training_time = details.get('training_time')
    if training_time:
        return f"{training_time:.2f}"
    return "N/A"

def get_status_class(details):
    """Helper function to get the status CSS class"""
    if details.get('status') == "SUCCESS":
        return "status-success"
    return "status-failed"

def generate_stock_table_rows(stock_summary):
    """Generate HTML table rows for stocks"""
    rows = []
    for ticker, details in stock_summary.get('stocks', {}).get('details', {}).items():
        status_class = get_status_class(details)
        status = details.get('status', 'N/A')
        data_points = details.get('data_points', 'N/A')
        training_time = format_training_time(details)
        
        row = f"<tr><td>{ticker}</td><td>{data_points}</td><td class='{status_class}'>{status}</td><td>{training_time}</td></tr>"
        rows.append(row)
    
    return ''.join(rows)

def generate_mf_table_rows(mf_summary):
    """Generate HTML table rows for mutual funds"""
    rows = []
    for fund, details in mf_summary.get('details', {}).items():
        status_class = get_status_class(details)
        status = details.get('status', 'N/A')
        data_points = details.get('data_points', 'N/A')
        training_time = format_training_time(details)
        
        row = f"<tr><td>{fund}</td><td>{data_points}</td><td class='{status_class}'>{status}</td><td>{training_time}</td></tr>"
        rows.append(row)
    
    return ''.join(rows)

def generate_html_report(stock_summary, mf_summary, metrics, charts):
    """
    Generate an HTML report with the investment recommendations
    """
    # Get top performing investments based on RMSE
    top_stocks = []
    for ticker, rmse_dict in metrics['stocks']['rmse_values'].items():
        best_horizon = min(rmse_dict.items(), key=lambda x: float(x[1]))[0]
        best_rmse = float(rmse_dict[best_horizon])
        top_stocks.append((ticker, best_rmse, best_horizon))
    
    top_mutual_funds = []
    for fund, rmse_dict in metrics['mutual_funds']['rmse_values'].items():
        best_horizon = min(rmse_dict.items(), key=lambda x: float(x[1]))[0]
        best_rmse = float(rmse_dict[best_horizon])
        top_mutual_funds.append((fund, best_rmse, best_horizon))
    
    # Sort by RMSE (lower is better)
    top_stocks.sort(key=lambda x: x[1])
    top_mutual_funds.sort(key=lambda x: x[1])
    
    # Take top 5 or all if less than 5
    top_stocks = top_stocks[:5]
    top_mutual_funds = top_mutual_funds[:5]
    
    # Generate stock and mutual fund table rows
    stock_table_rows = generate_stock_table_rows(stock_summary)
    mf_table_rows = generate_mf_table_rows(mf_summary)
    
    # Generate stock and mutual fund ranking rows
    stock_ranking_rows = ''.join([f"<tr><td>{i+1}</td><td>{ticker}</td><td>{rmse:.4f}</td><td>{horizon} days</td></tr>" 
                               for i, (ticker, rmse, horizon) in enumerate(top_stocks)])
    
    mf_ranking_rows = ''.join([f"<tr><td>{i+1}</td><td>{fund}</td><td>{rmse:.4f}</td><td>{horizon} days</td></tr>" 
                            for i, (fund, rmse, horizon) in enumerate(top_mutual_funds)])
    
    # Generate HTML
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Investment Recommendation Report</title>
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                line-height: 1.6;
                color: #333;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
            }}
            header {{
                background-color: #2c3e50;
                color: white;
                padding: 20px;
                text-align: center;
                border-radius: 8px;
                margin-bottom: 30px;
            }}
            h1, h2, h3 {{
                color: #2c3e50;
            }}
            h1 {{
                margin-bottom: 10px;
            }}
            .date {{
                color: #7f8c8d;
                font-style: italic;
                margin-bottom: 20px;
            }}
            .section {{
                margin-bottom: 40px;
                background-color: #f9f9f9;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            }}
            .chart-container {{
                text-align: center;
                margin: 30px 0;
            }}
            .chart-container img {{
                max-width: 100%;
                border-radius: 8px;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
            }}
            th, td {{
                padding: 12px 15px;
                border-bottom: 1px solid #ddd;
                text-align: left;
            }}
            th {{
                background-color: #2c3e50;
                color: white;
            }}
            tr:nth-child(even) {{
                background-color: #f2f2f2;
            }}
            tr:hover {{
                background-color: #e9e9e9;
            }}
            .metrics-container {{
                display: flex;
                justify-content: space-between;
                flex-wrap: wrap;
                margin: 20px 0;
            }}
            .metric-box {{
                background-color: white;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
                padding: 15px;
                width: 48%;
                margin-bottom: 20px;
            }}
            .metric-value {{
                font-size: 24px;
                font-weight: bold;
                color: #2c3e50;
            }}
            .status-success {{
                color: #27ae60;
            }}
            .status-failed {{
                color: #c0392b;
            }}
            .recommendation-container {{
                display: flex;
                flex-wrap: wrap;
                justify-content: space-between;
            }}
            .recommendation-box {{
                background-color: white;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
                padding: 20px;
                width: 48%;
                margin-bottom: 20px;
            }}
            .footer {{
                text-align: center;
                margin-top: 40px;
                color: #7f8c8d;
                font-size: 14px;
            }}
        </style>
    </head>
    <body>
        <header>
            <h1>Investment Recommendation System</h1>
            <p>Comprehensive Analysis and Recommendations</p>
        </header>
        
        <div class="date">
            Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        </div>
        
        <div class="section">
            <h2>Training Performance Overview</h2>
            <div class="metrics-container">
                <div class="metric-box">
                    <h3>Stocks</h3>
                    <p>Total: <span class="metric-value">{metrics['stocks']['total']}</span></p>
                    <p>Successfully Trained: <span class="metric-value status-success">{metrics['stocks']['success']}</span></p>
                    <p>Failed: <span class="metric-value status-failed">{metrics['stocks']['failed']}</span></p>
                    <p>Avg. Training Time: <span class="metric-value">{metrics['stocks']['avg_training_time']:.2f}s</span></p>
                </div>
                <div class="metric-box">
                    <h3>Mutual Funds</h3>
                    <p>Total: <span class="metric-value">{metrics['mutual_funds']['total']}</span></p>
                    <p>Successfully Trained: <span class="metric-value status-success">{metrics['mutual_funds']['success']}</span></p>
                    <p>Failed: <span class="metric-value status-failed">{metrics['mutual_funds']['failed']}</span></p>
                    <p>Avg. Training Time: <span class="metric-value">{metrics['mutual_funds']['avg_training_time']:.2f}s</span></p>
                </div>
            </div>
            
            <div class="chart-container">
                <img src="{os.path.basename(charts['training_performance'])}" alt="Training Performance">
                <p>Figure 1: Training success and failure comparison between stocks and mutual funds</p>
            </div>
        </div>
        
        <div class="section">
            <h2>Model Accuracy Analysis</h2>
            
            <div class="chart-container">
                <img src="{os.path.basename(charts['prediction_accuracy'])}" alt="Prediction Accuracy">
                <p>Figure 2: Prediction accuracy comparison (RMSE values) - lower is better</p>
            </div>
            
            <div class="chart-container">
                <img src="{os.path.basename(charts['training_size_vs_accuracy'])}" alt="Training Size vs Accuracy">
                <p>Figure 3: Relationship between training data size and model accuracy</p>
            </div>
        </div>
        
        <div class="section">
            <h2>Top Investment Recommendations</h2>
            
            <div class="recommendation-container">
                <div class="recommendation-box">
                    <h3>Top Performing Stocks</h3>
                    <table>
                        <tr>
                            <th>Rank</th>
                            <th>Ticker</th>
                            <th>RMSE</th>
                            <th>Best Horizon</th>
                        </tr>
                        {stock_ranking_rows}
                    </table>
                </div>
                
                <div class="recommendation-box">
                    <h3>Top Performing Mutual Funds</h3>
                    <table>
                        <tr>
                            <th>Rank</th>
                            <th>Fund</th>
                            <th>RMSE</th>
                            <th>Best Horizon</th>
                        </tr>
                        {mf_ranking_rows}
                    </table>
                </div>
            </div>
        </div>
        
        <div class="section">
            <h2>Training Details</h2>
            
            <h3>Stock Training Details</h3>
            <table>
                <tr>
                    <th>Ticker</th>
                    <th>Data Points</th>
                    <th>Status</th>
                    <th>Training Time (s)</th>
                </tr>
                {stock_table_rows}
            </table>
            
            <h3>Mutual Fund Training Details</h3>
            <table>
                <tr>
                    <th>Fund Name</th>
                    <th>Data Points</th>
                    <th>Status</th>
                    <th>Training Time (s)</th>
                </tr>
                {mf_table_rows}
            </table>
        </div>
        
        <div class="footer">
            <p>&copy; 2025 Investment Recommendation System | Generated by AI Reporting Engine</p>
        </div>
    </body>
    </html>
    """
    
    # Save the HTML report
    report_path = os.path.join(REPORT_DIR, 'investment_report.html')
    with open(report_path, 'w') as f:
        f.write(html_content)
    
    return report_path

def main():
    """
    Main function to generate the investment report
    """
    # Load training summaries
    stock_summary, mf_summary = load_training_summaries()
    
    # Extract metrics
    metrics = extract_metrics(stock_summary, mf_summary)
    
    # Generate charts
    training_performance_chart = generate_training_performance_chart(stock_summary, mf_summary)
    prediction_accuracy_chart = generate_prediction_accuracy_chart(metrics)
    training_size_vs_accuracy_chart = generate_training_size_vs_accuracy_chart(stock_summary, mf_summary, metrics)
    
    # Collect charts
    charts = {
        'training_performance': training_performance_chart,
        'prediction_accuracy': prediction_accuracy_chart,
        'training_size_vs_accuracy': training_size_vs_accuracy_chart,
    }
    
    # Generate HTML report
    report_path = generate_html_report(stock_summary, mf_summary, metrics, charts)
    
    print(f"\nInvestment report generated successfully at: {report_path}")
    print(f"Open this file in a web browser to view the complete report.\n")

if __name__ == '__main__':
    main() 