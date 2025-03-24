#!/usr/bin/env python
# coding: utf-8

"""
Data Processing Script for Investment Recommendation System
This script creates enhanced_data.csv from raw data by adding technical indicators.
"""

import pandas as pd
import numpy as np
import os
import ta
from datetime import datetime, timedelta

print("=== Investment Recommendation System - Data Processing ===")

# Create necessary directories
os.makedirs('data/processed', exist_ok=True)

# Check if enhanced_data.csv already exists
if os.path.exists('data/enhanced_data.csv'):
    print("Enhanced data file already exists.")
    print("If you want to recreate it, please delete data/enhanced_data.csv first.")
    exit(0)

# Load raw data
print("Loading raw data...")
data_file = 'data/DUMMY.csv'
if not os.path.exists(data_file):
    print(f"Error: {data_file} not found.")
    exit(1)

# Load the data
data = pd.read_csv(data_file)
print(f"Loaded {len(data)} rows from {data_file}")

# Convert Date column to datetime if needed
if 'Date' in data.columns and not pd.api.types.is_datetime64_any_dtype(data['Date']):
    data['Date'] = pd.to_datetime(data['Date'])

# Make sure we have the required columns
required_cols = ['Date', 'Price', 'ticker']
missing_cols = [col for col in required_cols if col not in data.columns]
if missing_cols:
    print(f"Error: Missing required columns: {missing_cols}")
    exit(1)

# Sort by date
data = data.sort_values('Date')

# Add technical indicators
print("Adding technical indicators...")

# Price-based indicators
data['Price_1d_change'] = data['Price'].pct_change() * 100
data['Price_5d_change'] = data['Price'].pct_change(periods=5) * 100
data['Price_20d_change'] = data['Price'].pct_change(periods=20) * 100

# Moving averages
data['SMA_5'] = ta.trend.sma_indicator(data['Price'], window=5)
data['SMA_20'] = ta.trend.sma_indicator(data['Price'], window=20)
data['SMA_50'] = ta.trend.sma_indicator(data['Price'], window=50)
data['EMA_5'] = ta.trend.ema_indicator(data['Price'], window=5)
data['EMA_20'] = ta.trend.ema_indicator(data['Price'], window=20)

# Volatility indicators
data['Volatility_5d'] = data['Price_1d_change'].rolling(window=5).std()
data['Volatility_20d'] = data['Price_1d_change'].rolling(window=20).std()
data['ATR'] = ta.volatility.average_true_range(data['Price'], data['Price'], data['Price'], window=14)

# Bollinger Bands
bollinger = ta.volatility.BollingerBands(data['Price'], window=20, window_dev=2)
data['BB_High'] = bollinger.bollinger_hband()
data['BB_Low'] = bollinger.bollinger_lband()
data['BB_Width'] = (data['BB_High'] - data['BB_Low']) / data['SMA_20'] * 100

# Momentum indicators
data['RSI_14'] = ta.momentum.rsi(data['Price'], window=14)
data['MACD'] = ta.trend.macd(data['Price'])
data['MACD_Signal'] = ta.trend.macd_signal(data['Price'])
data['MACD_Hist'] = data['MACD'] - data['MACD_Signal']

# Volume-based indicators if Volume is available
if 'Volume' in data.columns:
    data['Volume_1d_change'] = data['Volume'].pct_change() * 100
    data['Volume_SMA_5'] = ta.trend.sma_indicator(data['Volume'], window=5)
    data['Volume_SMA_20'] = ta.trend.sma_indicator(data['Volume'], window=20)
    data['OBV'] = ta.volume.on_balance_volume(data['Price'], data['Volume'])

# Fill NAs with appropriate values
numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
data[numeric_cols] = data[numeric_cols].fillna(method='ffill').fillna(method='bfill').fillna(0)

# Save the enhanced data
print("Saving enhanced data...")
data.to_csv('data/enhanced_data.csv', index=False)
print(f"Enhanced data saved with {len(data)} rows and {len(data.columns)} columns.")
print("You can now run the Investment Recommendation System with TimeframeInvestmentAnalyzer.") 