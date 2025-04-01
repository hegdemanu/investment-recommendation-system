#!/usr/bin/env python
# coding: utf-8

"""
Minimal fix for the TimeframeInvestmentAnalyzer ModelBuilder issue.
This file provides a patch to make models work with the existing code.
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

class ModelBuilder:
    """Class for building and managing ML models for price prediction"""
    
    def __init__(self, models_dir='models'):
        self.models_dir = models_dir
        
        # Ensure models directory exists
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
            
    def prepare_lstm_data(self, data, ticker, seq_length=30, target_col='Price', feature_cols=None, test_size=0.2, val_size=0.2):
        """Prepare data for LSTM model training"""
        if data is None or data.empty:
            print("Error: Empty data provided")
            return None, None, None
        
        # Ensure target column exists
        if target_col not in data.columns:
            print(f"Error: Target column '{target_col}' not found in data")
            return None, None, None
        
        # Use provided feature columns or all numeric columns
        if feature_cols is None:
            # Exclude non-numeric columns
            feature_cols = [col for col in data.columns if col not in ['Date', 'ticker']]
            feature_cols = [col for col in feature_cols if pd.api.types.is_numeric_dtype(data[col])]
        
        # Ensure we have enough features
        if len(feature_cols) < 1:
            print("Error: No valid features found in data")
            return None, None, None
        
        # Create a copy to avoid modifying the original data
        df = data[feature_cols].copy()
        
        # Ensure target column is first (for easier extraction later)
        if target_col in feature_cols and feature_cols[0] != target_col:
            feature_cols.remove(target_col)
            feature_cols.insert(0, target_col)
            df = data[feature_cols].copy()
        
        # Scale the data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(df.values)
        
        # Create sequences
        X, y = [], []
        for i in range(len(scaled_data) - seq_length):
            X.append(scaled_data[i:i+seq_length])
            y.append(scaled_data[i+seq_length, 0])  # Target is first column (Price)
        
        X, y = np.array(X), np.array(y)
        
        # Check if we have enough data
        if len(X) < 3:
            print(f"Error: Not enough data points to create sequences for {ticker}")
            return None, None, None
        
        # Split data
        X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)
        X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=val_size, shuffle=False)
        
        print(f"Data shapes - X_train: {X_train.shape}, X_val: {X_val.shape}, X_test: {X_test.shape}")
        
        return (X_train, y_train, X_val, y_val, X_test, y_test), scaler, feature_cols


# To fix the issue in your Jupyter notebook, add the following to a cell before using TimeframeInvestmentAnalyzer:
"""
# Fix for the ModelBuilder issue - Copy and paste this in a cell before using TimeframeInvestmentAnalyzer

# Option 1: If you want to modify the TimeframeInvestmentAnalyzer class to use this ModelBuilder
def fix_analyzer_class():
    # Find where the train_timeframe_model method attempts to create a ModelBuilder
    # Replace the lines:
    #    model_builder = ModelBuilder(models_dir=f"{self.models_dir}/{timeframe}_term")
    # With:
    #    from analyzer_fix import ModelBuilder
    #    model_builder = ModelBuilder(models_dir=f"{self.models_dir}/{timeframe}_term")
    print("Fix method 1: Add 'from analyzer_fix import ModelBuilder' to the TimeframeInvestmentAnalyzer class")

# Option 2: Simple fix - run this code in your notebook before using TimeframeInvestmentAnalyzer
def add_modelbuilder_to_module():
    # This adds the ModelBuilder class to the module namespace
    import sys
    from analyzer_fix import ModelBuilder
    # Add ModelBuilder to the main module
    sys.modules['__main__'].ModelBuilder = ModelBuilder
    print("ModelBuilder class added to the main module")

# You can call this function to apply the fix
add_modelbuilder_to_module()
""" 