#!/usr/bin/env python
# coding: utf-8

"""
ModelBuilder Fix for Investment Recommendation System
This script patches the ModelBuilder class to work with TimeframeInvestmentAnalyzer.
"""

import sys
import os
from models.analyzer_fix import ModelBuilder

print("=== Investment Recommendation System - ModelBuilder Fix ===")
print("Patching ModelBuilder class to make it available from analyzer_fix...")

# Add ModelBuilder to the main module
sys.modules['__main__'].ModelBuilder = ModelBuilder

# Also ensure from sklearn.model_selection import train_test_split is available
try:
    from sklearn.model_selection import train_test_split
    sys.modules['__main__'].train_test_split = train_test_split
    print("Successfully imported train_test_split")
except ImportError:
    print("Error: sklearn.model_selection.train_test_split not found")
    print("Please run 'pip install scikit-learn' to install it")
    exit(1)

print("ModelBuilder and train_test_split added to the main module")
print("Now you can run the TimeframeInvestmentAnalyzer without errors")
print("\nTo use this fix in your Jupyter notebook, add this cell at the beginning:")
print("```python")
print("# Fix ModelBuilder and train_test_split imports")
print("from analyzer_fix import ModelBuilder")
print("from sklearn.model_selection import train_test_split")
print("```") 