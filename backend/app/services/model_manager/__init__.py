"""
Model manager for AI/ML model loading, caching, and prediction.

This module manages loading, inference, and caching of machine learning models
for the investment recommendation system.
"""

from .manager import ModelManager, get_model_manager

__all__ = ["ModelManager", "get_model_manager"] 