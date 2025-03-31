#!/usr/bin/env python3
"""
Flask Application for Investment Recommendation System
Provides REST API endpoints for predictions and sentiment analysis
"""

from flask import Flask, jsonify
from datetime import datetime

# Create app
app = Flask(__name__)

@app.route("/")
def root():
    """Root endpoint to verify the API is working"""
    return jsonify({
        "message": "Investment Recommendation System API",
        "status": "running",
        "timestamp": datetime.now().isoformat()
    })

@app.route("/health")
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True) 