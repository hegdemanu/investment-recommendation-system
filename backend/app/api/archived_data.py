from fastapi import APIRouter, HTTPException
import os
import json
from pathlib import Path

router = APIRouter()

# Path to archive folder
ARCHIVE_DIR = Path("../archive")

@router.get("/stocks")
async def get_stocks():
    """Get stock data from archive"""
    try:
        stocks_file = ARCHIVE_DIR / "stocks_data.json"
        if not stocks_file.exists():
            return generate_sample_stocks()
        
        with open(stocks_file, "r") as f:
            stocks = json.load(f)
        return stocks[:12]  # Return 12 stocks
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching stocks: {str(e)}")

@router.get("/mutual-funds")
async def get_mutual_funds():
    """Get mutual fund data from archive"""
    try:
        mf_file = ARCHIVE_DIR / "mutual_funds_data.json"
        if not mf_file.exists():
            return generate_sample_mutual_funds()
        
        with open(mf_file, "r") as f:
            funds = json.load(f)
        return funds[:6]  # Return 6 mutual funds
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching mutual funds: {str(e)}")

def generate_sample_stocks():
    """Generate sample stock data if archive not found"""
    return [
        {"symbol": "AAPL", "name": "Apple Inc.", "price": 182.34, "change": 1.25, "exchange": "NASDAQ"},
        {"symbol": "MSFT", "name": "Microsoft Corporation", "price": 315.75, "change": 0.82, "exchange": "NASDAQ"},
        {"symbol": "GOOGL", "name": "Alphabet Inc.", "price": 141.23, "change": -0.45, "exchange": "NASDAQ"},
        {"symbol": "AMZN", "name": "Amazon.com Inc.", "price": 127.48, "change": 0.93, "exchange": "NASDAQ"},
        {"symbol": "TSLA", "name": "Tesla, Inc.", "price": 172.63, "change": -1.75, "exchange": "NASDAQ"},
        {"symbol": "META", "name": "Meta Platforms Inc.", "price": 329.12, "change": 2.08, "exchange": "NASDAQ"},
        {"symbol": "NFLX", "name": "Netflix Inc.", "price": 589.74, "change": 3.45, "exchange": "NASDAQ"},
        {"symbol": "JPM", "name": "JPMorgan Chase & Co.", "price": 158.92, "change": 0.32, "exchange": "NYSE"},
        {"symbol": "V", "name": "Visa Inc.", "price": 267.41, "change": 1.05, "exchange": "NYSE"},
        {"symbol": "PG", "name": "Procter & Gamble Co.", "price": 163.82, "change": -0.23, "exchange": "NYSE"},
        {"symbol": "JNJ", "name": "Johnson & Johnson", "price": 152.49, "change": -0.56, "exchange": "NYSE"},
        {"symbol": "WMT", "name": "Walmart Inc.", "price": 59.86, "change": 0.43, "exchange": "NYSE"}
    ]

def generate_sample_mutual_funds():
    """Generate sample mutual fund data if archive not found"""
    return [
        {"symbol": "VFINX", "name": "Vanguard 500 Index Fund", "nav": 398.42, "change": 0.85, "expense_ratio": 0.14},
        {"symbol": "FXAIX", "name": "Fidelity 500 Index Fund", "nav": 175.63, "change": 0.82, "expense_ratio": 0.015},
        {"symbol": "SWPPX", "name": "Schwab S&P 500 Index Fund", "nav": 68.29, "change": 0.81, "expense_ratio": 0.02},
        {"symbol": "VTSAX", "name": "Vanguard Total Stock Market Index", "nav": 112.78, "change": 0.74, "expense_ratio": 0.04},
        {"symbol": "VBTLX", "name": "Vanguard Total Bond Market Index", "nav": 10.41, "change": -0.12, "expense_ratio": 0.05},
        {"symbol": "VTIAX", "name": "Vanguard Total International Stock Index", "nav": 31.87, "change": 0.26, "expense_ratio": 0.11}
    ]
