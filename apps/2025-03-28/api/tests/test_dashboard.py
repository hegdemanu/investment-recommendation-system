import pytest
import pandas as pd
import numpy as np
from app.dashboard.dashboard_generator import DashboardGenerator
from app.dashboard.visualization import create_price_chart, create_volume_chart, create_technical_indicators

@pytest.fixture
def dashboard_generator():
    return DashboardGenerator()

@pytest.fixture
def sample_data():
    dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='D')
    data = pd.DataFrame({
        'date': dates,
        'open': np.random.normal(100, 10, len(dates)),
        'high': np.random.normal(105, 10, len(dates)),
        'low': np.random.normal(95, 10, len(dates)),
        'close': np.random.normal(100, 10, len(dates)),
        'volume': np.random.randint(1000000, 10000000, len(dates)),
        'rsi': np.random.uniform(0, 100, len(dates)),
        'macd': np.random.normal(0, 1, len(dates)),
        'bollinger_upper': np.random.normal(110, 5, len(dates)),
        'bollinger_lower': np.random.normal(90, 5, len(dates))
    })
    return data

@pytest.fixture
def sample_predictions():
    dates = pd.date_range(start='2024-01-02', end='2024-02-01', freq='D')
    predictions = pd.DataFrame({
        'date': dates,
        'predicted_price': np.random.normal(100, 10, len(dates)),
        'confidence': np.random.uniform(0.5, 0.9, len(dates))
    })
    return predictions

def test_dashboard_generation(dashboard_generator, sample_data, sample_predictions):
    dashboard = dashboard_generator.generate_dashboard(
        symbol="AAPL",
        historical_data=sample_data,
        predictions=sample_predictions
    )
    
    assert isinstance(dashboard, str)
    assert "AAPL" in dashboard
    assert "Stock Price Prediction" in dashboard
    assert "Technical Indicators" in dashboard

def test_price_chart_creation(sample_data):
    chart = create_price_chart(sample_data)
    assert isinstance(chart, dict)
    assert "data" in chart
    assert "layout" in chart
    assert "title" in chart["layout"]
    assert "xaxis" in chart["layout"]
    assert "yaxis" in chart["layout"]

def test_volume_chart_creation(sample_data):
    chart = create_volume_chart(sample_data)
    assert isinstance(chart, dict)
    assert "data" in chart
    assert "layout" in chart
    assert "title" in chart["layout"]
    assert "xaxis" in chart["layout"]
    assert "yaxis" in chart["layout"]

def test_technical_indicators_creation(sample_data):
    indicators = create_technical_indicators(sample_data)
    assert isinstance(indicators, dict)
    assert "rsi" in indicators
    assert "macd" in indicators
    assert "bollinger" in indicators
    
    for indicator in indicators.values():
        assert isinstance(indicator, dict)
        assert "data" in indicator
        assert "layout" in indicator

def test_dashboard_template_rendering(dashboard_generator, sample_data, sample_predictions):
    template = dashboard_generator.render_template(
        symbol="AAPL",
        historical_data=sample_data,
        predictions=sample_predictions
    )
    
    assert isinstance(template, str)
    assert "AAPL" in template
    assert "Stock Price Prediction" in template
    assert "Technical Indicators" in template
    assert "RSI" in template
    assert "MACD" in template
    assert "Bollinger Bands" in template

def test_dashboard_data_processing(dashboard_generator, sample_data, sample_predictions):
    processed_data = dashboard_generator.process_data(
        historical_data=sample_data,
        predictions=sample_predictions
    )
    
    assert isinstance(processed_data, dict)
    assert "historical" in processed_data
    assert "predictions" in processed_data
    assert "indicators" in processed_data
    assert isinstance(processed_data["historical"], pd.DataFrame)
    assert isinstance(processed_data["predictions"], pd.DataFrame)
    assert isinstance(processed_data["indicators"], dict) 