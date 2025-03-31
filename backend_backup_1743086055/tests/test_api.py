import pytest
from fastapi.testclient import TestClient
from app import create_app

@pytest.fixture
def client():
    app = create_app()
    return TestClient(app)

def test_health_check(client):
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_predict_endpoint(client):
    response = client.post(
        "/api/v1/predict",
        json={
            "symbol": "AAPL",
            "days": 30,
            "include_sentiment": True
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert "predictions" in data
    assert "sentiment" in data

def test_sentiment_analysis(client):
    response = client.post(
        "/api/v1/sentiment/analyze",
        json={
            "text": "Apple's new iPhone sales exceed expectations"
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert "sentiment" in data
    assert "confidence" in data

def test_model_training(client):
    response = client.post(
        "/api/v1/models/train",
        json={
            "symbol": "AAPL",
            "start_date": "2023-01-01",
            "end_date": "2024-01-01"
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "models" in data 