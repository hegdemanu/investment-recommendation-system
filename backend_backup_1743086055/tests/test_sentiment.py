import pytest
from app.models.sentiment_pipeline import SentimentAnalyzer

@pytest.fixture
def sentiment_analyzer():
    return SentimentAnalyzer()

@pytest.mark.asyncio
async def test_sentiment_analysis_basic(sentiment_analyzer):
    text = "Apple's new iPhone sales exceed expectations"
    result = await sentiment_analyzer.analyze_text(text)
    
    assert isinstance(result, dict)
    assert "sentiment" in result
    assert "confidence" in result
    assert "probabilities" in result
    assert result["sentiment"] in ["positive", "negative", "neutral"]
    assert 0 <= result["confidence"] <= 1
    assert all(0 <= prob <= 1 for prob in result["probabilities"].values())

@pytest.mark.asyncio
async def test_sentiment_analysis_batch(sentiment_analyzer):
    texts = [
        "Apple's new iPhone sales exceed expectations",
        "Market crash causes widespread panic",
        "Company reports steady growth"
    ]
    results = await sentiment_analyzer.analyze_batch(texts)
    
    assert isinstance(results, list)
    assert len(results) == len(texts)
    for result in results:
        assert isinstance(result, dict)
        assert "sentiment" in result
        assert "confidence" in result
        assert "probabilities" in result

@pytest.mark.asyncio
async def test_sentiment_analysis_empty(sentiment_analyzer):
    with pytest.raises(ValueError):
        await sentiment_analyzer.analyze_text("")

@pytest.mark.asyncio
async def test_sentiment_analysis_long_text(sentiment_analyzer):
    long_text = "Apple Inc. today announced financial results for its fiscal 2024 first quarter ended December 30, 2023. The Company posted quarterly revenue of $119.6 billion, up 2 percent year over year, and quarterly earnings per diluted share of $2.18, up 16 percent year over year. International sales accounted for 58 percent of the quarter's revenue."
    result = await sentiment_analyzer.analyze_text(long_text)
    
    assert isinstance(result, dict)
    assert "sentiment" in result
    assert "confidence" in result
    assert "probabilities" in result

@pytest.mark.asyncio
async def test_sentiment_analysis_special_characters(sentiment_analyzer):
    text = "Stock price up 10%! 🚀 #bullish"
    result = await sentiment_analyzer.analyze_text(text)
    
    assert isinstance(result, dict)
    assert "sentiment" in result
    assert "confidence" in result
    assert "probabilities" in result

def test_sentiment_score_calculation(sentiment_analyzer):
    results = [
        {"sentiment": "positive", "confidence": 0.8},
        {"sentiment": "negative", "confidence": 0.6},
        {"sentiment": "neutral", "confidence": 0.7}
    ]
    
    score = sentiment_analyzer.calculate_sentiment_score(results)
    assert isinstance(score, float)
    assert -1 <= score <= 1

def test_sentiment_score_with_weights(sentiment_analyzer):
    results = [
        {"sentiment": "positive", "confidence": 0.8},
        {"sentiment": "negative", "confidence": 0.6},
        {"sentiment": "neutral", "confidence": 0.7}
    ]
    weights = {"positive": 0.5, "negative": 0.3, "neutral": 0.2}
    
    score = sentiment_analyzer.calculate_sentiment_score(results, weights=weights)
    assert isinstance(score, float)
    assert -1 <= score <= 1

@pytest.mark.asyncio
async def test_sentiment_analysis_error_handling(sentiment_analyzer):
    # Test with invalid input
    with pytest.raises(Exception):
        await sentiment_analyzer.analyze_text(None)
    
    # Test with non-string input
    with pytest.raises(Exception):
        await sentiment_analyzer.analyze_text(123)

@pytest.mark.asyncio
async def test_sentiment_analysis_batch_error_handling(sentiment_analyzer):
    # Test with empty list
    with pytest.raises(ValueError):
        await sentiment_analyzer.analyze_batch([])
    
    # Test with invalid input
    with pytest.raises(Exception):
        await sentiment_analyzer.analyze_batch([None, "valid text"])
    
    # Test with non-string input
    with pytest.raises(Exception):
        await sentiment_analyzer.analyze_batch(["valid text", 123]) 