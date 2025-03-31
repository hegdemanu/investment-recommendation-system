import pytest
import pandas as pd
from app.data.market_data import MarketDataPipeline

@pytest.fixture
def market_data_pipeline():
    return MarketDataPipeline()

@pytest.mark.asyncio
async def test_fetch_market_data(market_data_pipeline):
    symbol = "AAPL"
    start_date = "2023-01-01"
    end_date = "2024-01-01"
    
    data = await market_data_pipeline.fetch_market_data(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date
    )
    
    assert isinstance(data, pd.DataFrame)
    assert not data.empty
    assert all(col in data.columns for col in ['open', 'high', 'low', 'close', 'volume'])
    assert len(data) > 0

@pytest.mark.asyncio
async def test_fetch_multiple_symbols(market_data_pipeline):
    symbols = ["AAPL", "GOOGL", "MSFT"]
    start_date = "2023-01-01"
    end_date = "2024-01-01"
    
    data = await market_data_pipeline.fetch_market_data(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date
    )
    
    assert isinstance(data, dict)
    assert all(symbol in data for symbol in symbols)
    assert all(isinstance(df, pd.DataFrame) for df in data.values())
    assert all(not df.empty for df in data.values())

@pytest.mark.asyncio
async def test_fetch_market_data_caching(market_data_pipeline):
    symbol = "AAPL"
    start_date = "2023-01-01"
    end_date = "2024-01-01"
    
    # First fetch
    data1 = await market_data_pipeline.fetch_market_data(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date
    )
    
    # Second fetch (should use cache)
    data2 = await market_data_pipeline.fetch_market_data(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date
    )
    
    pd.testing.assert_frame_equal(data1, data2)

@pytest.mark.asyncio
async def test_fetch_market_data_force_update(market_data_pipeline):
    symbol = "AAPL"
    start_date = "2023-01-01"
    end_date = "2024-01-01"
    
    # First fetch
    data1 = await market_data_pipeline.fetch_market_data(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date
    )
    
    # Second fetch with force update
    data2 = await market_data_pipeline.fetch_market_data(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        force_update=True
    )
    
    pd.testing.assert_frame_equal(data1, data2)

@pytest.mark.asyncio
async def test_fetch_market_data_error_handling(market_data_pipeline):
    # Test with invalid symbol
    with pytest.raises(ValueError):
        await market_data_pipeline.fetch_market_data(
            symbol="",
            start_date="2023-01-01",
            end_date="2024-01-01"
        )
    
    # Test with invalid date range
    with pytest.raises(ValueError):
        await market_data_pipeline.fetch_market_data(
            symbol="AAPL",
            start_date="2024-01-01",
            end_date="2023-01-01"
        )
    
    # Test with non-existent symbol
    with pytest.raises(Exception):
        await market_data_pipeline.fetch_market_data(
            symbol="INVALID_SYMBOL",
            start_date="2023-01-01",
            end_date="2024-01-01"
        )

@pytest.mark.asyncio
async def test_fetch_market_data_different_timeframes(market_data_pipeline):
    symbol = "AAPL"
    start_date = "2023-01-01"
    end_date = "2024-01-01"
    timeframes = ["1d", "1wk", "1mo"]
    
    for timeframe in timeframes:
        data = await market_data_pipeline.fetch_market_data(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            timeframe=timeframe
        )
        
        assert isinstance(data, pd.DataFrame)
        assert not data.empty
        assert all(col in data.columns for col in ['open', 'high', 'low', 'close', 'volume'])

@pytest.mark.asyncio
async def test_fetch_market_data_with_indicators(market_data_pipeline):
    symbol = "AAPL"
    start_date = "2023-01-01"
    end_date = "2024-01-01"
    
    data = await market_data_pipeline.fetch_market_data(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        include_indicators=True
    )
    
    assert isinstance(data, pd.DataFrame)
    assert not data.empty
    assert all(col in data.columns for col in [
        'open', 'high', 'low', 'close', 'volume',
        'rsi', 'macd', 'bollinger_upper', 'bollinger_lower'
    ])

@pytest.mark.asyncio
async def test_fetch_market_data_rate_limiting(market_data_pipeline):
    symbols = ["AAPL", "GOOGL", "MSFT", "AMZN", "META"]
    start_date = "2023-01-01"
    end_date = "2024-01-01"
    
    # Fetch multiple symbols with rate limiting
    data = await market_data_pipeline.fetch_market_data(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        rate_limit=1  # 1 second between requests
    )
    
    assert isinstance(data, dict)
    assert all(symbol in data for symbol in symbols)
    assert all(isinstance(df, pd.DataFrame) for df in data.values())
    assert all(not df.empty for df in data.values()) 