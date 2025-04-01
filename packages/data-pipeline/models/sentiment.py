import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from app.core.sentiment_weighted_model import SentimentWeightedModel

# Create directories
os.makedirs('results/sentiment_models', exist_ok=True)

# Create sample data for AAPL
print('Creating sample data for training...')
dates = pd.date_range(start='2022-01-01', end='2023-01-01', freq='D')
prices = np.random.normal(150, 10, len(dates)) + np.linspace(0, 20, len(dates))
volumes = np.random.normal(5000000, 1000000, len(dates))

# Add realistic features including technical indicators
def add_technical_indicators(df):
    """Add technical indicators to the dataframe"""
    # Moving averages
    df['ma5'] = df['close'].rolling(window=5).mean()
    df['ma20'] = df['close'].rolling(window=20).mean()
    
    # Relative Strength Index (simplified)
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # MACD (simplified)
    ema12 = df['close'].ewm(span=12).mean()
    ema26 = df['close'].ewm(span=26).mean()
    df['macd'] = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    
    # Bollinger Bands
    df['bb_middle'] = df['close'].rolling(window=20).mean()
    std = df['close'].rolling(window=20).std()
    df['bb_upper'] = df['bb_middle'] + (std * 2)
    df['bb_lower'] = df['bb_middle'] - (std * 2)
    
    # Clean NaN values
    return df.dropna()

# Create features
price_data = pd.DataFrame({
    'date': dates,
    'close': prices,
    'open': prices * 0.99 + np.random.normal(0, 1, len(dates)),
    'high': prices * 1.02 + np.random.normal(0, 1, len(dates)),
    'low': prices * 0.98 + np.random.normal(0, 1, len(dates)),
    'volume': volumes,
}).set_index('date')

# Add technical indicators
price_data = add_technical_indicators(price_data)
print(f'Generated price data with {len(price_data)} rows and {len(price_data.columns)} features')

# Create more realistic sample news data
print('Creating sample news data...')
news_data = []
# More frequent news near earnings dates (quarterly)
earning_dates = [
    '2022-01-25', '2022-04-26', '2022-07-25', '2022-10-25', 
    '2023-01-24'
]

for date in price_data.index:
    date_str = date.strftime('%Y-%m-%d')
    
    # More news near earnings dates
    if any(abs((date - pd.Timestamp(e_date)).days) <= 3 for e_date in earning_dates):
        if np.random.random() < 0.8:  # 80% chance of news on days near earnings
            sentiment_probs = [0.5, 0.3, 0.2]  # More bullish near earnings
            sentiment = np.random.choice(['bullish', 'neutral', 'bearish'], p=sentiment_probs)
            news_data.append({
                'title': f'AAPL Earnings News {date_str}',
                'description': f'Earnings related news for AAPL with {sentiment} sentiment',
                'publishedAt': date_str
            })
    # Regular days - less frequent news
    elif np.random.random() < 0.1:  # 10% chance of news on normal days
        sentiment_probs = [0.4, 0.4, 0.2]
        sentiment = np.random.choice(['bullish', 'neutral', 'bearish'], p=sentiment_probs)
        news_data.append({
            'title': f'AAPL News {date_str}',
            'description': f'Regular news for AAPL with {sentiment} sentiment',
            'publishedAt': date_str
        })

print(f'Generated {len(news_data)} news articles')

# Train each model type
model_types = ['lstm', 'rf', 'gbr', 'linear']
results = {}

for model_type in model_types:
    print(f'Training {model_type} model...')
    model = SentimentWeightedModel(model_type=model_type)
    
    # Train with sentiment analysis
    impact = model.get_sentiment_impact(price_data, news_data)
    results[model_type] = impact
    
    print(f'{model_type.upper()} Model Results:')
    print(f'RMSE with sentiment: {impact["with_sentiment"]["rmse"]:.4f}')
    print(f'RMSE without sentiment: {impact["without_sentiment"]["rmse"]:.4f}')
    print(f'Improvement: {impact["rmse_pct_improvement"]:.2f}%')
    
    # Save model
    model.save(f'models/sentiment_models/{model_type}/aapl')
    print(f'Model saved to models/sentiment_models/{model_type}/aapl')
    print('-------------------')

# Compare all models
print("\nModel Comparison Summary:")
print("="*50)
print(f"{'Model Type':<10} {'With Sentiment':<15} {'Without Sentiment':<20} {'Improvement %':<15}")
print("-"*50)
for model_type, impact in results.items():
    with_rmse = impact["with_sentiment"]["rmse"]
    without_rmse = impact["without_sentiment"]["rmse"]
    pct_improvement = impact["rmse_pct_improvement"]
    print(f"{model_type:<10} {with_rmse:<15.4f} {without_rmse:<20.4f} {pct_improvement:<15.2f}")

# Create visualizations
plt.figure(figsize=(12, 10))

# Plot 1: RMSE Comparison
plt.subplot(2, 1, 1)
model_names = list(results.keys())
with_sentiment = [results[m]["with_sentiment"]["rmse"] for m in model_names]
without_sentiment = [results[m]["without_sentiment"]["rmse"] for m in model_names]

x = np.arange(len(model_names))
width = 0.35

plt.bar(x - width/2, with_sentiment, width, label='With Sentiment')
plt.bar(x + width/2, without_sentiment, width, label='Without Sentiment')

plt.xlabel('Model Type')
plt.ylabel('RMSE (Lower is Better)')
plt.title('RMSE Comparison With and Without Sentiment Analysis')
plt.xticks(x, [m.upper() for m in model_names])
plt.legend()

# Plot 2: Improvement Percentage
plt.subplot(2, 1, 2)
improvements = [results[m]["rmse_pct_improvement"] for m in model_names]
colors = ['green' if i > 0 else 'red' for i in improvements]

plt.bar(x, improvements, color=colors)
plt.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
plt.xlabel('Model Type')
plt.ylabel('Improvement Percentage')
plt.title('Sentiment Analysis Impact (% Improvement in RMSE)')
plt.xticks(x, [m.upper() for m in model_names])

plt.tight_layout()
plt.savefig('results/sentiment_models/model_comparison.png')
print("Visualizations saved to results/sentiment_models/model_comparison.png") 