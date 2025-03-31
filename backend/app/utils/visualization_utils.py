"""
Visualization utility functions for the investment recommendation system.

This module provides utility functions for creating charts and visualizations
for investment data analysis.
"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Any, Union, Optional, Tuple
import matplotlib.dates as mdates
from matplotlib.figure import Figure
from datetime import datetime, timedelta

# Set up logging
logger = logging.getLogger(__name__)

# Set default styling
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette('viridis')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.titleweight'] = 'bold'

def plot_stock_price(data: pd.DataFrame, ticker: str = "", 
                    start_date: Optional[str] = None,
                    end_date: Optional[str] = None,
                    ma_periods: List[int] = [20, 50, 200],
                    volume: bool = True,
                    figsize: Tuple[int, int] = (12, 8)) -> Figure:
    """
    Plot stock price chart with moving averages and volume.
    
    Args:
        data: DataFrame with stock price data
        ticker: Stock ticker symbol for title
        start_date: Start date to plot from (None for all data)
        end_date: End date to plot to (None for all data)
        ma_periods: List of periods for moving averages
        volume: Whether to include volume subplot
        figsize: Figure size
        
    Returns:
        Matplotlib Figure object
    """
    try:
        # Filter data by date if specified
        if start_date:
            data = data[data.index >= start_date]
        if end_date:
            data = data[data.index <= end_date]
            
        # Create figure
        fig = plt.figure(figsize=figsize)
        
        # Determine subplot layout
        if volume and 'Volume' in data.columns:
            price_ax = plt.subplot2grid((5, 1), (0, 0), rowspan=3)
            volume_ax = plt.subplot2grid((5, 1), (3, 0), rowspan=1, sharex=price_ax)
            plt.subplots_adjust(hspace=0.1)
        else:
            price_ax = plt.subplot2grid((1, 1), (0, 0))
        
        # Plot price
        price_ax.plot(data.index, data['Close'], label='Close', linewidth=2)
        
        # Plot moving averages
        for period in ma_periods:
            if f'SMA_{period}' in data.columns:
                price_ax.plot(data.index, data[f'SMA_{period}'], 
                           label=f'SMA {period}', alpha=0.7)
            else:
                ma_col = f'MA_{period}'
                data[ma_col] = data['Close'].rolling(window=period).mean()
                price_ax.plot(data.index, data[ma_col], 
                           label=f'MA {period}', alpha=0.7)
        
        # Format price axis
        price_ax.set_ylabel('Price')
        price_ax.legend(loc='upper left')
        price_ax.set_title(f"{ticker} Stock Price" if ticker else "Stock Price")
        price_ax.grid(True, alpha=0.3)
        
        # Format x-axis dates
        if len(data) > 0:
            date_range = (data.index.max() - data.index.min()).days
            if date_range <= 30:
                price_ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
                price_ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
            elif date_range <= 180:
                price_ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
                price_ax.xaxis.set_major_locator(mdates.MonthLocator())
            else:
                price_ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
                price_ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        
        # Plot volume if requested
        if volume and 'Volume' in data.columns:
            volume_ax.bar(data.index, data['Volume'], alpha=0.5, color='navy')
            volume_ax.set_ylabel('Volume')
            volume_ax.grid(True, alpha=0.3)
            
            # Format volume numbers (K, M, B)
            import matplotlib.ticker as ticker
            def volume_formatter(x, pos):
                if x >= 1e9:
                    return f'{x/1e9:.1f}B'
                elif x >= 1e6:
                    return f'{x/1e6:.1f}M'
                elif x >= 1e3:
                    return f'{x/1e3:.1f}K'
                else:
                    return f'{x:.0f}'
            
            volume_ax.yaxis.set_major_formatter(ticker.FuncFormatter(volume_formatter))
        
        plt.tight_layout()
        logger.info(f"Generated stock price chart for {ticker or 'data'}")
        return fig
    
    except Exception as e:
        logger.error(f"Error creating stock price chart: {str(e)}")
        # Return empty figure on error
        return plt.figure()

def plot_technical_indicators(data: pd.DataFrame, ticker: str = "",
                             indicators: List[str] = ['RSI', 'MACD'],
                             figsize: Tuple[int, int] = (12, 10)) -> Figure:
    """
    Plot technical indicators chart.
    
    Args:
        data: DataFrame with price and indicator data
        ticker: Stock ticker symbol for title
        indicators: List of indicators to plot
        figsize: Figure size
        
    Returns:
        Matplotlib Figure object
    """
    try:
        # Create figure
        fig = plt.figure(figsize=figsize)
        
        # Determine number of subplots needed
        n_indicators = len(indicators)
        
        # First subplot is always price
        price_ax = plt.subplot2grid((n_indicators + 1, 1), (0, 0))
        
        # Plot price
        price_ax.plot(data.index, data['Close'], label='Close', linewidth=2)
        
        # Add Bollinger Bands if available
        if all(col in data.columns for col in ['BB_Upper', 'BB_Middle', 'BB_Lower']):
            price_ax.plot(data.index, data['BB_Upper'], 'g--', alpha=0.5, label='Upper BB')
            price_ax.plot(data.index, data['BB_Middle'], 'g-', alpha=0.5, label='Middle BB')
            price_ax.plot(data.index, data['BB_Lower'], 'g--', alpha=0.5, label='Lower BB')
        
        # Format price subplot
        price_ax.set_ylabel('Price')
        price_ax.legend(loc='upper left')
        price_ax.set_title(f"{ticker} Technical Analysis" if ticker else "Technical Analysis")
        price_ax.grid(True, alpha=0.3)
        
        # Create indicator subplots
        for i, indicator in enumerate(indicators, 1):
            if indicator == 'RSI' and 'RSI' in data.columns:
                ax = plt.subplot2grid((n_indicators + 1, 1), (i, 0), sharex=price_ax)
                ax.plot(data.index, data['RSI'], 'purple', label='RSI')
                ax.axhline(70, color='r', linestyle='--', alpha=0.5)
                ax.axhline(30, color='g', linestyle='--', alpha=0.5)
                ax.set_ylabel('RSI')
                ax.set_ylim(0, 100)
                ax.grid(True, alpha=0.3)
                
            elif indicator == 'MACD' and 'MACD' in data.columns:
                ax = plt.subplot2grid((n_indicators + 1, 1), (i, 0), sharex=price_ax)
                ax.plot(data.index, data['MACD'], 'blue', label='MACD')
                
                if 'MACD_Signal' in data.columns:
                    ax.plot(data.index, data['MACD_Signal'], 'red', label='Signal')
                
                if 'MACD_Hist' in data.columns:
                    ax.bar(data.index, data['MACD_Hist'], color='green', alpha=0.5, label='Histogram')
                
                ax.set_ylabel('MACD')
                ax.legend(loc='upper left')
                ax.grid(True, alpha=0.3)
                
            elif indicator == 'Stochastic' and '%K' in data.columns and '%D' in data.columns:
                ax = plt.subplot2grid((n_indicators + 1, 1), (i, 0), sharex=price_ax)
                ax.plot(data.index, data['%K'], 'orange', label='%K')
                ax.plot(data.index, data['%D'], 'blue', label='%D')
                ax.axhline(80, color='r', linestyle='--', alpha=0.5)
                ax.axhline(20, color='g', linestyle='--', alpha=0.5)
                ax.set_ylabel('Stochastic')
                ax.set_ylim(0, 100)
                ax.legend(loc='upper left')
                ax.grid(True, alpha=0.3)
                
            elif indicator in data.columns:
                ax = plt.subplot2grid((n_indicators + 1, 1), (i, 0), sharex=price_ax)
                ax.plot(data.index, data[indicator], label=indicator)
                ax.set_ylabel(indicator)
                ax.legend(loc='upper left')
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        logger.info(f"Generated technical indicator chart for {ticker or 'data'}")
        return fig
    
    except Exception as e:
        logger.error(f"Error creating technical indicator chart: {str(e)}")
        # Return empty figure on error
        return plt.figure()

def plot_correlation_matrix(data: pd.DataFrame, method: str = 'pearson',
                           figsize: Tuple[int, int] = (10, 8),
                           cmap: str = 'coolwarm') -> Figure:
    """
    Plot correlation matrix heatmap for numeric columns.
    
    Args:
        data: DataFrame with numeric data
        method: Correlation method ('pearson', 'kendall', 'spearman')
        figsize: Figure size
        cmap: Colormap for heatmap
        
    Returns:
        Matplotlib Figure object
    """
    try:
        # Select numeric columns
        numeric_data = data.select_dtypes(include=['number'])
        
        # Calculate correlation matrix
        corr = numeric_data.corr(method=method)
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create heatmap
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, cmap=cmap, annot=True, fmt=".2f",
                   linewidths=0.5, ax=ax, vmin=-1, vmax=1)
        
        # Format plot
        ax.set_title(f'Correlation Matrix ({method.capitalize()})')
        plt.tight_layout()
        
        logger.info(f"Generated correlation matrix for {len(numeric_data.columns)} variables")
        return fig
    
    except Exception as e:
        logger.error(f"Error creating correlation matrix: {str(e)}")
        # Return empty figure on error
        return plt.figure()

def plot_returns_distribution(data: pd.DataFrame, column: str = 'Close',
                             periods: List[int] = [1, 5, 20],
                             figsize: Tuple[int, int] = (12, 6)) -> Figure:
    """
    Plot distribution of returns over different periods.
    
    Args:
        data: DataFrame with price data
        column: Column to calculate returns from
        periods: List of periods for return calculation
        figsize: Figure size
        
    Returns:
        Matplotlib Figure object
    """
    try:
        # Calculate returns for each period
        returns_df = pd.DataFrame(index=data.index)
        
        for period in periods:
            returns_df[f'{period}d'] = data[column].pct_change(period) * 100
        
        # Create figure
        fig, axes = plt.subplots(1, len(periods), figsize=figsize, sharey=True)
        
        # Plot histograms for each period
        for i, period in enumerate(periods):
            ax = axes[i] if len(periods) > 1 else axes
            ax.hist(returns_df[f'{period}d'].dropna(), bins=30, alpha=0.7)
            
            # Add normal distribution fit
            x = returns_df[f'{period}d'].dropna()
            if len(x) > 0:
                mu, sigma = x.mean(), x.std()
                x_range = np.linspace(x.min(), x.max(), 100)
                y = ((1 / (np.sqrt(2 * np.pi) * sigma)) * 
                    np.exp(-0.5 * ((x_range - mu) / sigma) ** 2) * 
                    len(x) * (x.max() - x.min()) / 30)
                ax.plot(x_range, y, 'r--', linewidth=2)
                
                # Add statistics
                ax.axvline(mu, color='r', linestyle='-', alpha=0.7)
                ax.axvline(0, color='k', linestyle='--', alpha=0.7)
                
                # Add text with statistics
                stats_text = f"Mean: {mu:.2f}%\nStd: {sigma:.2f}%\nSkew: {x.skew():.2f}\nKurtosis: {x.kurtosis():.2f}"
                ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, 
                       verticalalignment='top', bbox=dict(boxstyle='round', alpha=0.2))
            
            ax.set_title(f'{period}-Day Returns')
            ax.set_xlabel('Return (%)')
            
            if i == 0:
                ax.set_ylabel('Frequency')
        
        plt.tight_layout()
        logger.info(f"Generated returns distribution chart for {len(periods)} periods")
        return fig
    
    except Exception as e:
        logger.error(f"Error creating returns distribution chart: {str(e)}")
        # Return empty figure on error
        return plt.figure()

def plot_comparison_chart(data: Dict[str, pd.DataFrame], column: str = 'Close',
                         start_date: Optional[str] = None,
                         end_date: Optional[str] = None,
                         normalize: bool = True,
                         figsize: Tuple[int, int] = (12, 6),
                         title: str = "Price Comparison") -> Figure:
    """
    Plot comparison chart for multiple stocks.
    
    Args:
        data: Dictionary of DataFrames with stock data
        column: Column to plot
        start_date: Start date to plot from
        end_date: End date to plot to
        normalize: Whether to normalize prices to 100 at start
        figsize: Figure size
        title: Chart title
        
    Returns:
        Matplotlib Figure object
    """
    try:
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Process and plot each stock
        for ticker, df in data.items():
            # Filter by date
            filtered = df.copy()
            if start_date:
                filtered = filtered[filtered.index >= start_date]
            if end_date:
                filtered = filtered[filtered.index <= end_date]
                
            if len(filtered) == 0:
                logger.warning(f"No data for {ticker} in selected date range")
                continue
                
            # Normalize if requested
            if normalize:
                base_value = filtered[column].iloc[0]
                values = filtered[column] / base_value * 100
                ax.plot(filtered.index, values, label=ticker, linewidth=2)
                y_label = 'Normalized Price (Initial=100)'
            else:
                ax.plot(filtered.index, filtered[column], label=ticker, linewidth=2)
                y_label = column
        
        # Format plot
        ax.set_ylabel(y_label)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left')
        
        # Format dates
        if len(data) > 0 and len(next(iter(data.values()))) > 0:
            first_df = next(iter(data.values()))
            date_range = (first_df.index.max() - first_df.index.min()).days
            if date_range <= 30:
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
                ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
            elif date_range <= 180:
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
                ax.xaxis.set_major_locator(mdates.MonthLocator())
            else:
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
                ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        
        plt.tight_layout()
        logger.info(f"Generated comparison chart for {len(data)} stocks")
        return fig
    
    except Exception as e:
        logger.error(f"Error creating comparison chart: {str(e)}")
        # Return empty figure on error
        return plt.figure()

def save_plot(fig: Figure, filename: str, directory: Union[str, Path] = "results",
             formats: List[str] = ['png', 'pdf'],
             dpi: int = 300) -> List[str]:
    """
    Save plot to file(s).
    
    Args:
        fig: Matplotlib Figure to save
        filename: Base filename (without extension)
        directory: Directory to save to
        formats: List of formats to save
        dpi: Resolution for raster formats
        
    Returns:
        List of saved filepaths
    """
    try:
        # Create directory if it doesn't exist
        dir_path = Path(directory)
        dir_path.mkdir(parents=True, exist_ok=True)
        
        # Save in each requested format
        saved_files = []
        
        for fmt in formats:
            filepath = dir_path / f"{filename}.{fmt}"
            fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
            saved_files.append(str(filepath))
            logger.info(f"Saved plot to {filepath}")
        
        return saved_files
    
    except Exception as e:
        logger.error(f"Error saving plot: {str(e)}")
        return []

def plot_forecast(historical_data: pd.DataFrame, forecast_data: pd.DataFrame,
                 date_col: str = 'ds', value_col: str = 'y',
                 forecast_col: str = 'yhat', 
                 upper_col: str = 'yhat_upper', lower_col: str = 'yhat_lower',
                 title: str = "Forecast",
                 figsize: Tuple[int, int] = (12, 6)) -> Figure:
    """
    Plot forecast chart with historical data and prediction intervals.
    
    Args:
        historical_data: DataFrame with historical data
        forecast_data: DataFrame with forecast data
        date_col: Column name for date
        value_col: Column name for actual values
        forecast_col: Column name for forecasted values
        upper_col: Column name for upper bound
        lower_col: Column name for lower bound
        title: Chart title
        figsize: Figure size
        
    Returns:
        Matplotlib Figure object
    """
    try:
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot historical data
        ax.plot(historical_data[date_col], historical_data[value_col], 
               label='Historical', color='blue', linewidth=2)
        
        # Plot forecast
        ax.plot(forecast_data[date_col], forecast_data[forecast_col], 
               label='Forecast', color='green', linewidth=2)
        
        # Plot prediction intervals if available
        if all(col in forecast_data.columns for col in [upper_col, lower_col]):
            ax.fill_between(forecast_data[date_col], 
                           forecast_data[lower_col], 
                           forecast_data[upper_col], 
                           color='green', alpha=0.2, label='95% Confidence Interval')
        
        # Add vertical line at the forecast start
        forecast_start = forecast_data[date_col].min()
        ax.axvline(forecast_start, color='r', linestyle='--', alpha=0.7, 
                  label='Forecast Start')
        
        # Format plot
        ax.set_title(title)
        ax.set_xlabel('Date')
        ax.set_ylabel('Value')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        # Format dates
        date_range = (forecast_data[date_col].max() - historical_data[date_col].min()).days
        if date_range <= 60:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=7))
        elif date_range <= 180:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax.xaxis.set_major_locator(mdates.MonthLocator())
        else:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
            
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        logger.info(f"Generated forecast chart")
        return fig
    
    except Exception as e:
        logger.error(f"Error creating forecast chart: {str(e)}")
        # Return empty figure on error
        return plt.figure()

def plot_sentiment_analysis(sentiment_data: pd.DataFrame, 
                           ticker: str = "",
                           date_col: str = 'date',
                           sentiment_col: str = 'sentiment_score',
                           price_data: Optional[pd.DataFrame] = None,
                           figsize: Tuple[int, int] = (12, 8)) -> Figure:
    """
    Plot sentiment analysis chart with optional price overlay.
    
    Args:
        sentiment_data: DataFrame with sentiment data
        ticker: Stock ticker symbol
        date_col: Column name for date
        sentiment_col: Column name for sentiment score
        price_data: Optional DataFrame with price data
        figsize: Figure size
        
    Returns:
        Matplotlib Figure object
    """
    try:
        # Create figure
        fig = plt.figure(figsize=figsize)
        
        # Determine if we need two subplots
        if price_data is not None and not price_data.empty:
            sentiment_ax = plt.subplot2grid((5, 1), (0, 0), rowspan=2)
            price_ax = plt.subplot2grid((5, 1), (2, 0), rowspan=3, sharex=sentiment_ax)
        else:
            sentiment_ax = plt.subplot2grid((1, 1), (0, 0))
        
        # Plot sentiment
        sentiment_ax.plot(sentiment_data[date_col], sentiment_data[sentiment_col], 
                       label='Sentiment', color='blue', marker='o', markersize=4)
        
        # Add horizontal line at neutral sentiment
        sentiment_ax.axhline(0, color='gray', linestyle='--', alpha=0.7)
        
        # Calculate rolling average if enough data points
        if len(sentiment_data) > 5:
            window = min(7, len(sentiment_data) // 2)
            sentiment_data['rolling_sentiment'] = sentiment_data[sentiment_col].rolling(window=window).mean()
            sentiment_ax.plot(sentiment_data[date_col], sentiment_data['rolling_sentiment'], 
                           label=f'{window}-Day Avg', color='red', linewidth=2)
        
        # Format sentiment axis
        sentiment_ax.set_ylabel('Sentiment Score')
        sentiment_ax.set_title(f"{ticker} Sentiment Analysis" if ticker else "Sentiment Analysis")
        sentiment_ax.legend(loc='upper left')
        sentiment_ax.grid(True, alpha=0.3)
        
        # Set y-axis limits with padding
        y_min = sentiment_data[sentiment_col].min()
        y_max = sentiment_data[sentiment_col].max()
        padding = (y_max - y_min) * 0.1 if y_max > y_min else 0.1
        sentiment_ax.set_ylim(y_min - padding, y_max + padding)
        
        # Plot price if available
        if price_data is not None and not price_data.empty:
            price_ax.plot(price_data.index, price_data['Close'], 
                       label='Price', color='green', linewidth=2)
            
            price_ax.set_ylabel('Price')
            price_ax.legend(loc='upper left')
            price_ax.grid(True, alpha=0.3)
            
            # Format x-axis dates
            date_range = (price_data.index.max() - price_data.index.min()).days
            if date_range <= 30:
                price_ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
                price_ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
            elif date_range <= 180:
                price_ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
                price_ax.xaxis.set_major_locator(mdates.MonthLocator())
            else:
                price_ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
                price_ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        
        plt.tight_layout()
        logger.info(f"Generated sentiment analysis chart for {ticker or 'data'}")
        return fig
    
    except Exception as e:
        logger.error(f"Error creating sentiment analysis chart: {str(e)}")
        # Return empty figure on error
        return plt.figure() 