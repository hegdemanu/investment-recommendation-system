"""
Module for long-term stock price prediction models.
Implements ARIMA-GARCH and Prophet models for mid to long-term forecasting.
"""

import os
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
import matplotlib
# Set the backend to non-interactive 'Agg' to avoid GUI issues
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
import warnings
from prophet import Prophet

class LongTermPredictor:
    """
    Module for long-term stock price prediction using ARIMA-GARCH and Prophet models.
    """
    
    def __init__(self, models_dir="./models"):
        """
        Initialize the LongTermPredictor module.
        
        Parameters:
        -----------
        models_dir : str, optional
            Directory to save trained models
        """
        self.models_dir = models_dir
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
            print(f"Created models directory: {models_dir}")
        
        # Suppress some common warnings
        warnings.filterwarnings('ignore', 'Non-stationary starting autoregressive parameters')
        warnings.filterwarnings('ignore', 'The coefficient on the constant is not significant')
        
    def fit_arima_garch(self, data, ticker, p=1, d=1, q=0, vol_p=1, vol_q=1):
        """
        Fit ARIMA-GARCH model for a specific ticker.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Data for model training
        ticker : str
            The ticker symbol to train model for
        p, d, q : int, optional
            ARIMA model order parameters
        vol_p, vol_q : int, optional
            GARCH model order parameters
            
        Returns:
        --------
        tuple : (ARIMA model, GARCH model)
        """
        print(f"Fitting ARIMA-GARCH model for {ticker}...")
        
        # Filter data for the specific ticker
        ticker_data = data[data['ticker'] == ticker].sort_values('Date')
        
        if len(ticker_data) < 60:  # Require at least 60 data points
            print(f"Not enough data for {ticker}, need at least 60 data points.")
            return None, None
        
        # Extract price series
        if 'Price' in ticker_data.columns:
            price_series = ticker_data['Price']
        elif 'Close' in ticker_data.columns:
            price_series = ticker_data['Close']
        else:
            print(f"No price column found for {ticker}.")
            return None, None
        
        try:
            # Fit ARIMA model for returns
            returns = price_series.pct_change().dropna()
            
            # Try to fit ARIMA model, with fallback to simpler models if needed
            try:
                arima_model = ARIMA(returns, order=(p, d, q))
                arima_result = arima_model.fit()
            except:
                print(f"Error fitting ARIMA({p},{d},{q}) for {ticker}, trying simpler model...")
                try:
                    arima_model = ARIMA(returns, order=(1, 0, 0))  # Simple AR(1)
                    arima_result = arima_model.fit()
                except:
                    print(f"Error fitting AR(1) for {ticker}, using mean model.")
                    return None, None
            
            # Get ARIMA residuals for GARCH modeling
            residuals = arima_result.resid
            
            # Fit GARCH model to residuals
            try:
                garch_model = arch_model(residuals, vol='GARCH', p=vol_p, q=vol_q)
                garch_result = garch_model.fit(disp='off')
            except:
                print(f"Error fitting GARCH({vol_p},{vol_q}) for {ticker}, trying simpler model...")
                try:
                    garch_model = arch_model(residuals, vol='GARCH', p=1, q=1)  # Simple GARCH(1,1)
                    garch_result = garch_model.fit(disp='off')
                except:
                    print(f"Error fitting GARCH models for {ticker}, skipping volatility modeling.")
                    garch_result = None
            
            # Save models
            model_path = os.path.join(self.models_dir, f"{ticker}_arima_garch.pkl")
            with open(model_path, 'wb') as f:
                pickle.dump((arima_result, garch_result), f)
            print(f"ARIMA-GARCH model saved to {model_path}")
            
            return arima_result, garch_result
        
        except Exception as e:
            print(f"Error fitting ARIMA-GARCH model for {ticker}: {str(e)}")
            return None, None
    
    def fit_prophet(self, data, ticker, changepoint_prior_scale=0.05):
        """
        Fit Prophet model for a specific ticker.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Data for model training
        ticker : str
            The ticker symbol to train model for
        changepoint_prior_scale : float, optional
            Flexibility of the trend, higher values allow more trend changes
            
        Returns:
        --------
        Prophet : Trained Prophet model
        """
        print(f"Fitting Prophet model for {ticker}...")
        
        # Filter data for the specific ticker
        ticker_data = data[data['ticker'] == ticker].sort_values('Date')
        
        if len(ticker_data) < 60:  # Require at least 60 data points
            print(f"Not enough data for {ticker}, need at least 60 data points.")
            return None
        
        # Extract price series
        if 'Price' in ticker_data.columns:
            price_series = ticker_data['Price']
        elif 'Close' in ticker_data.columns:
            price_series = ticker_data['Close']
        else:
            print(f"No price column found for {ticker}.")
            return None
        
        try:
            # Prepare data for Prophet
            df_prophet = pd.DataFrame({
                'ds': ticker_data['Date'],
                'y': price_series
            })
            
            # Fit Prophet model
            model = Prophet(
                changepoint_prior_scale=changepoint_prior_scale,
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=False
            )
            
            # Add Indian stock market holidays (Optional enhancement)
            # model.add_country_holidays(country_name='IN')
            
            model.fit(df_prophet)
            
            # Save model
            model_path = os.path.join(self.models_dir, f"{ticker}_prophet.pkl")
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            print(f"Prophet model saved to {model_path}")
            
            return model
        
        except Exception as e:
            print(f"Error fitting Prophet model for {ticker}: {str(e)}")
            return None
    
    def fit_long_term_models(self, data):
        """
        Fit long-term prediction models for all tickers in the data.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Data for model training
            
        Returns:
        --------
        dict : Dictionary of trained models
        """
        result = {
            "arima_garch": {},
            "prophet": {}
        }
        
        unique_tickers = data['ticker'].unique()
        print(f"Fitting long-term models for {len(unique_tickers)} tickers...")
        
        for ticker in unique_tickers:
            # Fit ARIMA-GARCH model
            arima_result, garch_result = self.fit_arima_garch(data, ticker)
            if arima_result is not None:
                result["arima_garch"][ticker] = (arima_result, garch_result)
            
            # Fit Prophet model
            prophet_model = self.fit_prophet(data, ticker)
            if prophet_model is not None:
                result["prophet"][ticker] = prophet_model
        
        print(f"Successfully fitted ARIMA-GARCH models for {len(result['arima_garch'])} tickers.")
        print(f"Successfully fitted Prophet models for {len(result['prophet'])} tickers.")
        return result
    
    def load_long_term_models(self):
        """
        Load pre-trained long-term prediction models.
        
        Returns:
        --------
        dict : Dictionary of loaded models
        """
        result = {
            "arima_garch": {},
            "prophet": {}
        }
        
        arima_garch_files = [f for f in os.listdir(self.models_dir) if f.endswith("_arima_garch.pkl")]
        prophet_files = [f for f in os.listdir(self.models_dir) if f.endswith("_prophet.pkl")]
        
        print(f"Loading {len(arima_garch_files)} ARIMA-GARCH models and {len(prophet_files)} Prophet models...")
        
        # Load ARIMA-GARCH models
        for model_file in arima_garch_files:
            ticker = model_file.replace("_arima_garch.pkl", "")
            model_path = os.path.join(self.models_dir, model_file)
            
            try:
                with open(model_path, 'rb') as f:
                    arima_result, garch_result = pickle.load(f)
                
                result["arima_garch"][ticker] = (arima_result, garch_result)
                print(f"Successfully loaded ARIMA-GARCH model for {ticker}")
            except Exception as e:
                print(f"Error loading ARIMA-GARCH model for {ticker}: {str(e)}")
        
        # Load Prophet models
        for model_file in prophet_files:
            ticker = model_file.replace("_prophet.pkl", "")
            model_path = os.path.join(self.models_dir, model_file)
            
            try:
                with open(model_path, 'rb') as f:
                    prophet_model = pickle.load(f)
                
                result["prophet"][ticker] = prophet_model
                print(f"Successfully loaded Prophet model for {ticker}")
            except Exception as e:
                print(f"Error loading Prophet model for {ticker}: {str(e)}")
        
        print(f"Successfully loaded {len(result['arima_garch'])} ARIMA-GARCH models and {len(result['prophet'])} Prophet models.")
        return result
    
    def predict_arima_garch(self, models_dict, data, horizon="long"):
        """
        Generate predictions using ARIMA-GARCH models.
        
        Parameters:
        -----------
        models_dict : dict
            Dictionary of trained models
        data : pd.DataFrame
            Data to base predictions on
        horizon : str, optional
            Time horizon for predictions - "mid" or "long"
            
        Returns:
        --------
        pd.DataFrame : Predictions
        """
        if "arima_garch" not in models_dict or not models_dict["arima_garch"]:
            print("No ARIMA-GARCH models available for prediction.")
            return pd.DataFrame()
        
        predictions = []
        
        # Define the number of days to predict based on horizon
        if horizon == "mid":
            prediction_days = [63, 126]  # ~3 months, ~6 months
            labels = ["next_quarter", "next_half_year"]
        elif horizon == "long":
            prediction_days = [252]  # ~1 year
            labels = ["next_year"]
        else:
            raise ValueError("Invalid horizon for ARIMA-GARCH. Choose 'mid' or 'long'.")
        
        for ticker, ticker_data in data.groupby('ticker'):
            if ticker not in models_dict["arima_garch"]:
                print(f"No ARIMA-GARCH model found for {ticker}, skipping predictions...")
                continue
            
            arima_result, garch_result = models_dict["arima_garch"][ticker]
            
            # Sort by date
            ticker_data = ticker_data.sort_values('Date')
            
            # Get latest price
            if 'Price' in ticker_data.columns:
                latest_price = ticker_data['Price'].iloc[-1]
            elif 'Close' in ticker_data.columns:
                latest_price = ticker_data['Close'].iloc[-1]
            else:
                print(f"No price column found for {ticker}.")
                continue
            
            # Get latest date
            latest_date = ticker_data['Date'].iloc[-1]
            
            ticker_pred = {
                'ticker': ticker,
                'last_date': latest_date,
                'latest_price': latest_price
            }
            
            # Make predictions for each time horizon
            for days, label in zip(prediction_days, labels):
                try:
                    # Forecast returns
                    return_forecast = arima_result.forecast(steps=days)
                    
                    # Forecast volatility if GARCH model is available
                    if garch_result is not None:
                        vol_forecast = garch_result.forecast(horizon=days).variance.iloc[-1]
                        
                        # Use mean volatility for simulation
                        mean_volatility = np.sqrt(vol_forecast.mean())
                    else:
                        # Use historical volatility if GARCH is not available
                        returns = ticker_data['Price'].pct_change().dropna() if 'Price' in ticker_data.columns else ticker_data['Close'].pct_change().dropna()
                        mean_volatility = returns.std() * np.sqrt(days)
                    
                    # Simulate price paths (simple version)
                    cumulative_return = np.sum(return_forecast)
                    
                    # Add uncertainty based on volatility
                    upper_return = cumulative_return + 1.96 * mean_volatility
                    lower_return = cumulative_return - 1.96 * mean_volatility
                    
                    # Calculate predicted prices
                    predicted_price = latest_price * (1 + cumulative_return)
                    upper_price = latest_price * (1 + upper_return)
                    lower_price = latest_price * (1 + lower_return)
                    
                    # Calculate percentage change
                    predicted_change = ((predicted_price / latest_price) - 1) * 100
                    
                    # Add to the prediction dictionary
                    ticker_pred[f'{label}_price'] = predicted_price
                    ticker_pred[f'{label}_change'] = predicted_change
                    ticker_pred[f'{label}_upper'] = upper_price
                    ticker_pred[f'{label}_lower'] = lower_price
                    
                except Exception as e:
                    print(f"Error generating ARIMA-GARCH prediction for {ticker}: {str(e)}")
                    continue
            
            predictions.append(ticker_pred)
        
        if predictions:
            return pd.DataFrame(predictions)
        else:
            print("No ARIMA-GARCH predictions generated.")
            return pd.DataFrame()
    
    def predict_prophet(self, models_dict, data, horizon="long"):
        """
        Generate predictions using Prophet models.
        
        Parameters:
        -----------
        models_dict : dict
            Dictionary of trained models
        data : pd.DataFrame
            Data to base predictions on
        horizon : str, optional
            Time horizon for predictions - "mid" or "long"
            
        Returns:
        --------
        pd.DataFrame : Predictions
        """
        if "prophet" not in models_dict or not models_dict["prophet"]:
            print("No Prophet models available for prediction.")
            return pd.DataFrame()
        
        predictions = []
        
        # Define the number of days to predict based on horizon
        if horizon == "mid":
            prediction_days = [63, 126]  # ~3 months, ~6 months
            labels = ["next_quarter", "next_half_year"]
        elif horizon == "long":
            prediction_days = [252]  # ~1 year
            labels = ["next_year"]
        else:
            raise ValueError("Invalid horizon for Prophet. Choose 'mid' or 'long'.")
        
        for ticker, ticker_data in data.groupby('ticker'):
            if ticker not in models_dict["prophet"]:
                print(f"No Prophet model found for {ticker}, skipping predictions...")
                continue
            
            prophet_model = models_dict["prophet"][ticker]
            
            # Sort by date
            ticker_data = ticker_data.sort_values('Date')
            
            # Get latest price
            if 'Price' in ticker_data.columns:
                latest_price = ticker_data['Price'].iloc[-1]
            elif 'Close' in ticker_data.columns:
                latest_price = ticker_data['Close'].iloc[-1]
            else:
                print(f"No price column found for {ticker}.")
                continue
            
            # Get latest date
            latest_date = ticker_data['Date'].iloc[-1]
            
            ticker_pred = {
                'ticker': ticker,
                'last_date': latest_date,
                'latest_price': latest_price
            }
            
            # Create future dataframe for prediction
            max_days = max(prediction_days)
            future = prophet_model.make_future_dataframe(periods=max_days)
            
            # Generate forecast
            try:
                forecast = prophet_model.predict(future)
                
                # Extract predictions for each horizon
                for days, label in zip(prediction_days, labels):
                    # Get prediction date
                    pred_date = (latest_date + pd.Timedelta(days=days)).strftime('%Y-%m-%d')
                    
                    # Get predicted value
                    pred_row = forecast[forecast['ds'] == pred_date]
                    
                    if not pred_row.empty:
                        predicted_price = pred_row['yhat'].values[0]
                        upper_price = pred_row['yhat_upper'].values[0]
                        lower_price = pred_row['yhat_lower'].values[0]
                        
                        # Calculate percentage change
                        predicted_change = ((predicted_price / latest_price) - 1) * 100
                        
                        # Add to the prediction dictionary
                        ticker_pred[f'{label}_price'] = predicted_price
                        ticker_pred[f'{label}_change'] = predicted_change
                        ticker_pred[f'{label}_upper'] = upper_price
                        ticker_pred[f'{label}_lower'] = lower_price
                    else:
                        print(f"No Prophet prediction available for {ticker} on {pred_date}")
            
            except Exception as e:
                print(f"Error generating Prophet prediction for {ticker}: {str(e)}")
                continue
            
            predictions.append(ticker_pred)
        
        if predictions:
            return pd.DataFrame(predictions)
        else:
            print("No Prophet predictions generated.")
            return pd.DataFrame()
    
    def predict_ensemble(self, models_dict, data, horizon="long", weights=None):
        """
        Generate ensemble predictions by combining ARIMA-GARCH and Prophet.
        
        Parameters:
        -----------
        models_dict : dict
            Dictionary of trained models
        data : pd.DataFrame
            Data to base predictions on
        horizon : str, optional
            Time horizon for predictions - "mid" or "long"
        weights : dict, optional
            Weights for each model (e.g., {'arima_garch': 0.5, 'prophet': 0.5})
            
        Returns:
        --------
        pd.DataFrame : Ensemble predictions
        """
        # Default weights give equal importance to both models
        if weights is None:
            weights = {'arima_garch': 0.5, 'prophet': 0.5}
        
        # Normalize weights to sum to 1
        total_weight = sum(weights.values())
        weights = {k: v/total_weight for k, v in weights.items()}
        
        # Get predictions from individual models
        arima_garch_preds = self.predict_arima_garch(models_dict, data, horizon)
        prophet_preds = self.predict_prophet(models_dict, data, horizon)
        
        if arima_garch_preds.empty and prophet_preds.empty:
            print("No predictions available for ensemble.")
            return pd.DataFrame()
        
        # Combine predictions
        ensemble_preds = []
        
        # Get all unique tickers from both prediction sets
        all_tickers = set()
        if not arima_garch_preds.empty:
            all_tickers.update(arima_garch_preds['ticker'])
        if not prophet_preds.empty:
            all_tickers.update(prophet_preds['ticker'])
        
        # Define the prediction labels based on horizon
        if horizon == "mid":
            labels = ["next_quarter", "next_half_year"]
        elif horizon == "long":
            labels = ["next_year"]
        else:
            raise ValueError("Invalid horizon for ensemble. Choose 'mid' or 'long'.")
        
        # Combine predictions for each ticker
        for ticker in all_tickers:
            ticker_ag_pred = None if arima_garch_preds.empty else arima_garch_preds[arima_garch_preds['ticker'] == ticker]
            ticker_prophet_pred = None if prophet_preds.empty else prophet_preds[prophet_preds['ticker'] == ticker]
            
            # Skip if both predictions are missing
            if (ticker_ag_pred is None or ticker_ag_pred.empty) and (ticker_prophet_pred is None or ticker_prophet_pred.empty):
                continue
            
            # Start with base ticker info
            if ticker_ag_pred is not None and not ticker_ag_pred.empty:
                ticker_pred = {
                    'ticker': ticker,
                    'last_date': ticker_ag_pred['last_date'].iloc[0],
                    'latest_price': ticker_ag_pred['latest_price'].iloc[0]
                }
            else:
                ticker_pred = {
                    'ticker': ticker,
                    'last_date': ticker_prophet_pred['last_date'].iloc[0],
                    'latest_price': ticker_prophet_pred['latest_price'].iloc[0]
                }
            
            # Combine predictions for each label
            for label in labels:
                price_key = f'{label}_price'
                change_key = f'{label}_change'
                
                # Get predictions from each model (if available)
                ag_price = None if ticker_ag_pred is None or ticker_ag_pred.empty or price_key not in ticker_ag_pred.columns else ticker_ag_pred[price_key].iloc[0]
                prophet_price = None if ticker_prophet_pred is None or ticker_prophet_pred.empty or price_key not in ticker_prophet_pred.columns else ticker_prophet_pred[price_key].iloc[0]
                
                ag_change = None if ticker_ag_pred is None or ticker_ag_pred.empty or change_key not in ticker_ag_pred.columns else ticker_ag_pred[change_key].iloc[0]
                prophet_change = None if ticker_prophet_pred is None or ticker_prophet_pred.empty or change_key not in ticker_prophet_pred.columns else ticker_prophet_pred[change_key].iloc[0]
                
                # Calculate ensemble prediction
                if ag_price is not None and prophet_price is not None:
                    # Weight average of both models
                    ticker_pred[price_key] = weights['arima_garch'] * ag_price + weights['prophet'] * prophet_price
                    ticker_pred[change_key] = weights['arima_garch'] * ag_change + weights['prophet'] * prophet_change
                elif ag_price is not None:
                    # Only ARIMA-GARCH available
                    ticker_pred[price_key] = ag_price
                    ticker_pred[change_key] = ag_change
                elif prophet_price is not None:
                    # Only Prophet available
                    ticker_pred[price_key] = prophet_price
                    ticker_pred[change_key] = prophet_change
            
            ensemble_preds.append(ticker_pred)
        
        if ensemble_preds:
            return pd.DataFrame(ensemble_preds)
        else:
            print("No ensemble predictions generated.")
            return pd.DataFrame()