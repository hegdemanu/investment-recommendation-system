"""
Risk Analysis Module for Investment Recommendation System
Classifies investments into risk categories based on multiple financial metrics
"""

import pandas as pd
import numpy as np
from datetime import datetime


class RiskAnalyzer:
    """
    Module for analyzing risk-reward profiles of investments.
    """
    
    def __init__(self):
        """Initialize the RiskAnalyzer module."""
        # Default metric weights for risk scoring
        self.default_weights = {
            'volatility': 0.20,
            'sharpe_ratio': 0.15,
            'pe_ratio': 0.10,
            'peg_ratio': 0.10,
            'debt_equity_ratio': 0.15,
            'cash_flow_growth': 0.10,
            'fear_greed_index': 0.10,
            'expected_return': 0.10
        }
    
    def classify(self, data, predictions, custom_weights=None):
        """
        Classify investments into risk categories.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Historical data
        predictions : pd.DataFrame
            Predicted future data
        custom_weights : dict, optional
            Custom weights for different metrics in risk calculation
            
        Returns:
        --------
        pd.DataFrame : Risk classifications
        """
        print("Analyzing risk profiles...")
        
        if predictions.empty:
            print("No predictions available for risk analysis.")
            return pd.DataFrame()
        
        # Use custom weights if provided, otherwise use defaults
        weights = custom_weights if custom_weights is not None else self.default_weights
        
        # Initialize results dataframe
        risk_profiles = []
        
        # Process each ticker
        for ticker, ticker_predictions in predictions.groupby('ticker'):
            print(f"Analyzing risk profile for {ticker}...")
            
            ticker_data = data[data['ticker'] == ticker]
            
            # Get the latest row for this ticker in predictions
            ticker_pred = ticker_predictions.iloc[-1].to_dict()
            
            # Basic metrics
            volatility = ticker_data['Volatility_20d'].mean() if 'Volatility_20d' in ticker_data.columns else np.nan
            
            # Calculate recent returns (last 30 days)
            recent_data = ticker_data.sort_values('Date').tail(30)
            if len(recent_data) > 1:
                recent_return = ((recent_data['Price'].iloc[-1] / recent_data['Price'].iloc[0]) - 1) * 100
            else:
                recent_return = np.nan
            
            # Get Sharpe ratio if available
            sharpe_ratio = ticker_data['Sharpe_Ratio'].iloc[-1] if 'Sharpe_Ratio' in ticker_data.columns else np.nan
            
            # Get PE and PEG ratios if available
            pe_ratio = ticker_data['PE Ratio'].iloc[-1] if 'PE Ratio' in ticker_data.columns else np.nan
            peg_ratio = ticker_data['PEG Ratio'].iloc[-1] if 'PEG Ratio' in ticker_data.columns else np.nan
            
            # Get debt-to-equity ratio if available
            debt_equity_ratio = ticker_data['debt_equity_ratio'].iloc[-1] if 'debt_equity_ratio' in ticker_data.columns else np.nan
            
            # Get cash flow growth if available
            cash_flow_growth = ticker_data['cash_flow_growth'].iloc[-1] if 'cash_flow_growth' in ticker_data.columns else np.nan
            
            # Get fear/greed index if available
            fear_greed_index = ticker_data['fear_greed_index'].iloc[-1] if 'fear_greed_index' in ticker_data.columns else np.nan
            
            # Get predicted returns
            next_month_change = ticker_pred.get('next_month_change', ticker_pred.get('next_quarter_change', np.nan))
            
            # Risk scoring based on individual metrics
            # 1. Volatility score (higher volatility = higher risk)
            if not np.isnan(volatility):
                volatility_score = min(max(volatility * 5, 1), 10)  # Scale to 1-10
            else:
                volatility_score = 5  # Default to medium
            
            # 2. Sharpe ratio score (higher Sharpe = lower risk)
            if not np.isnan(sharpe_ratio):
                sharpe_score = min(max(10 - sharpe_ratio * 2, 1), 10)  # Scale to 1-10, invert
            else:
                sharpe_score = 5  # Default to medium
            
            # 3. PE ratio score (higher PE = higher risk, assuming Indian market context)
            if not np.isnan(pe_ratio):
                if pe_ratio < 0:  # Negative earnings
                    pe_score = 10  # Very high risk
                else:
                    # Scale based on typical Indian market PE ratios
                    pe_score = min(max(pe_ratio / 5, 1), 10)  # Scale to 1-10
            else:
                pe_score = 5  # Default to medium
            
            # 4. PEG ratio score (higher PEG = higher risk)
            if not np.isnan(peg_ratio):
                if peg_ratio < 0:  # Negative growth
                    peg_score = 9  # High risk
                elif peg_ratio < 1:
                    peg_score = 3  # Low risk (good value)
                else:
                    peg_score = min(max(peg_ratio * 3, 1), 10)  # Scale to 1-10
            else:
                peg_score = 5  # Default to medium
            
            # 5. Debt-to-equity ratio score (higher ratio = higher risk)
            if not np.isnan(debt_equity_ratio):
                if debt_equity_ratio < 0:  # Negative equity
                    debt_equity_score = 10  # Very high risk
                else:
                    # Scale based on typical debt-to-equity ratios
                    debt_equity_score = min(max(debt_equity_ratio * 2.5, 1), 10)  # Scale to 1-10
            else:
                debt_equity_score = 5  # Default to medium
            
            # 6. Cash flow growth score (negative growth = higher risk)
            if not np.isnan(cash_flow_growth):
                if cash_flow_growth < -10:
                    cash_flow_score = 9  # High risk
                elif cash_flow_growth < 0:
                    cash_flow_score = 7  # Moderately high risk
                elif cash_flow_growth > 20:
                    cash_flow_score = 2  # Low risk
                else:
                    cash_flow_score = min(max(10 - cash_flow_growth / 3, 1), 10)  # Scale to 1-10, invert
            else:
                cash_flow_score = 5  # Default to medium
            
            # 7. Fear/greed index score (extreme values = higher risk)
            if not np.isnan(fear_greed_index):
                # Convert to a 1-10 scale where extreme values (0 or 100) are high risk
                # and neutral values (around 50) are lower risk
                fear_greed_score = min(max(abs(fear_greed_index - 50) / 5, 1), 10)
            else:
                fear_greed_score = 5  # Default to medium
            
            # 8. Expected return score (higher return = higher risk, generally)
            if not np.isnan(next_month_change):
                return_score = min(max(abs(next_month_change) / 2, 1), 10)  # Scale to 1-10
            else:
                return_score = 5  # Default to medium
            
            # Combine scores with different weights
            active_weights = {}
            
            # Assign weights - prioritize metrics that are available
            if not np.isnan(volatility):
                active_weights['volatility'] = weights.get('volatility', self.default_weights['volatility'])
            else:
                active_weights['volatility'] = 0
                
            if not np.isnan(sharpe_ratio):
                active_weights['sharpe'] = weights.get('sharpe_ratio', self.default_weights['sharpe_ratio'])
            else:
                active_weights['sharpe'] = 0
                
            if not np.isnan(pe_ratio):
                active_weights['pe'] = weights.get('pe_ratio', self.default_weights['pe_ratio'])
            else:
                active_weights['pe'] = 0
                
            if not np.isnan(peg_ratio):
                active_weights['peg'] = weights.get('peg_ratio', self.default_weights['peg_ratio'])
            else:
                active_weights['peg'] = 0
                
            if not np.isnan(debt_equity_ratio):
                active_weights['debt_equity'] = weights.get('debt_equity_ratio', self.default_weights['debt_equity_ratio'])
            else:
                active_weights['debt_equity'] = 0
                
            if not np.isnan(cash_flow_growth):
                active_weights['cash_flow'] = weights.get('cash_flow_growth', self.default_weights['cash_flow_growth'])
            else:
                active_weights['cash_flow'] = 0
                
            if not np.isnan(fear_greed_index):
                active_weights['fear_greed'] = weights.get('fear_greed_index', self.default_weights['fear_greed_index'])
            else:
                active_weights['fear_greed'] = 0
                
            if not np.isnan(next_month_change):
                active_weights['return'] = weights.get('expected_return', self.default_weights['expected_return'])
            else:
                active_weights['return'] = 0
            
            # Normalize weights to sum to 1
            total_weight = sum(active_weights.values())
            if total_weight > 0:
                active_weights = {k: v/total_weight for k, v in active_weights.items()}
            else:
                # If no metrics available, use equal weights for available metrics
                available_metrics = 0
                if not np.isnan(volatility): available_metrics += 1
                if not np.isnan(sharpe_ratio): available_metrics += 1
                if not np.isnan(pe_ratio): available_metrics += 1
                if not np.isnan(peg_ratio): available_metrics += 1
                if not np.isnan(debt_equity_ratio): available_metrics += 1
                if not np.isnan(cash_flow_growth): available_metrics += 1
                if not np.isnan(fear_greed_index): available_metrics += 1
                if not np.isnan(next_month_change): available_metrics += 1
                
                if available_metrics > 0:
                    equal_weight = 1.0 / available_metrics
                    active_weights = {
                        'volatility': equal_weight if not np.isnan(volatility) else 0,
                        'sharpe': equal_weight if not np.isnan(sharpe_ratio) else 0,
                        'pe': equal_weight if not np.isnan(pe_ratio) else 0,
                        'peg': equal_weight if not np.isnan(peg_ratio) else 0,
                        'debt_equity': equal_weight if not np.isnan(debt_equity_ratio) else 0,
                        'cash_flow': equal_weight if not np.isnan(cash_flow_growth) else 0,
                        'fear_greed': equal_weight if not np.isnan(fear_greed_index) else 0,
                        'return': equal_weight if not np.isnan(next_month_change) else 0
                    }
                else:
                    # If still no metrics, use default weights
                    active_weights = {k: v for k, v in self.default_weights.items()}
            
            # Calculate final risk score
            risk_score = (
                active_weights.get('volatility', 0) * volatility_score +
                active_weights.get('sharpe', 0) * sharpe_score +
                active_weights.get('pe', 0) * pe_score +
                active_weights.get('peg', 0) * peg_score +
                active_weights.get('debt_equity', 0) * debt_equity_score +
                active_weights.get('cash_flow', 0) * cash_flow_score +
                active_weights.get('fear_greed', 0) * fear_greed_score +
                active_weights.get('return', 0) * return_score
            )
            
            # Classify risk
            if risk_score < 4:
                risk_category = "Low Risk"
            elif risk_score < 7:
                risk_category = "Medium Risk"
            else:
                risk_category = "High Risk"
            
            # Store all individual risk scores for reporting
            individual_scores = {
                'volatility_score': volatility_score,
                'sharpe_score': sharpe_score,
                'pe_score': pe_score,
                'peg_score': peg_score,
                'debt_equity_score': debt_equity_score,
                'cash_flow_score': cash_flow_score,
                'fear_greed_score': fear_greed_score,
                'return_score': return_score
            }
            
            # Store results
            risk_profile = {
                'ticker': ticker,
                'volatility': volatility,
                'recent_return': recent_return,
                'sharpe_ratio': sharpe_ratio,
                'pe_ratio': pe_ratio,
                'peg_ratio': peg_ratio,
                'debt_equity_ratio': debt_equity_ratio,
                'cash_flow_growth': cash_flow_growth,
                'fear_greed_index': fear_greed_index,
                'expected_return': next_month_change,
                'risk_score': risk_score,
                'risk_category': risk_category,
                'individual_scores': individual_scores,
                'weights_used': active_weights
            }
            
            risk_profiles.append(risk_profile)
            print(f"Risk category for {ticker}: {risk_category} (Score: {risk_score:.2f})")
        
        # Create DataFrame and flatten individual scores for reporting
        risk_df = pd.DataFrame(risk_profiles)
        
        # Extract individual scores and weights if present
        if risk_profiles and 'individual_scores' in risk_profiles[0]:
            for score_name in risk_profiles[0]['individual_scores']:
                risk_df[score_name] = risk_df['individual_scores'].apply(lambda x: x.get(score_name, np.nan))
            risk_df = risk_df.drop('individual_scores', axis=1)
        
        if risk_profiles and 'weights_used' in risk_profiles[0]:
            for weight_name, weight_val in risk_profiles[0]['weights_used'].items():
                risk_df[f'weight_{weight_name}'] = risk_df['weights_used'].apply(lambda x: x.get(weight_name, np.nan))
            risk_df = risk_df.drop('weights_used', axis=1)
        
        return risk_df
    
    def get_market_risk_level(self, sentiment_data):
        """
        Determine overall market risk level based on sentiment.
        
        Parameters:
        -----------
        sentiment_data : dict
            Market sentiment data
            
        Returns:
        --------
        str : Market risk level
        float : Risk score (0-10)
        """
        fear_greed_index = sentiment_data.get('fear_greed_index', 50)
        
        # Calculate risk score based on distance from neutral (50)
        # Both extreme fear and extreme greed indicate high risk
        risk_score = min(abs(fear_greed_index - 50) / 5, 10)
        
        # Determine direction (fear or greed)
        if fear_greed_index < 40:
            direction = "Fearful"
        elif fear_greed_index > 60:
            direction = "Greedy"
        else:
            direction = "Neutral"
        
        # Classify risk level
        if risk_score < 4:
            risk_level = "Low"
        elif risk_score < 7:
            risk_level = "Moderate"
        else:
            risk_level = "High"
        
        market_risk = f"{risk_level} Risk ({direction} Market)"
        
        return market_risk, risk_score
    
    def adjust_weights_based_on_sentiment(self, sentiment_data):
        """
        Adjust risk metric weights based on market sentiment.
        
        Parameters:
        -----------
        sentiment_data : dict
            Market sentiment data
            
        Returns:
        --------
        dict : Adjusted weights
        """
        fear_greed_index = sentiment_data.get('fear_greed_index', 50)
        
        # Start with default weights
        adjusted_weights = self.default_weights.copy()
        
        # In fearful markets (low index), increase importance of cash flow, debt
        if fear_greed_index < 30:  # Extreme Fear or Fear
            adjusted_weights['cash_flow_growth'] = self.default_weights['cash_flow_growth'] * 1.5
            adjusted_weights['debt_equity_ratio'] = self.default_weights['debt_equity_ratio'] * 1.3
            adjusted_weights['volatility'] = self.default_weights['volatility'] * 1.2
            adjusted_weights['expected_return'] = self.default_weights['expected_return'] * 0.7
        
        # In greedy markets (high index), increase importance of valuation metrics
        elif fear_greed_index > 70:  # Extreme Greed or Greed
            adjusted_weights['pe_ratio'] = self.default_weights['pe_ratio'] * 1.5
            adjusted_weights['peg_ratio'] = self.default_weights['peg_ratio'] * 1.3
            adjusted_weights['expected_return'] = self.default_weights['expected_return'] * 1.2
            adjusted_weights['cash_flow_growth'] = self.default_weights['cash_flow_growth'] * 0.8
        
        # Normalize weights to sum to 1
        total_weight = sum(adjusted_weights.values())
        adjusted_weights = {k: v/total_weight for k, v in adjusted_weights.items()}
        
        return adjusted_weights 