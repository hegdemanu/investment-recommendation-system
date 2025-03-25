"""
Module for generating investment recommendations based on
risk profiles, predictions, and user preferences.
"""

import pandas as pd
import numpy as np
from datetime import datetime

class RecommendationEngine:
    """
    Module for generating investment recommendations based on
    risk profiles, predictions, and user preferences.
    """
    
    def __init__(self):
        """Initialize the RecommendationEngine module."""
        self.risk_thresholds = {
            'conservative': {
                'max_risk_score': 4.0,
                'min_sharpe_ratio': 1.0,
                'max_volatility': 15.0,
                'min_expected_return': 5.0
            },
            'moderate': {
                'max_risk_score': 7.0,
                'min_sharpe_ratio': 0.7,
                'max_volatility': 25.0,
                'min_expected_return': 8.0
            },
            'aggressive': {
                'max_risk_score': 10.0,
                'min_sharpe_ratio': 0.5,
                'max_volatility': 40.0,
                'min_expected_return': 12.0
            }
        }
    
    def generate_recommendations(self, risk_profiles, predictions, user_profile='moderate', max_recommendations=5):
        """
        Generate investment recommendations based on risk profiles and predictions.
        
        Parameters:
        -----------
        risk_profiles : pd.DataFrame
            Risk profiles for different investments
        predictions : pd.DataFrame
            Predicted future data
        user_profile : str, optional
            User risk profile: 'conservative', 'moderate', or 'aggressive'
        max_recommendations : int, optional
            Maximum number of recommendations to generate
            
        Returns:
        --------
        pd.DataFrame : Investment recommendations
        """
        print(f"Generating recommendations for {user_profile} investor profile...")
        
        if risk_profiles.empty or predictions.empty:
            print("No risk profiles or predictions available for recommendations.")
            return pd.DataFrame()
        
        # Merge risk profiles with predictions
        merged_data = risk_profiles.merge(predictions, on='ticker', how='inner')
        
        if merged_data.empty:
            print("No matching data between risk profiles and predictions.")
            return pd.DataFrame()
        
        # Get thresholds for user profile
        if user_profile not in self.risk_thresholds:
            print(f"Invalid user profile: {user_profile}. Using 'moderate' as default.")
            user_profile = 'moderate'
        
        thresholds = self.risk_thresholds[user_profile]
        
        # Filter investments based on user profile
        filtered_investments = merged_data.copy()
        
        if user_profile == 'conservative':
            # Conservative investors prioritize lower risk
            filtered_investments = filtered_investments[
                (filtered_investments['risk_score'] <= thresholds['max_risk_score']) &
                (filtered_investments['sharpe_ratio'] >= thresholds['min_sharpe_ratio'] if 'sharpe_ratio' in filtered_investments.columns else True) &
                (filtered_investments['volatility'] <= thresholds['max_volatility'] if 'volatility' in filtered_investments.columns else True)
            ]
            # Sort by risk (ascending) then expected return (descending)
            filtered_investments = filtered_investments.sort_values(
                ['risk_score', 'expected_return'],
                ascending=[True, False]
            )
        
        elif user_profile == 'moderate':
            # Moderate investors want balanced risk/reward
            filtered_investments = filtered_investments[
                (filtered_investments['risk_score'] <= thresholds['max_risk_score']) &
                (filtered_investments['expected_return'] >= thresholds['min_expected_return'] if 'expected_return' in filtered_investments.columns else True)
            ]
            # Create a balanced score (higher is better)
            filtered_investments['balanced_score'] = (
                filtered_investments['expected_return'] / filtered_investments['risk_score']
                if ('expected_return' in filtered_investments.columns and 'risk_score' in filtered_investments.columns)
                else 0
            )
            # Sort by balanced score (descending)
            filtered_investments = filtered_investments.sort_values('balanced_score', ascending=False)
        
        elif user_profile == 'aggressive':
            # Aggressive investors prioritize higher returns
            filtered_investments = filtered_investments[
                (filtered_investments['expected_return'] >= thresholds['min_expected_return'] if 'expected_return' in filtered_investments.columns else True)
            ]
            # Sort by expected return (descending)
            filtered_investments = filtered_investments.sort_values('expected_return', ascending=False)
        
        # Take top recommendations
        top_recommendations = filtered_investments.head(max_recommendations)
        
        if top_recommendations.empty:
            print(f"No suitable investments found for {user_profile} investor profile.")
            # Fall back to top performers without filtering
            top_recommendations = merged_data.sort_values('expected_return', ascending=False).head(max_recommendations)
            print(f"Falling back to top {len(top_recommendations)} performers by expected return.")
        
        # Format recommendations with relevant information
        recommendations = []
        for _, row in top_recommendations.iterrows():
            recommendation = {
                'ticker': row['ticker'],
                'current_price': row['latest_price'],
                'risk_category': row['risk_category'],
                'expected_return': row['expected_return'],
                'short_term_forecast': row.get('next_month_price', row.get('next_quarter_price', None)),
                'long_term_forecast': row.get('next_year_price', row.get('next_half_year_price', None)),
                'recommendation_strength': self._calculate_recommendation_strength(row, user_profile),
                'key_metrics': {
                    'volatility': row.get('volatility', None),
                    'sharpe_ratio': row.get('sharpe_ratio', None),
                    'pe_ratio': row.get('pe_ratio', None),
                    'peg_ratio': row.get('peg_ratio', None)
                },
                'notes': self._generate_recommendation_notes(row, user_profile)
            }
            recommendations.append(recommendation)
        
        print(f"Generated {len(recommendations)} recommendations for {user_profile} investor profile.")
        return pd.DataFrame(recommendations)
    
    def _calculate_recommendation_strength(self, investment_data, user_profile):
        """
        Calculate the strength of a recommendation (1-5 scale).
        
        Parameters:
        -----------
        investment_data : pd.Series
            Data for a single investment
        user_profile : str
            User risk profile
            
        Returns:
        --------
        int : Recommendation strength (1-5)
        """
        strength = 3  # Default to neutral
        
        # Extract relevant metrics
        risk_score = investment_data.get('risk_score', 5)
        expected_return = investment_data.get('expected_return', 0)
        sharpe_ratio = investment_data.get('sharpe_ratio', 0)
        
        # Adjust based on user profile
        if user_profile == 'conservative':
            # Conservative investors value safety
            if risk_score < 3 and expected_return > 5:
                strength = 5  # Strong buy
            elif risk_score < 4 and expected_return > 3:
                strength = 4  # Buy
            elif risk_score > 5:
                strength = 2  # Weak recommendation
            
            # Sharpe ratio is important for conservative investors
            if sharpe_ratio > 1.5:
                strength += 1
            elif sharpe_ratio < 0.5:
                strength -= 1
        
        elif user_profile == 'moderate':
            # Moderate investors value balance
            if 4 <= risk_score <= 6 and expected_return > 8:
                strength = 5  # Strong buy
            elif 3 <= risk_score <= 7 and expected_return > 6:
                strength = 4  # Buy
            elif risk_score > 8 or expected_return < 4:
                strength = 2  # Weak recommendation
            
            # Adjust based on return/risk ratio
            return_risk_ratio = expected_return / risk_score if risk_score > 0 else 0
            if return_risk_ratio > 2:
                strength += 1
            elif return_risk_ratio < 0.5:
                strength -= 1
        
        elif user_profile == 'aggressive':
            # Aggressive investors value high returns
            if expected_return > 15:
                strength = 5  # Strong buy
            elif expected_return > 10:
                strength = 4  # Buy
            elif expected_return < 8:
                strength = 2  # Weak recommendation
            
            # For aggressive investors, high volatility isn't necessarily bad
            # if it comes with high returns
            if expected_return / (risk_score if risk_score > 0 else 1) > 1.5:
                strength += 1
        
        # Cap strength between 1-5
        return max(1, min(5, strength))
    
    def _generate_recommendation_notes(self, investment_data, user_profile):
        """
        Generate explanatory notes for a recommendation.
        
        Parameters:
        -----------
        investment_data : pd.Series
            Data for a single investment
        user_profile : str
            User risk profile
            
        Returns:
        --------
        str : Recommendation notes
        """
        ticker = investment_data['ticker']
        risk_category = investment_data['risk_category']
        expected_return = investment_data.get('expected_return', 0)
        
        notes = []
        
        # Basic information
        notes.append(f"{ticker} is classified as a {risk_category.lower()} investment with an expected return of {expected_return:.2f}%.")
        
        # Additional metrics notes
        pe_ratio = investment_data.get('pe_ratio')
        if pe_ratio is not None and not pd.isna(pe_ratio):
            if pe_ratio < 15:
                notes.append(f"PE ratio of {pe_ratio:.2f} suggests the stock may be undervalued compared to the Indian market average.")
            elif pe_ratio > 25:
                notes.append(f"PE ratio of {pe_ratio:.2f} is relatively high, indicating possible overvaluation.")
        
        peg_ratio = investment_data.get('peg_ratio')
        if peg_ratio is not None and not pd.isna(peg_ratio):
            if peg_ratio < 1:
                notes.append(f"PEG ratio of {peg_ratio:.2f} suggests good value relative to growth.")
            elif peg_ratio > 2:
                notes.append("Growth may be overpriced based on the PEG ratio.")
        
        volatility = investment_data.get('volatility')
        if volatility is not None and not pd.isna(volatility):
            if user_profile == 'conservative' and volatility > 20:
                notes.append(f"Volatility of {volatility:.2f}% is relatively high for a conservative portfolio.")
            elif user_profile == 'aggressive' and volatility < 10:
                notes.append(f"Volatility of {volatility:.2f}% is relatively low for an aggressive strategy.")
        
        sharpe_ratio = investment_data.get('sharpe_ratio')
        if sharpe_ratio is not None and not pd.isna(sharpe_ratio):
            if sharpe_ratio > 1:
                notes.append(f"Sharpe ratio of {sharpe_ratio:.2f} indicates good risk-adjusted returns.")
            elif sharpe_ratio < 0.5:
                notes.append(f"Low Sharpe ratio of {sharpe_ratio:.2f} suggests poor risk-adjusted performance.")
        
        # Profile-specific advice
        if user_profile == 'conservative':
            if risk_category == "Low Risk":
                notes.append("This investment aligns well with your conservative profile.")
            elif risk_category == "High Risk":
                notes.append("Consider limiting exposure to this high-risk investment in your conservative portfolio.")
        
        elif user_profile == 'aggressive':
            if risk_category == "High Risk" and expected_return > 15:
                notes.append("This high-risk, high-return investment matches your aggressive profile.")
            elif risk_category == "Low Risk":
                notes.append("This low-risk investment may underperform your aggressive growth targets.")
        
        return " ".join(notes)
    
    def create_portfolio(self, recommendations, investment_amount=100000, max_securities=5):
        """
        Create a balanced portfolio allocation based on recommendations.
        
        Parameters:
        -----------
        recommendations : pd.DataFrame
            Investment recommendations
        investment_amount : float, optional
            Total amount to invest (in INR)
        max_securities : int, optional
            Maximum number of securities in the portfolio
            
        Returns:
        --------
        pd.DataFrame : Portfolio allocation
        """
        if recommendations.empty:
            print("No recommendations available for portfolio allocation.")
            return pd.DataFrame()
        
        # Take top recommendations up to max_securities
        top_recommendations = recommendations.head(max_securities)
        
        # Allocate weights based on recommendation strength
        if 'recommendation_strength' in top_recommendations.columns:
            weights = top_recommendations['recommendation_strength'] / top_recommendations['recommendation_strength'].sum()
        else:
            # Equal weights if no strength information
            weights = [1/len(top_recommendations)] * len(top_recommendations)
        
        # Calculate allocation amounts
        allocation_amounts = [w * investment_amount for w in weights]
        
        # Create portfolio dataframe
        portfolio = []
        for i, (_, row) in enumerate(top_recommendations.iterrows()):
            # Calculate number of shares to buy (rounded down)
            shares = int(allocation_amounts[i] / row['current_price'])
            actual_amount = shares * row['current_price']
            
            allocation = {
                'ticker': row['ticker'],
                'allocation_percentage': weights[i] * 100,
                'allocation_amount': actual_amount,
                'shares': shares,
                'price_per_share': row['current_price'],
                'expected_return': row['expected_return'],
                'risk_category': row['risk_category']
            }
            portfolio.append(allocation)
        
        # Calculate actual allocation percentages based on rounded shares
        portfolio_df = pd.DataFrame(portfolio)
        total_allocated = portfolio_df['allocation_amount'].sum()
        portfolio_df['actual_allocation_percentage'] = (portfolio_df['allocation_amount'] / total_allocated) * 100
        
        # Calculate unallocated amount due to rounding
        portfolio_df['unallocated_amount'] = investment_amount - total_allocated
        
        print(f"Created portfolio allocation for {len(portfolio)} investments.")
        print(f"Total allocated: ₹{total_allocated:.2f}, Unallocated: ₹{investment_amount - total_allocated:.2f}")
        
        return portfolio_df
    
    def generate_report(self, portfolio, risk_profiles, predictions):
        """
        Generate a comprehensive investment report.
        
        Parameters:
        -----------
        portfolio : pd.DataFrame
            Portfolio allocation
        risk_profiles : pd.DataFrame
            Risk profiles for different investments
        predictions : pd.DataFrame
            Predicted future data
            
        Returns:
        --------
        dict : Investment report
        """
        if portfolio.empty:
            print("No portfolio data available for report generation.")
            return {}
        
        # Current date for the report
        report_date = datetime.now().strftime("%Y-%m-%d")
        
        # Calculate portfolio metrics
        total_invested = portfolio['allocation_amount'].sum()
        tickers = portfolio['ticker'].tolist()
        
        # Calculate weighted expected return
        weighted_return = (portfolio['expected_return'] * portfolio['actual_allocation_percentage'] / 100).sum()
        
        # Calculate weighted risk score
        ticker_risk_scores = {row['ticker']: row['risk_score'] for _, row in risk_profiles.iterrows()}
        portfolio['risk_score'] = portfolio['ticker'].map(ticker_risk_scores)
        weighted_risk = (portfolio['risk_score'] * portfolio['actual_allocation_percentage'] / 100).sum() if 'risk_score' in portfolio.columns else None
        
        # Categorize portfolio risk
        if weighted_risk is not None:
            if weighted_risk < 4:
                portfolio_risk = "Low Risk"
            elif weighted_risk < 7:
                portfolio_risk = "Medium Risk"
            else:
                portfolio_risk = "High Risk"
        else:
            portfolio_risk = "Undefined"
        
        # Get short and long term forecasts
        forecast_data = {}
        for ticker in tickers:
            ticker_pred = predictions[predictions['ticker'] == ticker]
            if not ticker_pred.empty:
                forecast_data[ticker] = {
                    'short_term': ticker_pred['next_month_price'].iloc[0] if 'next_month_price' in ticker_pred.columns else None,
                    'long_term': ticker_pred['next_year_price'].iloc[0] if 'next_year_price' in ticker_pred.columns else None
                }
        
        # Calculate potential portfolio value
        short_term_value = 0
        long_term_value = 0
        
        for _, row in portfolio.iterrows():
            ticker = row['ticker']
            shares = row['shares']
            current_value = row['allocation_amount']
            
            if ticker in forecast_data:
                short_term_price = forecast_data[ticker]['short_term']
                long_term_price = forecast_data[ticker]['long_term']
                
                if short_term_price is not None:
                    short_term_value += shares * short_term_price
                else:
                    short_term_value += current_value
                
                if long_term_price is not None:
                    long_term_value += shares * long_term_price
                else:
                    long_term_value += current_value
            else:
                short_term_value += current_value
                long_term_value += current_value
        
        # Calculate potential returns
        short_term_return = ((short_term_value / total_invested) - 1) * 100
        long_term_return = ((long_term_value / total_invested) - 1) * 100
        
        # Generate report
        report = {
            'report_date': report_date,
            'portfolio_summary': {
                'total_invested': total_invested,
                'number_of_securities': len(portfolio),
                'tickers': tickers,
                'weighted_expected_return': weighted_return,
                'weighted_risk_score': weighted_risk,
                'portfolio_risk_category': portfolio_risk
            },
            'potential_returns': {
                'short_term_value': short_term_value,
                'short_term_return': short_term_return,
                'long_term_value': long_term_value,
                'long_term_return': long_term_return
            },
            'allocations': portfolio.to_dict(orient='records'),
            'market_outlook': self._generate_market_outlook(predictions),
            'portfolio_recommendations': self._generate_portfolio_recommendations(portfolio, risk_profiles, predictions)
        }
        
        print(f"Generated investment report dated {report_date}")
        print(f"Portfolio risk category: {portfolio_risk}, Expected return: {weighted_return:.2f}%")
        
        return report
    
    def _generate_market_outlook(self, predictions):
        """
        Generate market outlook based on predictions.
        
        Parameters:
        -----------
        predictions : pd.DataFrame
            Predicted future data
            
        Returns:
        --------
        str : Market outlook
        """
        if predictions.empty:
            return "Insufficient data for market outlook."
        
        # Calculate average expected returns across different time horizons
        time_horizons = {}
        for col in predictions.columns:
            if col.endswith('_change'):
                horizon = col.replace('_change', '')
                time_horizons[horizon] = predictions[col].mean()
        
        # Sort horizons by time period
        horizon_order = ['next_day', 'next_week', 'next_month', 'next_quarter', 'next_half_year', 'next_year']
        sorted_horizons = sorted(
            time_horizons.items(),
            key=lambda x: horizon_order.index(x[0]) if x[0] in horizon_order else 999
        )
        
        # Generate outlook text
        outlook_parts = []
        for horizon, avg_change in sorted_horizons:
            friendly_name = horizon.replace('next_', '').replace('_', ' ').capitalize()
            outlook_parts.append(f"{friendly_name}: {avg_change:.2f}%")
        
        # Overall sentiment
        short_term = time_horizons.get('next_month', time_horizons.get('next_week', 0))
        long_term = time_horizons.get('next_year', time_horizons.get('next_half_year', 0))
        
        if short_term > 0 and long_term > 0:
            sentiment = "The market outlook is positive across all time horizons."
        elif short_term <= 0 and long_term > 0:
            sentiment = "The market may face short-term challenges but shows long-term promise."
        elif short_term > 0 and long_term <= 0:
            sentiment = "The market appears favorable in the short term but may face headwinds over longer periods."
        else:
            sentiment = "The market outlook indicates caution is warranted across all time horizons."
        
        # Combine all parts
        outlook = sentiment + " Average expected returns: " + ", ".join(outlook_parts)
        
        return outlook
    
    def _generate_portfolio_recommendations(self, portfolio, risk_profiles, predictions):
        """
        Generate recommendations for portfolio adjustments.
        
        Parameters:
        -----------
        portfolio : pd.DataFrame
            Portfolio allocation
        risk_profiles : pd.DataFrame
            Risk profiles for different investments
        predictions : pd.DataFrame
            Predicted future data
            
        Returns:
        --------
        list : Recommendations for portfolio adjustments
        """
        recommendations = []
        
        if portfolio.empty:
            recommendations.append("No portfolio data available for recommendations.")
            return recommendations
        
        # Check portfolio diversification
        if len(portfolio) < 3:
            recommendations.append("Consider adding more securities to increase diversification.")
        
        # Check if any single security dominates the portfolio
        for _, row in portfolio.iterrows():
            if row['actual_allocation_percentage'] > 40:
                recommendations.append(f"{row['ticker']} comprises {row['actual_allocation_percentage']:.1f}% of the portfolio. Consider reducing allocation to reduce concentration risk.")
        
        # Check for potentially underperforming securities
        for _, row in portfolio.iterrows():
            ticker_pred = predictions[predictions['ticker'] == row['ticker']]
            if not ticker_pred.empty and 'next_month_change' in ticker_pred.columns:
                expected_change = ticker_pred['next_month_change'].iloc[0]
                if expected_change < 0:
                    recommendations.append(f"{row['ticker']} has a negative short-term outlook ({expected_change:.2f}%). Consider monitoring closely or reducing exposure.")
        
        # Check for securities with poor risk-reward profiles
        for _, row in portfolio.iterrows():
            ticker_risk = risk_profiles[risk_profiles['ticker'] == row['ticker']]
            if not ticker_risk.empty:
                risk_score = ticker_risk['risk_score'].iloc[0]
                expected_return = ticker_risk['expected_return'].iloc[0] if 'expected_return' in ticker_risk.columns else None
                
                if expected_return is not None and risk_score > 0:
                    risk_reward_ratio = expected_return / risk_score
                    if risk_reward_ratio < 0.5:
                        recommendations.append(f"{row['ticker']} has a poor risk-reward profile. Consider alternatives with better return potential for the risk taken.")
        
        # If no specific recommendations, add a general one
        if not recommendations:
            recommendations.append("Portfolio appears well-balanced. Regular monitoring and periodic rebalancing recommended.")
        
        return recommendations 