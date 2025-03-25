"""
Screener Module for Investment Recommendation System
Provides filtering functionality based on various financial metrics and market indicators
"""

import pandas as pd
import numpy as np
from datetime import datetime


class InvestmentScreener:
    """
    Screens investments based on customizable criteria including
    fundamental metrics, technical indicators, and market sentiment.
    """
    
    def __init__(self):
        """Initialize the InvestmentScreener module."""
        # Default screening criteria
        self.default_criteria = {
            # Fundamental metrics
            'min_pe_ratio': 0,
            'max_pe_ratio': 50,
            'min_peg_ratio': 0,
            'max_peg_ratio': 3,
            'max_debt_equity': 2.0,
            'min_cash_flow_growth': 0,
            'min_sharpe_ratio': 0.5,
            
            # Market sentiment
            'fear_greed_index_range': (25, 75),  # Avoid extreme sentiment
            
            # Technical indicators
            'min_rsi': 30,
            'max_rsi': 70,
            
            # Performance metrics
            'min_return': 0,
            'max_volatility': 40
        }
    
    def screen_investments(self, data, criteria=None):
        """
        Screen investments based on the provided criteria.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Data containing investment metrics
        criteria : dict, optional
            Custom screening criteria
            
        Returns:
        --------
        pd.DataFrame : Screened investments
        """
        if criteria is None:
            criteria = self.default_criteria
        
        # Start with all investments
        screened_data = data.copy()
        initial_count = len(screened_data)
        
        # Track metrics used and investments filtered
        screening_metrics = []
        filtered_counts = {}
        
        print(f"Starting screening with {initial_count} investments...")
        
        # Apply fundamental metric filters
        if 'pe_ratio' in screened_data.columns:
            prev_count = len(screened_data)
            screened_data = screened_data[
                (screened_data['pe_ratio'] >= criteria.get('min_pe_ratio', self.default_criteria['min_pe_ratio'])) & 
                (screened_data['pe_ratio'] <= criteria.get('max_pe_ratio', self.default_criteria['max_pe_ratio']))
            ]
            filtered_counts['PE Ratio'] = prev_count - len(screened_data)
            screening_metrics.append('PE Ratio')
        
        if 'peg_ratio' in screened_data.columns:
            prev_count = len(screened_data)
            screened_data = screened_data[
                (screened_data['peg_ratio'] >= criteria.get('min_peg_ratio', self.default_criteria['min_peg_ratio'])) & 
                (screened_data['peg_ratio'] <= criteria.get('max_peg_ratio', self.default_criteria['max_peg_ratio']))
            ]
            filtered_counts['PEG Ratio'] = prev_count - len(screened_data)
            screening_metrics.append('PEG Ratio')
        
        if 'debt_equity_ratio' in screened_data.columns:
            prev_count = len(screened_data)
            screened_data = screened_data[
                screened_data['debt_equity_ratio'] <= criteria.get('max_debt_equity', self.default_criteria['max_debt_equity'])
            ]
            filtered_counts['Debt-to-Equity'] = prev_count - len(screened_data)
            screening_metrics.append('Debt-to-Equity')
        
        if 'cash_flow_growth' in screened_data.columns:
            prev_count = len(screened_data)
            screened_data = screened_data[
                screened_data['cash_flow_growth'] >= criteria.get('min_cash_flow_growth', self.default_criteria['min_cash_flow_growth'])
            ]
            filtered_counts['Cash Flow Growth'] = prev_count - len(screened_data)
            screening_metrics.append('Cash Flow Growth')
        
        if 'sharpe_ratio' in screened_data.columns:
            prev_count = len(screened_data)
            screened_data = screened_data[
                screened_data['sharpe_ratio'] >= criteria.get('min_sharpe_ratio', self.default_criteria['min_sharpe_ratio'])
            ]
            filtered_counts['Sharpe Ratio'] = prev_count - len(screened_data)
            screening_metrics.append('Sharpe Ratio')
        
        # Apply market sentiment filters
        if 'fear_greed_index' in screened_data.columns:
            prev_count = len(screened_data)
            fear_greed_range = criteria.get('fear_greed_index_range', self.default_criteria['fear_greed_index_range'])
            screened_data = screened_data[
                (screened_data['fear_greed_index'] >= fear_greed_range[0]) & 
                (screened_data['fear_greed_index'] <= fear_greed_range[1])
            ]
            filtered_counts['Fear/Greed Index'] = prev_count - len(screened_data)
            screening_metrics.append('Fear/Greed Index')
        
        # Apply technical indicator filters
        if 'rsi' in screened_data.columns:
            prev_count = len(screened_data)
            screened_data = screened_data[
                (screened_data['rsi'] >= criteria.get('min_rsi', self.default_criteria['min_rsi'])) & 
                (screened_data['rsi'] <= criteria.get('max_rsi', self.default_criteria['max_rsi']))
            ]
            filtered_counts['RSI'] = prev_count - len(screened_data)
            screening_metrics.append('RSI')
        
        # Apply performance metric filters
        if 'expected_return' in screened_data.columns:
            prev_count = len(screened_data)
            screened_data = screened_data[
                screened_data['expected_return'] >= criteria.get('min_return', self.default_criteria['min_return'])
            ]
            filtered_counts['Expected Return'] = prev_count - len(screened_data)
            screening_metrics.append('Expected Return')
        
        if 'volatility' in screened_data.columns:
            prev_count = len(screened_data)
            screened_data = screened_data[
                screened_data['volatility'] <= criteria.get('max_volatility', self.default_criteria['max_volatility'])
            ]
            filtered_counts['Volatility'] = prev_count - len(screened_data)
            screening_metrics.append('Volatility')
        
        # Generate screening summary
        screening_summary = {
            'initial_count': initial_count,
            'final_count': len(screened_data),
            'metrics_used': screening_metrics,
            'filtered_counts': filtered_counts,
            'screening_date': datetime.now().strftime("%Y-%m-%d"),
            'criteria_used': criteria
        }
        
        if len(screened_data) == 0:
            print("No investments passed all screening criteria. Consider relaxing your filters.")
        else:
            print(f"Screening complete: {len(screened_data)} investments passed all criteria.")
            for metric, count in filtered_counts.items():
                if count > 0:
                    print(f"  - {metric} filter removed {count} investments")
        
        return screened_data, screening_summary
    
    def generate_screener_presets(self):
        """
        Generate common screening presets for different investment strategies.
        
        Returns:
        --------
        dict : Dictionary of screening presets
        """
        presets = {
            'value_investing': {
                'max_pe_ratio': 15,
                'max_peg_ratio': 1.0,
                'max_debt_equity': 1.0,
                'min_cash_flow_growth': 5.0,
                'min_sharpe_ratio': 0.8
            },
            'growth_investing': {
                'min_cash_flow_growth': 10.0,
                'min_return': 15.0,
                'fear_greed_index_range': (40, 80)  # Slightly bullish bias
            },
            'income_investing': {
                'min_dividend_yield': 3.0,
                'max_debt_equity': 1.5,
                'max_volatility': 20.0,
                'fear_greed_index_range': (30, 70)  # Balanced sentiment
            },
            'momentum_investing': {
                'min_rsi': 50,  # Look for stocks with upward momentum
                'min_return': 10.0,
                'fear_greed_index_range': (45, 85)  # Bullish bias
            },
            'conservative': {
                'max_pe_ratio': 20,
                'max_debt_equity': 1.0,
                'max_volatility': 15.0,
                'min_sharpe_ratio': 1.0,
                'fear_greed_index_range': (30, 60)  # Slightly cautious
            }
        }
        
        return presets
    
    def screen_by_strategy(self, data, strategy):
        """
        Screen investments based on a predefined strategy preset.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Data containing investment metrics
        strategy : str
            Strategy name from the predefined presets
            
        Returns:
        --------
        tuple : (screened_data, screening_summary)
        """
        presets = self.generate_screener_presets()
        
        if strategy not in presets:
            available_strategies = list(presets.keys())
            print(f"Strategy '{strategy}' not found. Available strategies: {available_strategies}")
            return None, None
        
        criteria = presets[strategy]
        print(f"Applying '{strategy}' screening strategy with criteria: {criteria}")
        
        return self.screen_investments(data, criteria) 