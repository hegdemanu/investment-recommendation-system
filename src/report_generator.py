"""
Module for generating detailed investment reports in various formats.
Supports PDF and Excel report generation with visualizations.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
from fpdf import FPDF

class ReportGenerator:
    """
    Module for generating comprehensive investment reports in
    various formats including Excel, PDF, and JSON.
    """
    
    def __init__(self, reports_dir="./reports"):
        """
        Initialize the ReportGenerator module.
        
        Parameters:
        -----------
        reports_dir : str, optional
            Directory to save generated reports
        """
        self.reports_dir = reports_dir
        if not os.path.exists(reports_dir):
            os.makedirs(reports_dir)
            print(f"Created reports directory: {reports_dir}")
        
        # Configure plot style
        plt.style.use('ggplot')
        sns.set_palette("Set2")
    
    def generate_technical_report(self, data, risk_profiles, predictions, 
                                 portfolio=None, sentiment_data=None, 
                                 format='excel'):
        """
        Generate a comprehensive technical report on investments.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Historical price data
        risk_profiles : pd.DataFrame
            Risk classifications
        predictions : pd.DataFrame
            Predicted future data
        portfolio : pd.DataFrame, optional
            Portfolio allocation
        sentiment_data : dict, optional
            Market sentiment data
        format : str, optional
            Report format: 'excel', 'pdf', or 'json'
            
        Returns:
        --------
        str : Path to the generated report
        """
        report_date = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if format.lower() == 'excel':
            return self._generate_excel_report(data, risk_profiles, predictions, 
                                              portfolio, sentiment_data, report_date)
        elif format.lower() == 'pdf':
            return self._generate_pdf_report(data, risk_profiles, predictions, 
                                            portfolio, sentiment_data, report_date)
        elif format.lower() == 'json':
            return self._generate_json_report(data, risk_profiles, predictions, 
                                             portfolio, sentiment_data, report_date)
        else:
            print(f"Unsupported format: {format}. Using Excel instead.")
            return self._generate_excel_report(data, risk_profiles, predictions, 
                                              portfolio, sentiment_data, report_date)
    
    def _generate_excel_report(self, data, risk_profiles, predictions, 
                              portfolio, sentiment_data, report_date):
        """
        Generate an Excel report.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Historical price data
        risk_profiles : pd.DataFrame
            Risk classifications
        predictions : pd.DataFrame
            Predicted future data
        portfolio : pd.DataFrame, optional
            Portfolio allocation
        sentiment_data : dict, optional
            Market sentiment data
        report_date : str
            Report generation date
            
        Returns:
        --------
        str : Path to the generated Excel report
        """
        print("Generating Excel technical report...")
        
        # Create Excel writer
        report_path = os.path.join(self.reports_dir, f"investment_report_{report_date}.xlsx")
        with pd.ExcelWriter(report_path, engine='openpyxl') as writer:
            
            # Summary sheet
            summary_data = self._prepare_summary_data(risk_profiles, predictions, portfolio, sentiment_data)
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            # Risk profiles sheet
            if not risk_profiles.empty:
                # Select relevant columns and rename for better readability
                risk_cols = [
                    'ticker', 'risk_category', 'risk_score', 'volatility', 
                    'sharpe_ratio', 'pe_ratio', 'peg_ratio', 'debt_equity_ratio',
                    'cash_flow_growth', 'fear_greed_index', 'expected_return'
                ]
                
                # Use only available columns
                available_cols = [col for col in risk_cols if col in risk_profiles.columns]
                
                risk_df = risk_profiles[available_cols].copy()
                
                # Add individual score columns if available
                score_cols = [col for col in risk_profiles.columns if col.endswith('_score')]
                if score_cols:
                    risk_df = pd.concat([risk_df, risk_profiles[score_cols]], axis=1)
                
                # Format percentages and ratios
                for col in ['volatility', 'expected_return', 'cash_flow_growth']:
                    if col in risk_df.columns:
                        risk_df[col] = risk_df[col].apply(lambda x: f"{x:.2f}%" if pd.notnull(x) else "N/A")
                
                for col in ['sharpe_ratio', 'pe_ratio', 'peg_ratio', 'debt_equity_ratio']:
                    if col in risk_df.columns:
                        risk_df[col] = risk_df[col].apply(lambda x: f"{x:.2f}" if pd.notnull(x) else "N/A")
                
                risk_df.to_excel(writer, sheet_name='Risk Analysis', index=False)
            
            # Predictions sheet
            if not predictions.empty:
                # Format the predictions data
                pred_df = predictions.copy()
                
                # Format price and percentage columns
                for col in pred_df.columns:
                    if 'price' in col.lower():
                        pred_df[col] = pred_df[col].apply(lambda x: f"₹{x:.2f}" if pd.notnull(x) else "N/A")
                    elif 'change' in col.lower():
                        pred_df[col] = pred_df[col].apply(lambda x: f"{x:.2f}%" if pd.notnull(x) else "N/A")
                
                pred_df.to_excel(writer, sheet_name='Predictions', index=False)
            
            # Portfolio sheet
            if portfolio is not None and not portfolio.empty:
                # Format the portfolio data
                port_df = portfolio.copy()
                
                # Format monetary and percentage columns
                for col in port_df.columns:
                    if 'amount' in col.lower() or 'price' in col.lower():
                        port_df[col] = port_df[col].apply(lambda x: f"₹{x:.2f}" if pd.notnull(x) else "N/A")
                    elif 'percentage' in col.lower() or 'return' in col.lower():
                        port_df[col] = port_df[col].apply(lambda x: f"{x:.2f}%" if pd.notnull(x) else "N/A")
                
                port_df.to_excel(writer, sheet_name='Portfolio', index=False)
            
            # Market Sentiment sheet
            if sentiment_data is not None:
                sentiment_df = pd.DataFrame([sentiment_data])
                sentiment_df.to_excel(writer, sheet_name='Market Sentiment', index=False)
            
            # Historical Data sheet (sample)
            if not data.empty:
                # Use a sample of historical data to keep file size manageable
                if 'ticker' in data.columns:
                    # For each ticker, get the last 30 days
                    tickers = data['ticker'].unique()
                    sample_data = []
                    
                    for ticker in tickers:
                        ticker_data = data[data['ticker'] == ticker].sort_values('Date').tail(30)
                        sample_data.append(ticker_data)
                    
                    historical_sample = pd.concat(sample_data, ignore_index=True)
                else:
                    # If no ticker column, just take the last 30 days
                    historical_sample = data.sort_values('Date').tail(30)
                
                historical_sample.to_excel(writer, sheet_name='Historical Data (Sample)', index=False)
        
        print(f"Excel report generated: {report_path}")
        return report_path
    
    def _generate_pdf_report(self, data, risk_profiles, predictions, 
                            portfolio, sentiment_data, report_date):
        """
        Generate a PDF report.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Historical price data
        risk_profiles : pd.DataFrame
            Risk classifications
        predictions : pd.DataFrame
            Predicted future data
        portfolio : pd.DataFrame, optional
            Portfolio allocation
        sentiment_data : dict, optional
            Market sentiment data
        report_date : str
            Report generation date
            
        Returns:
        --------
        str : Path to the generated PDF report
        """
        print("Generating PDF technical report...")
        
        # Generate visualizations first
        viz_path = self._generate_visualizations(data, risk_profiles, predictions, portfolio, sentiment_data, report_date)
        
        # Create PDF
        report_path = os.path.join(self.reports_dir, f"investment_report_{report_date}.pdf")
        
        pdf = FPDF()
        pdf.add_page()
        
        # Set font
        pdf.set_font("Arial", "B", 16)
        
        # Title
        pdf.cell(0, 10, "Investment Recommendation System - Technical Report", 0, 1, "C")
        pdf.set_font("Arial", "", 12)
        pdf.cell(0, 10, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 0, 1, "C")
        pdf.ln(10)
        
        # Market Sentiment Section
        if sentiment_data is not None:
            pdf.set_font("Arial", "B", 14)
            pdf.cell(0, 10, "Market Sentiment Analysis", 0, 1, "L")
            pdf.set_font("Arial", "", 12)
            
            fear_greed = sentiment_data.get('fear_greed_index', 'N/A')
            classification = sentiment_data.get('fear_greed_classification', 'N/A')
            
            pdf.cell(0, 8, f"Current Fear & Greed Index: {fear_greed} ({classification})", 0, 1, "L")
            pdf.cell(0, 8, f"Previous Close: {sentiment_data.get('previous_close', 'N/A')}", 0, 1, "L")
            pdf.cell(0, 8, f"One Week Ago: {sentiment_data.get('one_week_ago', 'N/A')}", 0, 1, "L")
            pdf.cell(0, 8, f"One Month Ago: {sentiment_data.get('one_month_ago', 'N/A')}", 0, 1, "L")
            
            pdf.ln(5)
            
            # Add interpretation of market sentiment
            if fear_greed < 30:
                sentiment_interpretation = "Market sentiment is fearful. Consider defensive positions and value investments."
            elif fear_greed > 70:
                sentiment_interpretation = "Market sentiment is greedy. Be cautious of potential overvaluation in growth stocks."
            else:
                sentiment_interpretation = "Market sentiment is neutral. Balanced investment approach recommended."
            
            pdf.set_font("Arial", "I", 12)
            pdf.multi_cell(0, 8, f"Interpretation: {sentiment_interpretation}")
            pdf.ln(10)
        
        # Risk Analysis Section
        if not risk_profiles.empty:
            pdf.set_font("Arial", "B", 14)
            pdf.cell(0, 10, "Risk Analysis Summary", 0, 1, "L")
            pdf.set_font("Arial", "", 12)
            
            # Count risk categories
            risk_counts = risk_profiles['risk_category'].value_counts()
            
            pdf.cell(0, 8, f"Low Risk Investments: {risk_counts.get('Low Risk', 0)}", 0, 1, "L")
            pdf.cell(0, 8, f"Medium Risk Investments: {risk_counts.get('Medium Risk', 0)}", 0, 1, "L")
            pdf.cell(0, 8, f"High Risk Investments: {risk_counts.get('High Risk', 0)}", 0, 1, "L")
            
            # Top 5 investments by risk score (lowest risk)
            pdf.ln(5)
            pdf.set_font("Arial", "B", 12)
            pdf.cell(0, 8, "Top 5 Investments (Lowest Risk):", 0, 1, "L")
            pdf.set_font("Arial", "", 10)
            
            top_safe = risk_profiles.sort_values('risk_score').head(5)
            for _, row in top_safe.iterrows():
                ticker = row['ticker']
                risk_score = row['risk_score']
                expected_return = row.get('expected_return', 'N/A')
                if expected_return != 'N/A':
                    expected_return = f"{expected_return:.2f}%"
                
                pdf.cell(0, 6, f"{ticker}: Risk Score {risk_score:.2f}, Expected Return {expected_return}", 0, 1, "L")
            
            pdf.ln(10)
        
        # Predictions Section
        if not predictions.empty:
            pdf.set_font("Arial", "B", 14)
            pdf.cell(0, 10, "Price Predictions Summary", 0, 1, "L")
            pdf.set_font("Arial", "", 12)
            
            # Top 5 investments by expected return
            pdf.set_font("Arial", "B", 12)
            pdf.cell(0, 8, "Top 5 Investments (Highest Expected Return):", 0, 1, "L")
            pdf.set_font("Arial", "", 10)
            
            if 'next_month_change' in predictions.columns:
                top_return = predictions.sort_values('next_month_change', ascending=False).head(5)
                for _, row in top_return.iterrows():
                    ticker = row['ticker']
                    current = row.get('latest_price', 'N/A')
                    if current != 'N/A':
                        current = f"₹{current:.2f}"
                    
                    future = row.get('next_month_price', row.get('next_quarter_price', 'N/A'))
                    if future != 'N/A':
                        future = f"₹{future:.2f}"
                    
                    change = row.get('next_month_change', row.get('next_quarter_change', 'N/A'))
                    if change != 'N/A':
                        change = f"{change:.2f}%"
                    
                    pdf.cell(0, 6, f"{ticker}: Current {current}, Forecast {future} ({change})", 0, 1, "L")
            
            pdf.ln(10)
        
        # Portfolio Section
        if portfolio is not None and not portfolio.empty:
            pdf.set_font("Arial", "B", 14)
            pdf.cell(0, 10, "Portfolio Summary", 0, 1, "L")
            pdf.set_font("Arial", "", 12)
            
            # Portfolio statistics
            total_invested = portfolio['allocation_amount'].sum()
            num_securities = len(portfolio)
            
            pdf.cell(0, 8, f"Total Invested: ₹{total_invested:.2f}", 0, 1, "L")
            pdf.cell(0, 8, f"Number of Securities: {num_securities}", 0, 1, "L")
            
            # Calculate weighted expected return if available
            if 'expected_return' in portfolio.columns and 'actual_allocation_percentage' in portfolio.columns:
                weighted_return = (portfolio['expected_return'] * portfolio['actual_allocation_percentage'] / 100).sum()
                pdf.cell(0, 8, f"Weighted Expected Return: {weighted_return:.2f}%", 0, 1, "L")
            
            pdf.ln(10)
        
        # Add visualizations
        if viz_path and os.path.exists(viz_path):
            pdf.set_font("Arial", "B", 14)
            pdf.cell(0, 10, "Visualizations", 0, 1, "L")
            pdf.image(viz_path, x=10, w=180)
        
        # Save PDF
        pdf.output(report_path)
        
        print(f"PDF report generated: {report_path}")
        return report_path
    
    def _generate_json_report(self, data, risk_profiles, predictions, 
                             portfolio, sentiment_data, report_date):
        """
        Generate a JSON report.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Historical price data
        risk_profiles : pd.DataFrame
            Risk classifications
        predictions : pd.DataFrame
            Predicted future data
        portfolio : pd.DataFrame, optional
            Portfolio allocation
        sentiment_data : dict, optional
            Market sentiment data
        report_date : str
            Report generation date
            
        Returns:
        --------
        str : Path to the generated JSON report
        """
        print("Generating JSON technical report...")
        
        report = {
            "report_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "summary": self._prepare_summary_data(risk_profiles, predictions, portfolio, sentiment_data)
        }
        
        # Risk profiles
        if not risk_profiles.empty:
            report["risk_profiles"] = risk_profiles.to_dict(orient='records')
        
        # Predictions
        if not predictions.empty:
            report["predictions"] = predictions.to_dict(orient='records')
        
        # Portfolio
        if portfolio is not None and not portfolio.empty:
            report["portfolio"] = portfolio.to_dict(orient='records')
        
        # Market sentiment
        if sentiment_data is not None:
            report["market_sentiment"] = sentiment_data
        
        # Write to file
        report_path = os.path.join(self.reports_dir, f"investment_report_{report_date}.json")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=self._json_serializer)
        
        print(f"JSON report generated: {report_path}")
        return report_path
    
    def _json_serializer(self, obj):
        """
        JSON serializer for objects not serializable by default json code.
        
        Parameters:
        -----------
        obj : object
            Object to serialize
            
        Returns:
        --------
        str : Serialized representation
        """
        if isinstance(obj, (datetime, np.datetime64)):
            return obj.strftime("%Y-%m-%d %H:%M:%S")
        elif isinstance(obj, np.int64):
            return int(obj)
        elif isinstance(obj, np.float64):
            return float(obj)
        elif pd.isna(obj):
            return None
        return str(obj)
    
    def _prepare_summary_data(self, risk_profiles, predictions, portfolio, sentiment_data):
        """
        Prepare summary data for reports.
        
        Parameters:
        -----------
        risk_profiles : pd.DataFrame
            Risk classifications
        predictions : pd.DataFrame
            Predicted future data
        portfolio : pd.DataFrame, optional
            Portfolio allocation
        sentiment_data : dict, optional
            Market sentiment data
            
        Returns:
        --------
        list : Summary data
        """
        summary = []
        
        # Report generation time
        summary.append({
            "type": "header",
            "content": f"Investment Report - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        })
        
        # Market sentiment
        if sentiment_data is not None:
            fear_greed = sentiment_data.get('fear_greed_index', 'N/A')
            classification = sentiment_data.get('fear_greed_classification', 'N/A')
            
            summary.append({
                "type": "market_sentiment",
                "content": f"Market Sentiment: Fear & Greed Index is {fear_greed} ({classification})"
            })
            
            # Add sentiment interpretation
            if fear_greed < 30:
                interpretation = "Market sentiment is fearful. This may be a good time for value investing and accumulating quality assets at lower prices."
            elif fear_greed > 70:
                interpretation = "Market sentiment is greedy. Be cautious of potential overvaluation and consider reducing exposure to high-risk assets."
            else:
                interpretation = "Market sentiment is balanced. A diversified approach is recommended."
            
            summary.append({
                "type": "interpretation",
                "content": interpretation
            })
        
        # Risk distribution
        if not risk_profiles.empty:
            risk_counts = risk_profiles['risk_category'].value_counts()
            
            risk_distribution = f"Risk Distribution: "
            risk_distribution += f"Low Risk: {risk_counts.get('Low Risk', 0)}, "
            risk_distribution += f"Medium Risk: {risk_counts.get('Medium Risk', 0)}, "
            risk_distribution += f"High Risk: {risk_counts.get('High Risk', 0)}"
            
            summary.append({
                "type": "risk_distribution",
                "content": risk_distribution
            })
            
            # Top performers by different metrics
            if not predictions.empty and 'expected_return' in predictions.columns:
                top_return = predictions.sort_values('expected_return', ascending=False).head(3)
                top_tickers = [row['ticker'] for _, row in top_return.iterrows()]
                top_returns = [f"{row['expected_return']:.2f}%" for _, row in top_return.iterrows()]
                
                summary.append({
                    "type": "top_performers",
                    "content": f"Top Expected Returns: {', '.join([f'{t} ({r})' for t, r in zip(top_tickers, top_returns)])}"
                })
            
            # Investments with strong fundamentals
            if 'debt_equity_ratio' in risk_profiles.columns and 'cash_flow_growth' in risk_profiles.columns:
                # Find investments with low debt and positive cash flow growth
                strong_fundamentals = risk_profiles[
                    (risk_profiles['debt_equity_ratio'] < 1) & 
                    (risk_profiles['cash_flow_growth'] > 0)
                ].sort_values('risk_score').head(3)
                
                if not strong_fundamentals.empty:
                    fund_tickers = [row['ticker'] for _, row in strong_fundamentals.iterrows()]
                    
                    summary.append({
                        "type": "strong_fundamentals",
                        "content": f"Strong Fundamentals: {', '.join(fund_tickers)}"
                    })
            
            # Value investments (low PE, low PEG)
            if 'pe_ratio' in risk_profiles.columns and 'peg_ratio' in risk_profiles.columns:
                value_investments = risk_profiles[
                    (risk_profiles['pe_ratio'] < 15) & 
                    (risk_profiles['peg_ratio'] < 1)
                ].sort_values('pe_ratio').head(3)
                
                if not value_investments.empty:
                    value_tickers = [row['ticker'] for _, row in value_investments.iterrows()]
                    
                    summary.append({
                        "type": "value_investments",
                        "content": f"Value Investments: {', '.join(value_tickers)}"
                    })
        
        # Portfolio summary
        if portfolio is not None and not portfolio.empty:
            total_invested = portfolio['allocation_amount'].sum()
            
            # Calculate weighted expected return if available
            if 'expected_return' in portfolio.columns and 'actual_allocation_percentage' in portfolio.columns:
                weighted_return = (portfolio['expected_return'] * portfolio['actual_allocation_percentage'] / 100).sum()
                
                port_summary = f"Portfolio: ₹{total_invested:.2f} invested, {len(portfolio)} securities, "
                port_summary += f"Expected Return: {weighted_return:.2f}%"
            else:
                port_summary = f"Portfolio: ₹{total_invested:.2f} invested, {len(portfolio)} securities"
            
            summary.append({
                "type": "portfolio_summary",
                "content": port_summary
            })
        
        return summary
    
    def _generate_visualizations(self, data, risk_profiles, predictions, 
                                portfolio, sentiment_data, report_date):
        """
        Generate visualizations for PDF reports.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Historical price data
        risk_profiles : pd.DataFrame
            Risk classifications
        predictions : pd.DataFrame
            Predicted future data
        portfolio : pd.DataFrame, optional
            Portfolio allocation
        sentiment_data : dict, optional
            Market sentiment data
        report_date : str
            Report generation date
            
        Returns:
        --------
        str : Path to the generated visualization
        """
        # Create a larger figure with multiple subplots
        plt.figure(figsize=(15, 20))
        gs = gridspec.GridSpec(4, 2)
        
        # 1. Risk Distribution Plot
        if not risk_profiles.empty:
            ax1 = plt.subplot(gs[0, 0])
            risk_counts = risk_profiles['risk_category'].value_counts()
            risk_counts.plot(kind='pie', autopct='%1.1f%%', ax=ax1)
            ax1.set_title('Investment Risk Distribution')
            ax1.set_ylabel('')
        
        # 2. Expected Returns Plot
        if not predictions.empty and 'expected_return' in predictions.columns:
            ax2 = plt.subplot(gs[0, 1])
            top_return = predictions.sort_values('expected_return', ascending=False).head(10)
            top_return.set_index('ticker')['expected_return'].plot(kind='bar', ax=ax2)
            ax2.set_title('Top 10 Expected Returns')
            ax2.set_ylabel('Expected Return (%)')
            plt.xticks(rotation=45)
        
        # 3. Debt-to-Equity vs Cash Flow Growth Scatter Plot
        if (not risk_profiles.empty and 'debt_equity_ratio' in risk_profiles.columns 
            and 'cash_flow_growth' in risk_profiles.columns):
            ax3 = plt.subplot(gs[1, 0])
            
            # Remove extreme outliers for better visualization
            plot_data = risk_profiles[
                (risk_profiles['debt_equity_ratio'] < 5) & 
                (risk_profiles['cash_flow_growth'] > -50) & 
                (risk_profiles['cash_flow_growth'] < 50)
            ]
            
            scatter = ax3.scatter(
                plot_data['debt_equity_ratio'], 
                plot_data['cash_flow_growth'],
                c=plot_data['risk_score'],
                cmap='RdYlGn_r',
                alpha=0.7,
                s=100
            )
            
            for _, row in plot_data.iterrows():
                ax3.annotate(
                    row['ticker'], 
                    (row['debt_equity_ratio'], row['cash_flow_growth']),
                    fontsize=8
                )
            
            plt.colorbar(scatter, label='Risk Score (Higher = Riskier)')
            ax3.set_title('Debt-to-Equity Ratio vs Cash Flow Growth')
            ax3.set_xlabel('Debt-to-Equity Ratio')
            ax3.set_ylabel('Cash Flow Growth (%)')
            ax3.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            ax3.axvline(x=1, color='gray', linestyle='--', alpha=0.5)
        
        # 4. PE Ratio vs PEG Ratio Scatter Plot
        if (not risk_profiles.empty and 'pe_ratio' in risk_profiles.columns 
            and 'peg_ratio' in risk_profiles.columns):
            ax4 = plt.subplot(gs[1, 1])
            
            # Remove extreme outliers for better visualization
            plot_data = risk_profiles[
                (risk_profiles['pe_ratio'] < 50) & 
                (risk_profiles['pe_ratio'] > 0) & 
                (risk_profiles['peg_ratio'] < 5) & 
                (risk_profiles['peg_ratio'] > 0)
            ]
            
            scatter = ax4.scatter(
                plot_data['pe_ratio'], 
                plot_data['peg_ratio'],
                c=plot_data['expected_return'],
                cmap='viridis',
                alpha=0.7,
                s=100
            )
            
            for _, row in plot_data.iterrows():
                ax4.annotate(
                    row['ticker'], 
                    (row['pe_ratio'], row['peg_ratio']),
                    fontsize=8
                )
            
            plt.colorbar(scatter, label='Expected Return (%)')
            ax4.set_title('PE Ratio vs PEG Ratio')
            ax4.set_xlabel('PE Ratio')
            ax4.set_ylabel('PEG Ratio')
            ax4.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
        
        # 5. Portfolio Allocation Pie Chart
        if portfolio is not None and not portfolio.empty:
            ax5 = plt.subplot(gs[2, 0])
            portfolio.set_index('ticker')['allocation_amount'].plot(
                kind='pie', 
                autopct='%1.1f%%', 
                ax=ax5
            )
            ax5.set_title('Portfolio Allocation')
            ax5.set_ylabel('')
        
        # 6. Risk vs Return Scatter Plot
        if not risk_profiles.empty and 'expected_return' in risk_profiles.columns:
            ax6 = plt.subplot(gs[2, 1])
            
            scatter = ax6.scatter(
                risk_profiles['risk_score'], 
                risk_profiles['expected_return'],
                c=risk_profiles['volatility'],
                cmap='coolwarm',
                alpha=0.7,
                s=100
            )
            
            top_picks = risk_profiles.sort_values('expected_return', ascending=False).head(5)
            for _, row in top_picks.iterrows():
                ax6.annotate(
                    row['ticker'], 
                    (row['risk_score'], row['expected_return']),
                    fontsize=10
                )
            
            plt.colorbar(scatter, label='Volatility')
            ax6.set_title('Risk vs Return')
            ax6.set_xlabel('Risk Score')
            ax6.set_ylabel('Expected Return (%)')
        
        # 7. Fear & Greed Index Historical Trend (simulated)
        if sentiment_data is not None:
            ax7 = plt.subplot(gs[3, 0:])
            
            # Create a simulated 30-day history based on the available data points
            dates = pd.date_range(end=datetime.now(), periods=30)
            
            # Start with current value and interpolate to historical values provided
            current = sentiment_data.get('fear_greed_index', 50)
            week_ago = sentiment_data.get('one_week_ago', current)
            month_ago = sentiment_data.get('one_month_ago', week_ago)
            
            # Simplified simulation - linear interpolation between points
            values = np.linspace(month_ago, week_ago, 23)
            values = np.append(values, np.linspace(week_ago, current, 7))
            
            ax7.plot(dates, values, marker='o', linestyle='-', linewidth=2)
            
            # Add colored background for different sentiment zones
            ax7.axhspan(0, 25, alpha=0.2, color='red', label='Extreme Fear')
            ax7.axhspan(25, 45, alpha=0.2, color='orange', label='Fear')
            ax7.axhspan(45, 55, alpha=0.2, color='yellow', label='Neutral')
            ax7.axhspan(55, 75, alpha=0.2, color='lightgreen', label='Greed')
            ax7.axhspan(75, 100, alpha=0.2, color='green', label='Extreme Greed')
            
            ax7.set_title('Fear & Greed Index - Historical Trend')
            ax7.set_xlabel('Date')
            ax7.set_ylabel('Fear & Greed Index')
            ax7.set_ylim(0, 100)
            ax7.legend(loc='upper left')
            ax7.grid(True, alpha=0.3)
        
        # Adjust layout and save figure
        plt.tight_layout()
        viz_path = os.path.join(self.reports_dir, f"visualizations_{report_date}.png")
        plt.savefig(viz_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return viz_path