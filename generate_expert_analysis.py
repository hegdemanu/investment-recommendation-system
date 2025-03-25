#!/usr/bin/env python3
"""
Generate expert-level analytical reports for mutual fund model performance.
This script:
1. Loads training summary data for mutual funds
2. Analyzes model performance across different time horizons
3. Generates detailed visualizations comparing LSTM, ARIMA-GARCH, and Prophet models
4. Creates PDF reports with technical analysis for financial experts
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from datetime import datetime, timedelta
from pathlib import Path
import logging
from fpdf import FPDF
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("expert_analysis.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("ExpertAnalysis")

# Configure directories
MF_MODELS_DIR = "./models/mutual_funds"
EXPERT_REPORTS_DIR = "./results/reports/expert"
MF_TRAINING_SUMMARY = "./results/training/mutual_funds_training_summary.json"
EXPERT_PDF_DIR = "./results/reports/expert/pdf"
MF_DATA_DIR = "./data/mutual_funds/csv"

# Create directories if they don't exist
os.makedirs(EXPERT_REPORTS_DIR, exist_ok=True)
os.makedirs(EXPERT_PDF_DIR, exist_ok=True)

# Time horizon definitions from training script
TIME_HORIZONS = {
    'short': [1, 3, 5],      # Short-term: 1-5 days
    'medium': [7, 14, 21],   # Medium-term: 7-21 days
    'long': [30, 60, 90]     # Long-term: 30-90 days
}

# Model weights by horizon
MODEL_WEIGHTS = {
    'short': {'lstm': 0.5, 'arima_garch': 0.3, 'prophet': 0.2},
    'medium': {'lstm': 0.4, 'arima_garch': 0.3, 'prophet': 0.3},
    'long': {'lstm': 0.2, 'arima_garch': 0.4, 'prophet': 0.4}  # More weight to ARIMA-GARCH and Prophet for long-term
}

class PDF(FPDF):
    """Custom PDF class with header and footer for expert reports."""
    def header(self):
        # Logo
        # self.image('logo.png', 10, 8, 33)
        # Arial bold 15
        self.set_font('Arial', 'B', 15)
        # Move to the right
        self.cell(80)
        # Title
        self.cell(30, 10, 'Mutual Fund Model Analysis', 0, 0, 'C')
        # Line break
        self.ln(20)

    def footer(self):
        # Position at 1.5 cm from bottom
        self.set_y(-15)
        # Arial italic 8
        self.set_font('Arial', 'I', 8)
        # Page number
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')
        
    def chapter_title(self, title):
        # Arial 12
        self.set_font('Arial', 'B', 12)
        # Background color
        self.set_fill_color(200, 220, 255)
        # Title
        self.cell(0, 6, title, 0, 1, 'L', 1)
        # Line break
        self.ln(4)
        
    def chapter_body(self, body):
        # Times 12
        self.set_font('Arial', '', 10)
        # Output justified text
        self.multi_cell(0, 5, body)
        # Line break
        self.ln()

def load_mf_training_summary():
    """Load mutual fund training summary data."""
    try:
        if not os.path.exists(MF_TRAINING_SUMMARY):
            logger.error(f"Mutual fund training summary not found at {MF_TRAINING_SUMMARY}")
            return None
            
        with open(MF_TRAINING_SUMMARY, 'r') as f:
            summary = json.load(f)
            
        logger.info(f"Loaded mutual fund training summary with {len(summary.get('details', {}))} funds")
        return summary
    except Exception as e:
        logger.error(f"Error loading mutual fund training summary: {e}")
        return None

def load_fund_data(fund_name):
    """Load mutual fund data from CSV."""
    try:
        csv_path = os.path.join(MF_DATA_DIR, f"{fund_name}.csv")
        if not os.path.exists(csv_path):
            logger.error(f"Mutual fund data not found at {csv_path}")
            return None
            
        data = pd.read_csv(csv_path)
        data['Date'] = pd.to_datetime(data['Date'])
        
        logger.info(f"Loaded {len(data)} rows of data for {fund_name}")
        return data
    except Exception as e:
        logger.error(f"Error loading mutual fund data for {fund_name}: {e}")
        return None

def load_model_metadata(fund_name, model_type):
    """Load model metadata for the specified fund and model type."""
    try:
        metadata_paths = {
            'lstm': os.path.join(MF_MODELS_DIR, f"{fund_name}_metadata.json"),
            'arima_garch': os.path.join(MF_MODELS_DIR, f"{fund_name}_arima_garch_metadata.json"),
            'prophet': os.path.join(MF_MODELS_DIR, f"{fund_name}_prophet_metadata.json")
        }
        
        if not os.path.exists(metadata_paths[model_type]):
            logger.warning(f"Metadata not found for {fund_name} {model_type} model")
            return None
            
        with open(metadata_paths[model_type], 'r') as f:
            metadata = json.load(f)
            
        logger.info(f"Loaded metadata for {fund_name} {model_type} model")
        return metadata
    except Exception as e:
        logger.error(f"Error loading metadata for {fund_name} {model_type} model: {e}")
        return None

def analyze_model_performance(fund_name, fund_details):
    """Analyze the performance of different models across time horizons."""
    try:
        if 'model_results' not in fund_details:
            logger.error(f"No model results found for {fund_name}")
            return None
            
        model_results = fund_details['model_results']
        
        # Initialize performance metrics
        performance = {
            'fund_name': fund_name,
            'data_points': fund_details.get('data_points', 0),
            'models_trained': fund_details.get('models_trained', []),
            'time_horizons': TIME_HORIZONS,
            'model_weights': MODEL_WEIGHTS,
            'model_metrics': {}
        }
        
        # Collect metrics for each model type
        for model_type, model_result in model_results.items():
            if model_result.get('status') != 'SUCCESS':
                continue
                
            # Load model metadata for detailed metrics
            metadata = load_model_metadata(fund_name, model_type)
            
            if metadata:
                performance['model_metrics'][model_type] = {
                    'training_time': model_result.get('training_time', 0),
                    'rmse': metadata.get('rmse', {}),
                    'mape': metadata.get('mape', {}),
                    'mae': metadata.get('mae', {}),
                    'r2': metadata.get('r2', {})
                }
        
        # Calculate horizon-based weighted scores
        if performance['model_metrics']:
            performance['horizon_scores'] = {}
            
            for horizon_type, horizons in TIME_HORIZONS.items():
                weights = MODEL_WEIGHTS[horizon_type]
                horizon_scores = []
                
                for horizon in horizons:
                    horizon_str = str(horizon)
                    weighted_score = 0
                    contributing_models = 0
                    
                    for model_type, metrics in performance['model_metrics'].items():
                        if horizon_str in metrics.get('rmse', {}):
                            weighted_score += float(metrics['rmse'][horizon_str]) * weights[model_type]
                            contributing_models += 1
                    
                    if contributing_models > 0:
                        horizon_scores.append({
                            'horizon': horizon,
                            'weighted_score': weighted_score / sum(weights.values()),
                            'contributing_models': contributing_models
                        })
                
                if horizon_scores:
                    performance['horizon_scores'][horizon_type] = horizon_scores
        
        logger.info(f"Analyzed performance for {fund_name} across {len(performance.get('horizon_scores', {}))} time horizons")
        return performance
    except Exception as e:
        logger.error(f"Error analyzing model performance for {fund_name}: {e}")
        return None

def generate_performance_visualizations(fund_name, performance, fund_data):
    """Generate visualizations for model performance comparison."""
    vis_paths = {}
    
    try:
        # Set up styling
        plt.style.use('seaborn-v0_8-whitegrid')
        
        # 1. RMSE comparison across models and horizons
        if 'model_metrics' in performance and len(performance['model_metrics']) > 0:
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Prepare data for plotting
            horizons = []
            models = []
            rmse_values = []
            
            for model_type, metrics in performance['model_metrics'].items():
                for horizon, rmse in metrics.get('rmse', {}).items():
                    horizons.append(int(horizon))
                    models.append(model_type)
                    rmse_values.append(float(rmse))
            
            if horizons:
                df = pd.DataFrame({
                    'Horizon': horizons,
                    'Model': models,
                    'RMSE': rmse_values
                })
                
                # Plot
                sns.lineplot(data=df, x='Horizon', y='RMSE', hue='Model', marker='o', ax=ax)
                ax.set_title(f'{fund_name} - Model RMSE Comparison by Horizon', fontsize=16)
                ax.set_xlabel('Prediction Horizon (Days)', fontsize=14)
                ax.set_ylabel('Root Mean Square Error (RMSE)', fontsize=14)
                ax.legend(title='Model Type', fontsize=12)
                
                # Highlight long-term horizons
                for horizon in TIME_HORIZONS['long']:
                    ax.axvline(x=horizon, color='red', linestyle='--', alpha=0.3)
                
                plt.tight_layout()
                rmse_path = os.path.join(EXPERT_REPORTS_DIR, f"{fund_name}_rmse_comparison.png")
                plt.savefig(rmse_path, dpi=300)
                plt.close(fig)
                
                vis_paths['rmse_comparison'] = rmse_path
                logger.info(f"Generated RMSE comparison visualization for {fund_name}")
            
        # 2. Historical data with predictions visualization
        if fund_data is not None:
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Plot historical NAV
            fund_data.sort_values('Date', inplace=True)
            ax.plot(fund_data['Date'], fund_data['NAV'], label='Historical NAV', color='black')
            
            # Add legend and labels
            ax.set_title(f'{fund_name} - Historical NAV', fontsize=16)
            ax.set_xlabel('Date', fontsize=14)
            ax.set_ylabel('NAV Value', fontsize=14)
            
            # Format x-axis as dates
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.xticks(rotation=45)
            
            plt.tight_layout()
            history_path = os.path.join(EXPERT_REPORTS_DIR, f"{fund_name}_historical_nav.png")
            plt.savefig(history_path, dpi=300)
            plt.close(fig)
            
            vis_paths['historical_nav'] = history_path
            logger.info(f"Generated historical NAV visualization for {fund_name}")
            
        # 3. Model weight visualization by time horizon
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Prepare data
        horizon_types = list(MODEL_WEIGHTS.keys())
        model_types = list(MODEL_WEIGHTS['short'].keys())
        
        bar_width = 0.25
        index = np.arange(len(horizon_types))
        
        # Plot bars for each model type
        for i, model_type in enumerate(model_types):
            weights = [MODEL_WEIGHTS[h][model_type] for h in horizon_types]
            ax.bar(index + i*bar_width, weights, bar_width, label=model_type)
        
        ax.set_xlabel('Time Horizon', fontsize=14)
        ax.set_ylabel('Model Weight', fontsize=14)
        ax.set_title('Model Weights by Time Horizon', fontsize=16)
        ax.set_xticks(index + bar_width)
        ax.set_xticklabels(horizon_types)
        ax.legend()
        
        plt.tight_layout()
        weights_path = os.path.join(EXPERT_REPORTS_DIR, f"{fund_name}_model_weights.png")
        plt.savefig(weights_path, dpi=300)
        plt.close(fig)
        
        vis_paths['model_weights'] = weights_path
        logger.info(f"Generated model weights visualization for {fund_name}")
        
        return vis_paths
    except Exception as e:
        logger.error(f"Error generating visualizations for {fund_name}: {e}")
        return vis_paths

def generate_pdf_report(fund_name, performance, visualizations):
    """Generate a PDF report with technical analysis for experts."""
    try:
        # Initialize PDF
        pdf = PDF()
        pdf.add_page()
        
        # Add report title
        pdf.set_font('Arial', 'B', 16)
        pdf.cell(0, 10, f"Expert Analysis: {fund_name} Mutual Fund", 0, 1, 'C')
        pdf.ln(10)
        
        # Add report date and summary
        pdf.set_font('Arial', '', 10)
        pdf.cell(0, 5, f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 0, 1)
        pdf.cell(0, 5, f"Data Points: {performance.get('data_points', 'N/A')}", 0, 1)
        pdf.cell(0, 5, f"Models Trained: {', '.join(performance.get('models_trained', []))}", 0, 1)
        pdf.ln(5)
        
        # Data quality and model training section
        pdf.chapter_title("1. Data Quality and Model Training")
        data_quality_text = (
            f"The mutual fund data for {fund_name} contains {performance.get('data_points', 0)} data points. "
            f"This data was used to train {len(performance.get('models_trained', []))} different model types: "
            f"{', '.join(performance.get('models_trained', []))}."
        )
        pdf.chapter_body(data_quality_text)
        
        # Model specific training details
        if 'model_metrics' in performance:
            pdf.ln(5)
            for model_type, metrics in performance['model_metrics'].items():
                pdf.set_font('Arial', 'B', 11)
                pdf.cell(0, 5, f"{model_type.upper()} Model", 0, 1)
                pdf.set_font('Arial', '', 10)
                pdf.cell(0, 5, f"Training Time: {metrics.get('training_time', 'N/A'):.2f} seconds", 0, 1)
                
                # RMSE by horizon if available
                if 'rmse' in metrics and metrics['rmse']:
                    pdf.cell(0, 5, "RMSE by Horizon:", 0, 1)
                    for horizon, rmse in sorted(metrics['rmse'].items(), key=lambda x: int(x[0])):
                        pdf.cell(0, 5, f"  {horizon} days: {float(rmse):.4f}", 0, 1)
                
                pdf.ln(5)
        
        # Performance analysis section
        pdf.chapter_title("2. Performance Analysis by Time Horizon")
        
        # Add text explaining the performance analysis
        perf_analysis_text = (
            "The models are evaluated across different time horizons: short-term (1-5 days), "
            "medium-term (7-21 days), and long-term (30-90 days). For long-term predictions, "
            "ARIMA-GARCH and Prophet models are given higher weights due to their superior "
            "performance in capturing long-term trends and seasonality."
        )
        pdf.chapter_body(perf_analysis_text)
        
        # Add horizon scores if available
        if 'horizon_scores' in performance:
            pdf.ln(5)
            for horizon_type, scores in performance['horizon_scores'].items():
                pdf.set_font('Arial', 'B', 11)
                pdf.cell(0, 5, f"{horizon_type.capitalize()} Term Horizons", 0, 1)
                pdf.set_font('Arial', '', 10)
                
                for score in scores:
                    pdf.cell(0, 5, 
                        f"  {score['horizon']} days: Weighted RMSE {score['weighted_score']:.4f} "
                        f"(from {score['contributing_models']} models)", 0, 1)
                
                pdf.ln(5)
        
        # Add visualizations
        pdf.chapter_title("3. Performance Visualizations")
        
        # RMSE comparison chart
        if 'rmse_comparison' in visualizations:
            pdf.ln(5)
            pdf.cell(0, 5, "RMSE Comparison Across Models and Horizons:", 0, 1)
            pdf.image(visualizations['rmse_comparison'], x=10, w=180)
            pdf.ln(5)
            
            rmse_desc = (
                "The chart above shows the Root Mean Square Error (RMSE) for each model across "
                "different prediction horizons. Lower RMSE indicates better predictive accuracy. "
                "Note how the models perform differently as the prediction horizon extends."
            )
            pdf.chapter_body(rmse_desc)
        
        # Historical NAV chart
        if 'historical_nav' in visualizations:
            pdf.ln(5)
            pdf.cell(0, 5, "Historical NAV Values:", 0, 1)
            pdf.image(visualizations['historical_nav'], x=10, w=180)
            pdf.ln(5)
            
            nav_desc = (
                "This chart displays the historical Net Asset Value (NAV) for the mutual fund. "
                "Understanding the historical pattern is crucial for evaluating model predictions."
            )
            pdf.chapter_body(nav_desc)
        
        # Model weights chart
        if 'model_weights' in visualizations:
            pdf.ln(5)
            pdf.cell(0, 5, "Model Weights by Time Horizon:", 0, 1)
            pdf.image(visualizations['model_weights'], x=10, w=180)
            pdf.ln(5)
            
            weights_desc = (
                "The chart shows how different models are weighted across time horizons. "
                "For long-term predictions, ARIMA-GARCH and Prophet models are given higher weights "
                "due to their ability to capture long-term trends."
            )
            pdf.chapter_body(weights_desc)
        
        # Recommendations section
        pdf.add_page()
        pdf.chapter_title("4. Expert Recommendations")
        
        # Determine best model for each horizon type
        best_models = {}
        if 'model_metrics' in performance:
            for horizon_type, horizons in TIME_HORIZONS.items():
                best_model = None
                best_rmse = float('inf')
                best_horizon = None
                
                for model_type, metrics in performance['model_metrics'].items():
                    for horizon in horizons:
                        horizon_str = str(horizon)
                        if horizon_str in metrics.get('rmse', {}):
                            rmse = float(metrics['rmse'][horizon_str])
                            if rmse < best_rmse:
                                best_rmse = rmse
                                best_model = model_type
                                best_horizon = horizon
                
                if best_model:
                    best_models[horizon_type] = {
                        'model': best_model,
                        'horizon': best_horizon,
                        'rmse': best_rmse
                    }
        
        # Add recommendations based on best models
        recommendations = []
        
        if 'short' in best_models:
            rec = (
                f"For short-term predictions (1-5 days), the {best_models['short']['model'].upper()} model "
                f"provides the best accuracy with RMSE of {best_models['short']['rmse']:.4f} at "
                f"{best_models['short']['horizon']} days horizon."
            )
            recommendations.append(rec)
            
        if 'medium' in best_models:
            rec = (
                f"For medium-term forecasting (7-21 days), use the {best_models['medium']['model'].upper()} model "
                f"which achieved RMSE of {best_models['medium']['rmse']:.4f} at "
                f"{best_models['medium']['horizon']} days horizon."
            )
            recommendations.append(rec)
            
        if 'long' in best_models:
            rec = (
                f"For long-term analysis (30-90 days), the {best_models['long']['model'].upper()} model "
                f"is recommended with RMSE of {best_models['long']['rmse']:.4f} at "
                f"{best_models['long']['horizon']} days horizon."
            )
            recommendations.append(rec)
        
        # Add ensemble recommendation if multiple models are available
        if len(performance.get('models_trained', [])) > 1:
            rec = (
                "For optimal results, consider using an ensemble approach that combines predictions "
                "from multiple models using the specified weights for each time horizon. This approach "
                "leverages the strengths of each model and typically provides more robust predictions "
                "across different market conditions."
            )
            recommendations.append(rec)
        
        # Add recommendations to PDF
        for rec in recommendations:
            pdf.cell(0, 5, "•", 0, 0)
            pdf.multi_cell(180, 5, rec)
            pdf.ln(3)
        
        # Save PDF
        pdf_path = os.path.join(EXPERT_PDF_DIR, f"{fund_name}_expert_analysis.pdf")
        pdf.output(pdf_path)
        
        logger.info(f"Generated expert PDF report for {fund_name} at {pdf_path}")
        return pdf_path
    except Exception as e:
        logger.error(f"Error generating PDF report for {fund_name}: {e}")
        return None

def process_fund(fund_name, fund_details):
    """Process a single mutual fund for expert analysis."""
    logger.info(f"Processing {fund_name} for expert analysis")
    
    try:
        # Skip funds without successful model training
        if fund_details.get('status') != 'SUCCESS':
            logger.warning(f"Skipping {fund_name}: No successful models trained")
            return None
            
        # Load fund data
        fund_data = load_fund_data(fund_name)
        if fund_data is None:
            logger.warning(f"Skipping {fund_name}: Could not load fund data")
            return None
            
        # Analyze model performance
        performance = analyze_model_performance(fund_name, fund_details)
        if performance is None:
            logger.warning(f"Skipping {fund_name}: Could not analyze performance")
            return None
            
        # Generate visualizations
        visualizations = generate_performance_visualizations(fund_name, performance, fund_data)
        
        # Generate PDF report
        pdf_path = generate_pdf_report(fund_name, performance, visualizations)
        
        return {
            'fund_name': fund_name,
            'performance': performance,
            'visualizations': visualizations,
            'pdf_report': pdf_path
        }
    except Exception as e:
        logger.error(f"Error processing {fund_name} for expert analysis: {e}")
        return None

def main():
    """Main function to generate expert analysis reports."""
    logger.info("=== Generating Expert Analysis Reports ===")
    
    # Load mutual fund training summary
    mf_summary = load_mf_training_summary()
    if mf_summary is None:
        logger.error("Failed to load mutual fund training summary")
        return False
        
    # Get funds with successful model training
    funds = mf_summary.get('details', {})
    
    if not funds:
        logger.warning("No mutual funds found in training summary")
        return False
        
    logger.info(f"Found {len(funds)} mutual funds in training summary")
    
    # Process each fund
    reports = {}
    for fund_name, fund_details in funds.items():
        result = process_fund(fund_name, fund_details)
        if result:
            reports[fund_name] = result
    
    # Generate summary of expert reports
    summary = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'total_funds': len(funds),
        'reports_generated': len(reports),
        'report_details': {name: {
            'fund_name': report['fund_name'],
            'pdf_report': os.path.basename(report['pdf_report']) if report.get('pdf_report') else None,
            'visualizations': {k: os.path.basename(v) for k, v in report.get('visualizations', {}).items()}
        } for name, report in reports.items()}
    }
    
    # Save summary
    summary_path = os.path.join(EXPERT_REPORTS_DIR, "expert_reports_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Expert analysis summary saved to {summary_path}")
    
    # Print summary
    logger.info(f"\n=== Expert Analysis Summary ===")
    logger.info(f"Total funds processed: {len(funds)}")
    logger.info(f"Expert reports generated: {len(reports)}")
    logger.info(f"Reports saved to: {EXPERT_PDF_DIR}")
    
    return len(reports) > 0

if __name__ == "__main__":
    main() 