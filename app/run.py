"""
Main Application Runner

This module provides the main entry point for running the Investment Recommendation System.
It supports various commands for generating reports, training models, and running the web interface.
"""

import os
import sys
import argparse
import webbrowser
from time import sleep

# Import configuration
from config.settings import ensure_directories

# Import modules
from app.dashboard.dashboard_generator import generate_dashboard
from app.utils.sample_generator import create_sample_json_file, create_sample_csv_file

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Investment Recommendation System')
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Dashboard command
    dashboard_parser = subparsers.add_parser('dashboard', help='Generate and view the dashboard')
    dashboard_parser.add_argument('--no-browser', action='store_true', help='Do not open the browser automatically')
    
    # Train command (placeholder for future implementation)
    train_parser = subparsers.add_parser('train', help='Train prediction models')
    train_parser.add_argument('--stocks', nargs='*', help='Specific stocks to train models for')
    train_parser.add_argument('--all', action='store_true', help='Train models for all available stocks')
    
    # Report command (placeholder for future implementation)
    report_parser = subparsers.add_parser('report', help='Generate investment recommendation report')
    report_parser.add_argument('--risk-profile', choices=['low', 'medium', 'high'], 
                              default='medium', help='Risk profile for recommendations')
    
    # Web interface command (placeholder for future implementation)
    web_parser = subparsers.add_parser('web', help='Run the web interface')
    web_parser.add_argument('--port', type=int, default=8000, help='Port to run the web server on')
    
    # Sample command to generate sample data
    sample_parser = subparsers.add_parser('sample', help='Generate sample data for demonstration')
    
    return parser.parse_args()

def run_dashboard(no_browser=False):
    """Generate and open the dashboard."""
    print("Generating investment dashboard...")
    dashboard_file = generate_dashboard()
    
    if dashboard_file and not no_browser:
        print(f"Opening dashboard in browser: {dashboard_file}")
        webbrowser.open('file://' + os.path.abspath(dashboard_file))
    elif dashboard_file:
        print(f"Dashboard generated at: {dashboard_file}")
    else:
        print("Failed to generate dashboard.")
        return False
    
    return True

def run_sample_generation():
    """Generate sample data files for demonstration."""
    print("Generating sample data files...")
    json_files = create_sample_json_file()
    csv_file = create_sample_csv_file()
    
    print("\nSample generation complete. The following files were created:")
    for file in json_files:
        print(f"- {file}")
    print(f"- {csv_file}")
    
    return True

def run_web_interface(port=8000):
    """Run the web interface (placeholder for future implementation)."""
    print(f"Starting web interface on port {port}...")
    print("This feature is not yet implemented.")
    print("For now, you can use the dashboard command to view reports.")
    return True

def main():
    """Main entry point for the application."""
    # Ensure all required directories exist
    ensure_directories()
    
    # Parse command line arguments
    args = parse_arguments()
    
    # If no command is provided, show help and use dashboard as default
    if not args.command:
        print("No command specified. Defaulting to dashboard.\n")
        return run_dashboard()
    
    # Execute the appropriate command
    if args.command == 'dashboard':
        return run_dashboard(args.no_browser)
    elif args.command == 'train':
        print("Training functionality is not yet implemented.")
        return False
    elif args.command == 'report':
        print("Report generation is not yet implemented.")
        return False
    elif args.command == 'web':
        return run_web_interface(args.port)
    elif args.command == 'sample':
        return run_sample_generation()
    else:
        print(f"Unknown command: {args.command}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 