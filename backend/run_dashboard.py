#!/usr/bin/env python3
"""
Script to generate and display the investment dashboard.
"""

import os
import sys
from dashboard.dashboard_generator import generate_dashboard, main

if __name__ == "__main__":
    print("Generating investment dashboard...")
    # Run with open_browser=True to automatically open the dashboard in a browser
    dashboard_path = main()
    print(f"Dashboard generated at: {dashboard_path}")
    print("Access the dashboard by opening the above file in your browser.") 