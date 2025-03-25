"""
Dashboard Template

This module provides the HTML template for the investment dashboard
with tabs for various sections including charts, JSON reports, and data files.
"""

def get_dashboard_template():
    """Returns the HTML template for the investment dashboard."""
    return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Investment Analysis Dashboard</title>
    <style>
        /* Reset and Global Styles */
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        :root {{
            --primary-color: #3498db;
            --secondary-color: #2c3e50;
            --bg-color: #f5f5f5;
            --card-bg: #ffffff;
            --text-color: #333333;
            --border-color: #dddddd;
            --success-color: #2ecc71;
            --warning-color: #f39c12;
            --error-color: #e74c3c;
            --tab-active-bg: #ffffff;
            --tab-inactive-bg: #e9e9e9;
        }}
        
        [data-theme="dark"] {{
            --primary-color: #2980b9;
            --secondary-color: #34495e;
            --bg-color: #1a1a1a;
            --card-bg: #2c2c2c;
            --text-color: #f5f5f5;
            --border-color: #444444;
            --success-color: #27ae60;
            --warning-color: #d35400;
            --error-color: #c0392b;
            --tab-active-bg: #2c2c2c;
            --tab-inactive-bg: #1a1a1a;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            background-color: var(--bg-color);
            color: var(--text-color);
            transition: all 0.3s ease;
        }}
        
        /* Header and Navigation */
        .header {{
            background: rgba(44, 62, 80, 0.9);
            backdrop-filter: blur(10px);
            color: white;
            padding: 1rem;
            position: sticky;
            top: 0;
            z-index: 100;
            display: flex;
            justify-content: space-between;
            align-items: center;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }}
        
        .header h1 {{
            margin: 0;
            font-size: 1.5rem;
            cursor: pointer; /* Make header clickable */
        }}
        
        .nav-container {{
            display: flex;
            align-items: center;
        }}
        
        .nav-tabs {{
            display: flex;
            margin-right: 1rem;
        }}
        
        .tab {{
            background: none;
            border: none;
            color: white;
            padding: 0.5rem 1rem;
            margin: 0 0.25rem;
            border-radius: 4px;
            cursor: pointer;
            transition: background-color 0.3s;
            opacity: 0.7;
            user-select: none;
            outline: none; /* Remove outline focus */
        }}
        
        .tab:hover {{
            background-color: rgba(255, 255, 255, 0.1);
            opacity: 1;
        }}
        
        .tab.active {{
            background-color: rgba(255, 255, 255, 0.2);
            opacity: 1;
        }}
        
        .theme-toggle {{
            background: none;
            border: none;
            color: white;
            cursor: pointer;
            font-size: 1rem;
            width: 40px;
            height: 40px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: background-color 0.3s;
            outline: none; /* Remove outline focus */
        }}
        
        .theme-toggle:hover {{
            background-color: rgba(255, 255, 255, 0.1);
        }}
        
        /* Main Content */
        .container {{
            max-width: 1200px;
            margin: 1rem auto;
            padding: 0 1rem;
        }}
        
        .content-section {{
            display: none;
            padding: 1rem;
            background-color: var(--card-bg);
            border-radius: 8px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            margin-bottom: 1rem;
        }}
        
        .content-section.active {{
            display: block;
        }}
        
        /* Charts Section */
        .charts-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(400px, 1fr));
            gap: 1rem;
        }}
        
        .chart-card {{
            background-color: var(--card-bg);
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            transition: transform 0.3s;
        }}
        
        .chart-card:hover {{
            transform: translateY(-5px);
        }}
        
        .chart-img {{
            width: 100%;
            height: auto;
            display: block;
        }}
        
        /* JSON Reports Section */
        .json-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 1rem;
        }}
        
        .json-card {{
            background-color: var(--card-bg);
            border-radius: 8px;
            padding: 1rem;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            cursor: pointer;
            transition: transform 0.3s, box-shadow 0.3s;
            position: relative;
            overflow: hidden;
        }}
        
        .json-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        }}
        
        .json-card::after {{
            content: '';
            position: absolute;
            bottom: 0;
            left: 0;
            width: 100%;
            height: 0;
            background: rgba(52, 152, 219, 0.1);
            transition: height 0.3s ease;
            z-index: 0;
        }}
        
        .json-card:hover::after {{
            height: 100%;
        }}
        
        .json-card h3 {{
            margin-bottom: 0.5rem;
            color: var(--primary-color);
            position: relative;
            z-index: 1;
        }}
        
        /* Data Files Section */
        .data-filters {{
            margin-bottom: 1rem;
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
        }}
        
        .filter-btn {{
            background-color: var(--tab-inactive-bg);
            border: none;
            padding: 0.5rem 1rem;
            border-radius: 4px;
            cursor: pointer;
            transition: background-color 0.3s, transform 0.2s;
            color: var(--text-color);
            outline: none; /* Remove outline focus */
        }}
        
        .filter-btn:hover {{
            background-color: var(--primary-color);
            color: white;
            transform: translateY(-2px);
        }}
        
        .filter-btn.active {{
            background-color: var(--primary-color);
            color: white;
        }}
        
        .data-table {{
            width: 100%;
            border-collapse: collapse;
        }}
        
        .data-table th, .data-table td {{
            border: 1px solid var(--border-color);
            padding: 0.75rem;
            text-align: left;
        }}
        
        .data-table th {{
            background-color: var(--tab-inactive-bg);
            position: sticky;
            top: 70px;
        }}
        
        .data-table tr:hover {{
            background-color: rgba(0,0,0,0.05);
        }}
        
        /* JSON Viewer Modal */
        .modal {{
            display: none;
            position: fixed;
            z-index: 1000;
            left: 0;
            top: 0;
            width: 100%;
            height: 100%;
            overflow: auto;
            background-color: rgba(0,0,0,0.7);
        }}
        
        .modal-content {{
            position: relative;
            background-color: var(--card-bg);
            margin: 5% auto;
            padding: 1.5rem;
            border-radius: 8px;
            width: 80%;
            max-height: 80vh;
            overflow-y: auto;
            box-shadow: 0 5px 15px rgba(0,0,0,0.3);
        }}
        
        .close-btn {{
            position: absolute;
            top: 1rem;
            right: 1rem;
            color: var(--text-color);
            font-size: 1.5rem;
            font-weight: bold;
            cursor: pointer;
            transition: transform 0.3s, color 0.3s;
            width: 30px;
            height: 30px;
            display: flex;
            align-items: center;
            justify-content: center;
            border-radius: 50%;
        }}
        
        .close-btn:hover {{
            color: var(--primary-color);
            transform: rotate(90deg);
        }}
        
        .json-content {{
            white-space: pre-wrap;
            overflow-x: auto;
            font-family: 'Courier New', Courier, monospace;
            background-color: var(--bg-color);
            padding: 1rem;
            border-radius: 4px;
        }}
        
        /* Search functionality */
        .search-container {{
            margin-bottom: 1rem;
        }}
        
        .search-input {{
            width: 100%;
            padding: 0.75rem;
            border: 1px solid var(--border-color);
            border-radius: 4px;
            background-color: var(--card-bg);
            color: var(--text-color);
        }}
        
        /* Responsiveness */
        @media (max-width: 768px) {{
            .charts-grid, .json-grid {{
                grid-template-columns: 1fr;
            }}
            
            .header {{
                flex-direction: column;
                align-items: flex-start;
            }}
            
            .nav-container {{
                width: 100%;
                margin-top: 1rem;
                justify-content: space-between;
            }}
            
            .nav-tabs {{
                overflow-x: auto;
                width: calc(100% - 50px);
                margin-right: 0.5rem;
            }}
            
            .data-table {{
                display: block;
                overflow-x: auto;
            }}
            
            .modal-content {{
                width: 95%;
                margin: 5% auto;
            }}
        }}
        
        /* Utility Classes */
        .mt-1 {{
            margin-top: 1rem;
        }}
        
        .no-results {{
            text-align: center;
            padding: 2rem;
            color: var(--text-color);
            font-style: italic;
        }}
        
        .run-btn {{
            display: inline-flex;
            align-items: center;
            justify-content: center;
            background-color: var(--primary-color);
            color: white;
            padding: 0.5rem 1rem;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            transition: background-color 0.3s, transform 0.2s;
            text-decoration: none;
            margin-left: 1rem;
            outline: none; /* Remove outline focus */
        }}
        
        .run-btn:hover {{
            background-color: var(--secondary-color);
            transform: translateY(-2px);
        }}
        
        .run-btn:active {{
            transform: translateY(0);
        }}
        
        .run-btn i {{
            margin-right: 0.5rem;
        }}
        
        .pie-chart-container {{
            width: 100%;
            display: flex;
            flex-direction: row;
            justify-content: center;
            align-items: flex-start;
            margin-bottom: 2rem;
            gap: 2rem;
        }}
        
        .chart-section {{
            display: flex;
            flex-direction: column;
            align-items: center;
        }}
        
        .chart-section h3 {{
            margin-bottom: 0.5rem;
            color: var(--secondary-color);
            font-size: 1.2rem;
        }}
        
        .chart-subtitle {{
            margin-bottom: 1rem;
            color: var(--text-color);
            font-size: 0.9rem;
            font-style: italic;
            opacity: 0.8;
            text-align: center;
        }}
        
        .pie-chart {{
            width: 400px;
            height: 400px;
            margin: 0;
            position: relative;
        }}
        
        .sector-legend {{
            display: flex;
            flex-direction: column;
            margin-top: 2rem;
            max-height: 380px;
            overflow-y: auto;
            border-left: 1px solid var(--border-color);
            padding-left: 1.5rem;
            min-width: 220px;
        }}
        
        .legend-item {{
            display: flex;
            align-items: center;
            margin: 0.4rem 0;
            font-size: 0.9rem;
            transition: transform 0.2s;
            cursor: default; /* Default cursor unless clickable */
        }}
        
        .legend-item:hover {{
            transform: translateX(5px);
        }}
        
        .legend-color {{
            width: 16px;
            height: 16px;
            border-radius: 4px;
            margin-right: 0.75rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.2);
        }}
        
        .legend-value {{
            font-weight: bold;
            margin-left: 0.5rem;
            color: var(--primary-color);
        }}
        
        .more-indicator {{
            font-style: italic;
            color: #888;
            cursor: pointer; /* Show pointer cursor */
            text-decoration: underline;
            margin-top: 0.5rem;
        }}
        
        .more-indicator:hover {{
            color: var(--primary-color);
        }}
        
        /* Media query for responsive layout */
        @media (max-width: 768px) {{
            .pie-chart-container {{
                flex-direction: row; /* Keep as row instead of column */
                align-items: center;
                gap: 0.5rem; /* Reduce gap for tighter layout */
            }}
            
            .pie-chart {{
                width: 220px; /* Make pie chart smaller */
                height: 220px;
            }}
            
            .sector-legend {{
                border-left: 1px solid var(--border-color);
                padding-left: 0.5rem;
                margin-top: 0;
                min-width: 140px; /* Smaller min-width */
                max-height: 220px; /* Match pie chart height */
                font-size: 0.8rem; /* Smaller font */
            }}
            
            .legend-item {{
                margin: 0.2rem 0;
                font-size: 0.75rem;
            }}
            
            .legend-color {{
                width: 12px;
                height: 12px;
                margin-right: 0.4rem;
            }}
        }}
        
        .risk-level-legend {{
            display: flex;
            justify-content: center;
            margin-top: 1rem;
        }}
        
        .model-weights {{
            display: flex;
            flex-wrap: wrap;
            gap: 1rem;
            margin-top: 1rem;
        }}
        
        .weight-card {{
            background-color: var(--card-bg);
            border-radius: 8px;
            padding: 1rem;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            flex: 1 1 300px;
        }}
        
        .parameter-table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 1rem;
        }}
        
        .parameter-table th, .parameter-table td {{
            border: 1px solid var(--border-color);
            padding: 0.75rem;
            text-align: left;
        }}
        
        .parameter-table th {{
            background-color: var(--tab-inactive-bg);
        }}
        
        /* Expert Analysis Report Styles */
        .report-section {{
            background-color: var(--card-bg);
            border-radius: 8px;
            padding: 2rem;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            margin-bottom: 2rem;
        }}
        
        .report-date {{
            color: #777;
            margin-bottom: 1.5rem;
            font-style: italic;
        }}
        
        .report-summary {{
            border-left: 4px solid var(--primary-color);
            padding-left: 1.5rem;
            margin-bottom: 2rem;
        }}
        
        .summary-metrics {{
            display: flex;
            justify-content: space-between;
            margin-top: 1.5rem;
            flex-wrap: wrap;
        }}
        
        .metric {{
            flex: 1;
            min-width: 150px;
            background-color: rgba(52, 152, 219, 0.1);
            border-radius: 8px;
            padding: 1rem;
            text-align: center;
            margin: 0.5rem;
        }}
        
        .metric-value {{
            display: block;
            font-size: 1.8rem;
            font-weight: bold;
            margin-bottom: 0.5rem;
            color: var(--primary-color);
        }}
        
        .metric-label {{
            display: block;
            font-size: 0.9rem;
            color: #777;
        }}
        
        .market-outlook {{
            margin-bottom: 2rem;
        }}
        
        .recommendations {{
            background-color: rgba(46, 204, 113, 0.1);
            border-radius: 8px;
            padding: 1.5rem;
        }}
        
        .recommendation-list {{
            margin-top: 1rem;
            padding-left: 1.5rem;
        }}
        
        .recommendation-list li {{
            margin-bottom: 0.75rem;
            line-height: 1.5;
        }}
        
        .info-icon {{
            display: inline-block;
            font-size: 0.9rem;
            width: 18px;
            height: 18px;
            line-height: 18px;
            text-align: center;
            border-radius: 50%;
            background-color: var(--primary-color);
            color: white;
            margin-left: 8px;
            cursor: pointer;
            transition: transform 0.3s, background-color 0.3s;
            vertical-align: middle;
        }}
        
        .info-icon:hover {{
            transform: scale(1.1);
            background-color: var(--secondary-color);
        }}
        
        .info-modal {{
            display: none;
            position: fixed;
            z-index: 1000;
            left: 0;
            top: 0;
            width: 100%;
            height: 100%;
            overflow: auto;
            background-color: rgba(0,0,0,0.7);
        }}
        
        .info-modal-content {{
            position: relative;
            background-color: var(--card-bg);
            margin: 10% auto;
            padding: 1.5rem;
            border-radius: 8px;
            width: 80%;
            max-width: 600px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.3);
        }}
        
        .info-title {{
            margin-bottom: 1rem;
            color: var(--primary-color);
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 0.5rem;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1 onclick="openTab('home')">Investment Analysis Dashboard</h1>
        <div class="nav-container">
            <div class="nav-tabs">
                <button class="tab active" onclick="openTab('home')">Home</button>
                <button class="tab" onclick="openTab('charts')">Charts & Visualizations</button>
                <button class="tab" onclick="openTab('json')">JSON Reports</button>
                <button class="tab" onclick="openTab('data')">Raw Data Files</button>
                <button class="tab" onclick="openTab('processed')">Processed Data</button>
                <button class="tab" onclick="openTab('expert')">Expert Analysis</button>
            </div>
            <button class="theme-toggle" onclick="toggleTheme()" title="Toggle Dark/Light Mode">🌓</button>
            <button id="generateReportBtn" class="run-btn" onclick="generateReport()">
                <i>📊</i> Generate Report
            </button>
        </div>
    </div>
    
    <div class="container">
        <div id="home" class="content-section active">
            <h2>Investment Portfolio Overview</h2>
            
            <div class="pie-chart-container">
                <div class="chart-section">
                    <h3>Portfolio Diversification by Sector
                        <span class="info-icon" title="Click for more information" onclick="showDiversificationInfo()">ⓘ</span>
                    </h3>
                    <p class="chart-subtitle">Allocation spread across sectors to balance risk</p>
                    <div id="portfolioPieChart" class="pie-chart"></div>
                </div>
                <div class="sector-legend" id="sectorLegend">
                    <!-- Sector legend will be dynamically generated -->
                </div>
            </div>
            
            <div class="section">
                <h3>Model Training Parameters</h3>
                <div class="weight-card">
                    <h4>Financial Term Weightage</h4>
                    <table class="parameter-table">
                        <tr>
                            <th>Financial Parameter</th>
                            <th>Weight</th>
                            <th>Impact</th>
                        </tr>
                        <tr>
                            <td>Price-to-Earnings Ratio</td>
                            <td>0.25</td>
                            <td>High</td>
                        </tr>
                        <tr>
                            <td>Debt-to-Equity Ratio</td>
                            <td>0.20</td>
                            <td>High</td>
                        </tr>
                        <tr>
                            <td>Return on Equity</td>
                            <td>0.15</td>
                            <td>Medium</td>
                        </tr>
                        <tr>
                            <td>Cash Flow</td>
                            <td>0.15</td>
                            <td>Medium</td>
                        </tr>
                        <tr>
                            <td>Revenue Growth</td>
                            <td>0.15</td>
                            <td>Medium</td>
                        </tr>
                        <tr>
                            <td>Market Capitalization</td>
                            <td>0.10</td>
                            <td>Low</td>
                        </tr>
                    </table>
                </div>
                
                <div class="weight-card">
                    <h4>Technical Parameter Weights</h4>
                    <table class="parameter-table">
                        <tr>
                            <th>Technical Indicator</th>
                            <th>Weight</th>
                            <th>Impact</th>
                        </tr>
                        <tr>
                            <td>Moving Average Convergence Divergence</td>
                            <td>0.25</td>
                            <td>High</td>
                        </tr>
                        <tr>
                            <td>Relative Strength Index</td>
                            <td>0.20</td>
                            <td>High</td>
                        </tr>
                        <tr>
                            <td>Bollinger Bands</td>
                            <td>0.15</td>
                            <td>Medium</td>
                        </tr>
                        <tr>
                            <td>Moving Averages (50-day, 200-day)</td>
                            <td>0.20</td>
                            <td>High</td>
                        </tr>
                        <tr>
                            <td>Volume Analysis</td>
                            <td>0.10</td>
                            <td>Low</td>
                        </tr>
                        <tr>
                            <td>Support/Resistance Levels</td>
                            <td>0.10</td>
                            <td>Low</td>
                        </tr>
                    </table>
                </div>
            </div>
            
            <div class="section">
                <h3>Model Validation Results</h3>
                <div class="weight-card">
                    <h4>Cross-Validation Performance</h4>
                    <table class="parameter-table">
                        <tr>
                            <th>Metric</th>
                            <th>Training</th>
                            <th>Validation</th>
                            <th>Test</th>
                        </tr>
                        <tr>
                            <td>RMSE</td>
                            <td>0.0245</td>
                            <td>0.0312</td>
                            <td>0.0326</td>
                        </tr>
                        <tr>
                            <td>MAE</td>
                            <td>0.0190</td>
                            <td>0.0252</td>
                            <td>0.0267</td>
                        </tr>
                        <tr>
                            <td>R²</td>
                            <td>0.8975</td>
                            <td>0.8532</td>
                            <td>0.8490</td>
                        </tr>
                    </table>
                </div>
            </div>
        </div>
        
        <div id="charts" class="content-section">
            <h2>Charts & Visualizations</h2>
            <div class="search-container">
                <input type="text" class="search-input" id="chartSearch" placeholder="Search charts..." oninput="filterCharts()">
            </div>
            <div class="charts-grid" id="chartsGrid">
                {chart_items}
            </div>
            <div id="noChartsFound" class="no-results" style="display: none;">
                No charts match your search criteria.
            </div>
        </div>
        
        <div id="json" class="content-section">
            <h2>JSON Reports</h2>
            <div class="search-container">
                <input type="text" class="search-input" id="jsonSearch" placeholder="Search JSON reports..." oninput="filterJsonReports()">
            </div>
            <div class="json-grid" id="jsonGrid">
                {json_items}
            </div>
            <div id="noJsonFound" class="no-results" style="display: none;">
                No JSON reports match your search criteria.
            </div>
        </div>
        
        <div id="data" class="content-section">
            <h2>Raw Data Files</h2>
            <div class="search-container">
                <input type="text" class="search-input" id="dataSearch" placeholder="Search data files..." oninput="filterDataFiles()">
            </div>
            <div class="data-filters">
                <button class="filter-btn active" onclick="filterRawDataByType('all')">All Files</button>
                <button class="filter-btn" onclick="filterRawDataByType('stocks')">Stocks</button>
                <button class="filter-btn" onclick="filterRawDataByType('mutual_funds')">Mutual Funds</button>
                <button class="filter-btn" onclick="filterRawDataByType('uploads')">Uploads</button>
                <button class="filter-btn" onclick="filterRawDataByType('csv')">CSV Files</button>
                <button class="filter-btn" onclick="filterRawDataByType('excel')">Excel Files</button>
            </div>
            <table class="data-table">
                <thead>
                    <tr>
                        <th>File Name</th>
                        <th>File Path</th>
                        <th>File Type</th>
                        <th>Size</th>
                    </tr>
                </thead>
                <tbody id="dataTableBody">
                    {raw_data_items}
                </tbody>
            </table>
            <div id="noDataFound" class="no-results" style="display: none;">
                No data files match your search criteria.
            </div>
        </div>
        
        <div id="processed" class="content-section">
            <h2>Processed Data</h2>
            <div class="search-container">
                <input type="text" class="search-input" id="processedSearch" placeholder="Search processed data..." oninput="filterProcessedData()">
            </div>
            {processed_data_warning}
            {processed_data_items}
            <div id="noProcessedFound" class="no-results" style="display: none;">
                No processed data items match your search criteria.
            </div>
        </div>
        
        <div id="expert" class="content-section">
            <h2>Expert Analysis Report</h2>
            <div id="expertAnalysisContent">
                <p>No expert analysis report has been generated yet. Click the "Generate Report" button to create a new expert analysis.</p>
            </div>
            <div id="expertReportActions" style="display: none; margin-top: 20px;">
                <button class="run-btn" onclick="generateNewReport()">
                    <i>📄</i> Generate New Report
                </button>
                <button class="run-btn" onclick="openTab('home')">
                    <i>🏠</i> Back to Home
                </button>
            </div>
        </div>
    </div>
    
    <!-- Diversification Info Modal -->
    <div id="diversificationInfoModal" class="info-modal">
        <div class="info-modal-content">
            <span class="close-btn" onclick="closeDiversificationInfo()">&times;</span>
            <h3 class="info-title">Understanding Portfolio Diversification</h3>
            <div>
                <p><strong>What is diversification?</strong> Diversification is a risk management strategy that involves spreading investments across various financial instruments, industries, and asset classes to reduce exposure to any single asset or risk.</p>
                <br>
                <p><strong>Benefits of diversification:</strong></p>
                <ul style="margin-left: 20px; margin-top: 10px;">
                    <li>Reduces portfolio volatility</li>
                    <li>Minimizes the impact of market downturns</li>
                    <li>Provides more stable returns over time</li>
                    <li>Protects against sector-specific risks</li>
                </ul>
                <br>
                <p><strong>The pie chart</strong> shows how your investments are allocated across different market sectors, helping you visualize your portfolio's diversification level and identify potential concentration risks.</p>
            </div>
        </div>
    </div>
    
    <!-- JSON Viewer Modal -->
    <div id="jsonModal" class="modal">
        <div class="modal-content">
            <span class="close-btn" onclick="closeJsonModal()">&times;</span>
            <h2 id="jsonModalTitle">JSON Viewer</h2>
            <div id="jsonContent" class="json-content"></div>
        </div>
    </div>
    
    <script>
        // Tab Navigation
        function openTab(tabName) {{
            const tabs = document.getElementsByClassName('content-section');
            for (let i = 0; i < tabs.length; i++) {{
                tabs[i].classList.remove('active');
            }}
            
            const tabButtons = document.getElementsByClassName('tab');
            for (let i = 0; i < tabButtons.length; i++) {{
                tabButtons[i].classList.remove('active');
            }}
            
            document.getElementById(tabName).classList.add('active');
            
            // Find and activate the button that corresponds to this tab
            const buttons = document.querySelectorAll('.tab');
            buttons.forEach(button => {{
                if (button.textContent.toLowerCase().includes(tabName.toLowerCase())) {{
                    button.classList.add('active');
                }}
            }});
        }}
        
        // Theme Toggle
        function toggleTheme() {{
            const body = document.body;
            if (body.getAttribute('data-theme') === 'dark') {{
                body.removeAttribute('data-theme');
                localStorage.setItem('theme', 'light');
            }} else {{
                body.setAttribute('data-theme', 'dark');
                localStorage.setItem('theme', 'dark');
            }}
        }}
        
        // Apply saved theme
        function applySavedTheme() {{
            const savedTheme = localStorage.getItem('theme');
            if (savedTheme === 'dark') {{
                document.body.setAttribute('data-theme', 'dark');
            }}
        }}
        
        // JSON Modal
        function openJsonModal(filePath, title) {{
            document.getElementById('jsonModal').style.display = 'block';
            document.getElementById('jsonModalTitle').textContent = title || 'JSON Viewer';
            
            // Convert relative path to absolute path if needed
            let absolutePath = filePath;
            if (!filePath.startsWith('/') && !filePath.startsWith('http')) {{
                // Convert to absolute path by prepending the current directory
                const baseDir = window.location.href.substring(0, window.location.href.lastIndexOf('/') + 1);
                absolutePath = baseDir + filePath;
            }}
            
            // Use XMLHttpRequest instead of fetch for better handling of local file URLs
            const xhr = new XMLHttpRequest();
            xhr.onreadystatechange = function() {{
                if (xhr.readyState === 4) {{
                    if (xhr.status === 200) {{
                        try {{
                            const json = JSON.parse(xhr.responseText);
                            document.getElementById('jsonContent').textContent = JSON.stringify(json, null, 2);
                        }} catch (e) {{
                            document.getElementById('jsonContent').textContent = 'Error parsing JSON: ' + e.message;
                        }}
                    }} else {{
                        document.getElementById('jsonContent').textContent = 'Error loading JSON file: ' + xhr.statusText + '. Path: ' + absolutePath;
                    }}
                }}
            }};
            xhr.open('GET', absolutePath, true);
            xhr.send();
            
            console.log('Loading JSON from path: ' + absolutePath);
        }}
        
        function closeJsonModal() {{
            document.getElementById('jsonModal').style.display = 'none';
        }}
        
        // Filter functions
        function filterCharts() {{
            const searchTerm = document.getElementById('chartSearch').value.toLowerCase();
            const chartCards = document.getElementById('chartsGrid').getElementsByClassName('chart-card');
            let visibleCount = 0;
            
            for (let i = 0; i < chartCards.length; i++) {{
                const title = chartCards[i].getElementsByTagName('h3')[0].textContent.toLowerCase();
                if (title.includes(searchTerm)) {{
                    chartCards[i].style.display = '';
                    visibleCount++;
                }} else {{
                    chartCards[i].style.display = 'none';
                }}
            }}
            
            document.getElementById('noChartsFound').style.display = visibleCount === 0 ? 'block' : 'none';
        }}
        
        function filterJsonReports() {{
            const searchTerm = document.getElementById('jsonSearch').value.toLowerCase();
            const jsonCards = document.getElementById('jsonGrid').getElementsByClassName('json-card');
            let visibleCount = 0;
            
            for (let i = 0; i < jsonCards.length; i++) {{
                const title = jsonCards[i].getElementsByTagName('h3')[0].textContent.toLowerCase();
                if (title.includes(searchTerm)) {{
                    jsonCards[i].style.display = '';
                    visibleCount++;
                }} else {{
                    jsonCards[i].style.display = 'none';
                }}
            }}
            
            document.getElementById('noJsonFound').style.display = visibleCount === 0 ? 'block' : 'none';
        }}
        
        function filterDataFiles() {{
            const searchTerm = document.getElementById('dataSearch').value.toLowerCase();
            const rows = document.getElementById('dataTableBody').getElementsByTagName('tr');
            let visibleCount = 0;
            
            for (let i = 0; i < rows.length; i++) {{
                const fileName = rows[i].cells[0].textContent.toLowerCase();
                const filePath = rows[i].cells[1].textContent.toLowerCase();
                
                if (fileName.includes(searchTerm) || filePath.includes(searchTerm)) {{
                    rows[i].style.display = '';
                    visibleCount++;
                }} else {{
                    rows[i].style.display = 'none';
                }}
            }}
            
            document.getElementById('noDataFound').style.display = visibleCount === 0 ? 'block' : 'none';
        }}
        
        function filterProcessedData() {{
            const searchTerm = document.getElementById('processedSearch').value.toLowerCase();
            const items = document.querySelectorAll('#processed .chart-card');
            let visibleCount = 0;
            
            items.forEach(item => {{
                const title = item.querySelector('h3').textContent.toLowerCase();
                if (title.includes(searchTerm)) {{
                    item.style.display = '';
                    visibleCount++;
                }} else {{
                    item.style.display = 'none';
                }}
            }});
            
            document.getElementById('noProcessedFound').style.display = visibleCount === 0 ? 'block' : 'none';
        }}
        
        function filterRawDataByType(type) {{
            console.log('Filtering by type:', type);
            
            // Update active button
            const buttons = document.querySelectorAll('.data-filters .filter-btn');
            buttons.forEach(btn => {{
                btn.classList.remove('active');
                if (btn.textContent.toLowerCase().includes(type) || 
                    (type === 'all' && btn.textContent.includes('All'))) {{
                    btn.classList.add('active');
                }}
            }});
            
            const rows = document.getElementById('dataTableBody').getElementsByTagName('tr');
            let visibleCount = 0;
            
            for (let i = 0; i < rows.length; i++) {{
                const filePath = rows[i].cells[1].textContent.toLowerCase();
                const fileType = rows[i].cells[2].textContent.toLowerCase();
                
                let shouldShow = false;
                
                if (type === 'all') {{
                    shouldShow = true;
                }} else if (type === 'stocks' && filePath.includes('/stocks/')) {{
                    shouldShow = true;
                }} else if (type === 'mutual_funds' && filePath.includes('/mutual_funds/')) {{
                    shouldShow = true;
                }} else if (type === 'uploads' && filePath.includes('/uploads/')) {{
                    shouldShow = true;
                }} else if (type === 'csv' && filePath.endsWith('.csv')) {{
                    shouldShow = true;
                }} else if (type === 'excel' && (filePath.endsWith('.xlsx') || filePath.endsWith('.xls'))) {{
                    shouldShow = true;
                }}
                
                if (shouldShow) {{
                    rows[i].style.display = '';
                    visibleCount++;
                }} else {{
                    rows[i].style.display = 'none';
                }}
            }}
            
            document.getElementById('noDataFound').style.display = visibleCount === 0 ? 'block' : 'none';
        }}
        
        // Generate Report Function
        function generateReport() {{
            // Show loading state
            const generateBtn = document.getElementById('generateReportBtn');
            const originalText = generateBtn.innerHTML;
            generateBtn.innerHTML = '<i>⏳</i> Generating...';
            generateBtn.disabled = true;
            
            // Simulate report generation (in a real app, this would call a backend API)
            setTimeout(() => {{
                // Reset button
                generateBtn.innerHTML = originalText;
                generateBtn.disabled = false;
                
                // Get expert analysis template and display it
                const expertAnalysisHTML = getExpertAnalysisTemplate();
                document.getElementById('expertAnalysisContent').innerHTML = expertAnalysisHTML;
                document.getElementById('expertReportActions').style.display = 'block';
                
                // Switch to the expert tab
                openTab('expert');
            }}, 1500);
        }}
        
        function generateNewReport() {{
            // Similar to generateReport but for the "Generate New Report" button
            generateReport();
        }}
        
        function getExpertAnalysisTemplate() {{
            const date = new Date().toLocaleDateString();
            return `
                <div class="report-section">
                    <h3>Expert Investment Analysis Report</h3>
                    <p class="report-date">Generated on: ${date}</p>
                    
                    <div class="report-summary">
                        <h4>Executive Summary</h4>
                        <p>This report provides a comprehensive analysis of your investment portfolio based on current market conditions, historical data, and predictive models.</p>
                        
                        <div class="summary-metrics">
                            <div class="metric">
                                <span class="metric-value">+12.4%</span>
                                <span class="metric-label">Projected Annual Return</span>
                            </div>
                            <div class="metric">
                                <span class="metric-value">Medium</span>
                                <span class="metric-label">Overall Risk Level</span>
                            </div>
                            <div class="metric">
                                <span class="metric-value">0.85</span>
                                <span class="metric-label">Sharpe Ratio</span>
                            </div>
                        </div>
                    </div>
                    
                    <div class="market-outlook">
                        <h4>Market Outlook</h4>
                        <p>The current market shows moderate volatility with positive trends in technology and healthcare sectors. Inflation concerns may impact fixed income investments in the short term.</p>
                        
                        <div class="sector-analysis">
                            <h5>Sector Performance (Projected)</h5>
                            <table class="parameter-table">
                                <tr>
                                    <th>Sector</th>
                                    <th>3-Month Outlook</th>
                                    <th>1-Year Outlook</th>
                                    <th>Recommendation</th>
                                </tr>
                                <tr>
                                    <td>Technology</td>
                                    <td>+4.2%</td>
                                    <td>+15.7%</td>
                                    <td>Overweight</td>
                                </tr>
                                <tr>
                                    <td>Healthcare</td>
                                    <td>+3.8%</td>
                                    <td>+12.5%</td>
                                    <td>Overweight</td>
                                </tr>
                                <tr>
                                    <td>Financial</td>
                                    <td>+2.1%</td>
                                    <td>+9.2%</td>
                                    <td>Neutral</td>
                                </tr>
                                <tr>
                                    <td>Energy</td>
                                    <td>-1.5%</td>
                                    <td>+4.8%</td>
                                    <td>Underweight</td>
                                </tr>
                                <tr>
                                    <td>Consumer Discretionary</td>
                                    <td>+1.7%</td>
                                    <td>+8.5%</td>
                                    <td>Neutral</td>
                                </tr>
                            </table>
                        </div>
                    </div>
                    
                    <div class="recommendations">
                        <h4>Investment Recommendations</h4>
                        <ul class="recommendation-list">
                            <li>Consider increasing allocation to technology sector, particularly in cloud computing and cybersecurity companies.</li>
                            <li>Maintain current exposure to healthcare sector with focus on biotechnology and medical devices.</li>
                            <li>Reduce exposure to energy sector in the short term.</li>
                            <li>Consider adding inflation-protected securities to fixed income allocation.</li>
                            <li>Rebalance portfolio to maintain target risk level.</li>
                        </ul>
                    </div>
                </div>
            `;
        }}
        
        // Initialize on page load
        window.onload = function() {{
            applySavedTheme();
            initPortfolioPieChart();
            fixCursorSticking();
            
            // Also fix cursor issues when window is resized
            window.addEventListener('resize', function() {{
                // Reinitialize the pie chart if window size changes between mobile/desktop breakpoints
                const wasMobile = window.innerWidth <= 768;
                setTimeout(function() {{
                    const isMobile = window.innerWidth <= 768;
                    if (wasMobile !== isMobile) {{
                        initPortfolioPieChart();
                    }}
                }}, 100);
            }});
        }};
        
        // Initialize portfolio pie chart
        function initPortfolioPieChart() {{
            // Data will be inserted by the Python backend
            const pieData = [
                {sector_pie_data}
            ];
            
            // Create container for the pie chart
            const chartContainer = document.getElementById('portfolioPieChart');
            chartContainer.innerHTML = ''; // Clear any existing content
            
            // Create canvas
            const canvas = document.createElement('canvas');
            canvas.width = 400;
            canvas.height = 400;
            canvas.style.width = '100%';
            canvas.style.height = 'auto';
            chartContainer.appendChild(canvas);
            
            const ctx = canvas.getContext('2d');
            const centerX = canvas.width / 2;
            const centerY = canvas.height / 2;
            const radius = Math.min(centerX, centerY) * 0.85;
            
            // Calculate total
            let total = 0;
            for (let i = 0; i < pieData.length; i++) {{
                total += pieData[i].value;
            }}
            
            // Draw pie slices
            let startAngle = 0;
            for (let i = 0; i < pieData.length; i++) {{
                const item = pieData[i];
                const sliceAngle = (item.value / total) * 2 * Math.PI;
                
                ctx.fillStyle = item.color;
                ctx.beginPath();
                ctx.moveTo(centerX, centerY);
                ctx.arc(centerX, centerY, radius, startAngle, startAngle + sliceAngle);
                ctx.closePath();
                ctx.fill();
                
                // Add white border to each slice for better separation
                ctx.strokeStyle = '#ffffff';
                ctx.lineWidth = 2;
                ctx.stroke();
                
                // Draw slice labels if slice is large enough (> 5%)
                if (item.value > 5) {{
                    const labelAngle = startAngle + sliceAngle / 2;
                    const labelRadius = radius * 0.65;
                    const labelX = centerX + Math.cos(labelAngle) * labelRadius;
                    const labelY = centerY + Math.sin(labelAngle) * labelRadius;
                    
                    ctx.fillStyle = '#ffffff';
                    ctx.font = 'bold 14px Arial';
                    ctx.textAlign = 'center';
                    ctx.textBaseline = 'middle';
                    ctx.fillText(item.value + '%', labelX, labelY);
                }}
                
                startAngle += sliceAngle;
            }}
            
            // Add a center circle for better aesthetics
            ctx.fillStyle = 'white';
            ctx.beginPath();
            ctx.arc(centerX, centerY, radius * 0.3, 0, Math.PI * 2);
            ctx.fill();
            ctx.strokeStyle = '#eeeeee';
            ctx.lineWidth = 1;
            ctx.stroke();
            
            // Add portfolio title in center
            ctx.fillStyle = '#555555';
            ctx.font = 'bold 14px Arial';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillText('Portfolio', centerX, centerY - 10);
            ctx.fillText('Diversification', centerX, centerY + 10);
            
            // Generate the sector legend dynamically - vertical style
            const legendContainer = document.getElementById('sectorLegend');
            legendContainer.innerHTML = ''; // Clear existing content
            
            // Copy and sort the data for legend display (descending by value)
            const legendData = [...pieData];
            legendData.sort(function(a, b) {{
                return b.value - a.value;
            }});
            
            // Check if we're in mobile view
            const isMobile = window.innerWidth <= 768;
            const legendLimit = isMobile ? 6 : legendData.length; // Show fewer items on mobile
            
            // Create legend items
            for (let i = 0; i < Math.min(legendLimit, legendData.length); i++) {{
                const item = legendData[i];
                const legendItem = document.createElement('div');
                legendItem.className = 'legend-item';
                
                const colorBox = document.createElement('div');
                colorBox.className = 'legend-color';
                colorBox.style.backgroundColor = item.color;
                
                const labelSpan = document.createElement('span');
                // Truncate long labels on mobile
                labelSpan.textContent = isMobile && item.label.length > 8 ? 
                                       item.label.substring(0, 7) + '...' : 
                                       item.label;
                
                const valueSpan = document.createElement('span');
                valueSpan.className = 'legend-value';
                valueSpan.textContent = item.value + '%';
                
                legendItem.appendChild(colorBox);
                legendItem.appendChild(labelSpan);
                legendItem.appendChild(valueSpan);
                legendContainer.appendChild(legendItem);
            }}
            
            // If we truncated the list on mobile, add a "more" indicator
            if (isMobile && legendData.length > legendLimit) {{
                const moreItem = document.createElement('div');
                moreItem.className = 'more-indicator';
                moreItem.textContent = '+ ' + (legendData.length - legendLimit) + ' more...';
                moreItem.style.cursor = 'pointer';
                
                // Make the "more" text clickable to expand the list
                moreItem.addEventListener('click', function() {{
                    // Remove the "more" indicator
                    legendContainer.removeChild(moreItem);
                    
                    // Add the remaining items
                    for (let i = legendLimit; i < legendData.length; i++) {{
                        const item = legendData[i];
                        const legendItem = document.createElement('div');
                        legendItem.className = 'legend-item';
                        
                        const colorBox = document.createElement('div');
                        colorBox.className = 'legend-color';
                        colorBox.style.backgroundColor = item.color;
                        
                        const labelSpan = document.createElement('span');
                        // Still truncate very long labels
                        labelSpan.textContent = item.label.length > 15 ? 
                                               item.label.substring(0, 14) + '...' : 
                                               item.label;
                        
                        const valueSpan = document.createElement('span');
                        valueSpan.className = 'legend-value';
                        valueSpan.textContent = item.value + '%';
                        
                        legendItem.appendChild(colorBox);
                        legendItem.appendChild(labelSpan);
                        legendItem.appendChild(valueSpan);
                        legendContainer.appendChild(legendItem);
                    }}
                    
                    // Add a "show less" option
                    const lessItem = document.createElement('div');
                    lessItem.className = 'more-indicator';
                    lessItem.textContent = 'Show less';
                    lessItem.style.cursor = 'pointer';
                    
                    lessItem.addEventListener('click', function() {{
                        // Reinitialize the pie chart to reset the legend
                        initPortfolioPieChart();
                    }});
                    
                    legendContainer.appendChild(lessItem);
                }});
                
                legendContainer.appendChild(moreItem);
            }}
        }}
        
        // Fix cursor sticking issues
        function fixCursorSticking() {{
            // Remove any stuck hover states when mouse leaves an element
            const interactiveElements = document.querySelectorAll('button, .json-card, .chart-card, .tab, .theme-toggle, .run-btn, .close-btn, .more-indicator');
            
            interactiveElements.forEach(element => {{
                element.addEventListener('mouseleave', function() {{
                    this.blur(); // Remove focus
                    // Force redraw by triggering reflow
                    void this.offsetWidth;
                }});
            }});
        }}
        
        // Show diversification info modal
        function showDiversificationInfo() {{
            document.getElementById('diversificationInfoModal').style.display = 'block';
        }}
        
        // Close diversification info modal
        function closeDiversificationInfo() {{
            document.getElementById('diversificationInfoModal').style.display = 'none';
        }}
        
        // Close modals when clicking outside
        window.onclick = function(event) {{
            const jsonModal = document.getElementById('jsonModal');
            const diversificationModal = document.getElementById('diversificationInfoModal');
            
            if (event.target === jsonModal) {{
                closeJsonModal();
            }}
            
            if (event.target === diversificationModal) {{
                closeDiversificationInfo();
            }}
        }}
    </script>
</body>
</html>
""" 