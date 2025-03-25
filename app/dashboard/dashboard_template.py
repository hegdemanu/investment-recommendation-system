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
        }}
        
        .json-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        }}
        
        .json-card h3 {{
            margin-bottom: 0.5rem;
            color: var(--primary-color);
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
            transition: background-color 0.3s;
            color: var(--text-color);
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
            transition: background-color 0.3s;
            text-decoration: none;
            margin-left: 1rem;
        }}
        
        .run-btn:hover {{
            background-color: var(--secondary-color);
        }}
        
        .run-btn i {{
            margin-right: 0.5rem;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Investment Analysis Dashboard</h1>
        <div class="nav-container">
            <div class="nav-tabs">
                <button class="tab active" onclick="openTab('charts')">Charts & Visualizations</button>
                <button class="tab" onclick="openTab('json')">JSON Reports</button>
                <button class="tab" onclick="openTab('data')">Raw Data Files</button>
                <button class="tab" onclick="openTab('processed')">Processed Data</button>
            </div>
            <button class="theme-toggle" onclick="toggleTheme()" title="Toggle Dark/Light Mode">🌓</button>
            <button id="generateReportBtn" class="run-btn" onclick="generateReport()">
                <i>📊</i> Generate Report
            </button>
        </div>
    </div>
    
    <div class="container">
        <div id="charts" class="content-section active">
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
                        document.getElementById('jsonContent').textContent = 'Error loading JSON file: ' + xhr.statusText;
                    }}
                }}
            }};
            xhr.open('GET', filePath, true);
            xhr.send();
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
            // Try to use form-based submission for better compatibility
            const form = document.createElement('form');
            form.method = 'GET';
            form.action = 'run_app.py';
            document.body.appendChild(form);
            
            try {{
                form.submit();
            }} catch (e) {{
                console.error('Form submission failed:', e);
                // Fallback to window.open
                window.open('run_app.py', '_blank');
            }}
        }}
        
        // Initialize on page load
        window.onload = function() {{
            applySavedTheme();
        }};
        
        // Close JSON modal when clicking outside
        window.onclick = function(event) {{
            const modal = document.getElementById('jsonModal');
            if (event.target === modal) {{
                closeJsonModal();
            }}
        }}
    </script>
</body>
</html>
""" 