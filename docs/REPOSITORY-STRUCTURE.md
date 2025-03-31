# 📂 Repository Structure

This document outlines the organization and structure of the Investment Recommendation System codebase.

## 🗂️ Root Folder Structure

```
/investment-recommendation-system
├── /archive              # Deprecated/archived code with versioning
├── /backend              # FastAPI (Python) backend code, APIs, and endpoints
├── /backend-express      # Express.js backend API server
├── /client               # Legacy client code
├── /data_pipeline        # Data ingestion, preprocessing, and API integrations
├── /docs                 # All documentation
├── /frontend             # Legacy frontend code
├── /frontend-new         # Next.js (TypeScript) frontend with Tailwind CSS
├── /scripts              # CI/CD, automation, and DevOps-related scripts
└── /trading_engine       # AI models, order execution logic, and trading strategies
```

## 📁 Directory Details

### `/archive`
Stores deprecated/archived code with versioning and logs to maintain history of past implementations.

### `/backend`
Contains the FastAPI Python backend responsible for AI/ML models, data processing, and specialized endpoints for machine learning operations.

### `/backend-express`
Houses the primary Node.js/Express API server that handles authentication, business logic, and serves as a gateway to the Python ML services.

### `/client`
Legacy client-side code that is pending archival once the new frontend is fully implemented.

### `/data_pipeline`
Manages all aspects of data ingestion, transformation, and storage including:
- API connectors for financial data sources
- ETL processes for market data
- News and sentiment data collection
- Preprocessing pipelines for ML models

### `/docs`
Central repository for all documentation including:
- System architecture diagrams
- API specifications
- User guides
- Development guidelines
- Changelog and versioning information

### `/frontend`
Legacy frontend code that is pending archival once the new frontend is fully implemented.

### `/frontend-new`
Modern React/Next.js implementation with TypeScript and Tailwind CSS, featuring:
- Dashboard components
- Visualization tools
- User authentication interfaces
- Portfolio management views

### `/scripts`
Utility scripts for automation, deployment, database migration, and other DevOps operations.

### `/trading_engine`
Core algorithms and models for financial analysis and trading strategies:
- LSTM, ARIMA, and Prophet model implementations
- Backtesting frameworks
- Order execution logic
- Portfolio optimization algorithms

## 📝 File Naming Conventions

- **Component files**: PascalCase (e.g., `DashboardWidget.tsx`)
- **Utility files**: camelCase (e.g., `dataFormatter.js`)
- **CSS modules**: camelCase with `.module.css` suffix (e.g., `button.module.css`)
- **Python files**: snake_case (e.g., `data_processor.py`)
- **Test files**: Same name as the file being tested with `.test` or `.spec` suffix (e.g., `UserAuth.test.js`)

## 🔄 Version Control Practices

1. Use meaningful commit messages that clearly explain the purpose of changes
2. Branch naming format: `type/description` (e.g., `feature/portfolio-optimization`, `fix/authentication-bug`)
3. Create Pull Requests for all significant changes
4. Ensure CI/CD checks pass before merging code 