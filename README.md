# InvestSage AI - Investment Recommendation System

A comprehensive, full-stack investment recommendation system that leverages AI to provide data-driven investment insights. Built with modern technologies and best practices in software development.

![Investment Dashboard](./docs/landing.png)

## 🎥 Demo

Check out our [demo video](investment-system-demo.mp4) to see the system in action, featuring:
- Interactive Compound Calculator
- Real-time Market Sentiment Analysis
- AI-Powered Report Generation
- RAG-based Investment Research

## 🚀 Key Features

### Frontend
- 📊 Interactive Compound Interest Calculator
- 📈 Real-time Market Sentiment Analysis
- 📑 AI-Powered Report Generation
- 🤖 RAG-based Investment Research Assistant
- 🌓 Dark/Light Mode Support
- 📱 Fully Responsive Design

### Backend
- 🔄 Real-time Market Data Integration
- 🧠 Machine Learning Models for Market Analysis
- 🔒 Secure API Architecture
- 📦 Efficient Data Caching
- 🔍 Advanced Search Capabilities
- 📊 Data Analytics Pipeline

## 🧠 ML Models Architecture

Our investment recommendation system leverages state-of-the-art machine learning models to deliver accurate predictions and insights:

### 🔮 Time-Series Prediction Models
- **LSTM Networks**: Long Short-Term Memory networks that capture long-term dependencies in market data
- **Bidirectional GRU**: Bidirectional Gated Recurrent Units for enhanced pattern recognition
- **Hybrid CNN-LSTM Models**: Combining convolutional layers for feature extraction with LSTM layers for sequence prediction
- **Multi-Horizon Forecast Models**: Specialized models for short-term (1-7 days), medium-term (1-4 weeks), and long-term (1-12 months) predictions

### 📊 Feature Engineering Pipeline
- **Technical Indicators**: Automated extraction of 50+ technical indicators (RSI, MACD, Bollinger Bands, etc.)
- **Market Sentiment Features**: Integration of news sentiment, social media metrics, and market mood indicators
- **Macroeconomic Factors**: Incorporation of interest rates, inflation metrics, sector performance, and global market indices
- **Volatility Estimators**: Custom volatility measures using GARCH models and implied volatility from options markets

### 🔍 Model Training and Evaluation
- **Bayesian Hyperparameter Optimization**: Automated hyperparameter tuning using Bayesian optimization
- **Multi-Metric Evaluation**: Models evaluated on RMSE, MAE, MAPE, Sharpe ratio, and custom directional accuracy metrics
- **Ensemble Methods**: Stacking and blending multiple models for improved prediction accuracy
- **Backtesting Framework**: Rigorous historical backtesting against multiple market regimes
- **Continuous Learning**: Models automatically retrain on new data to adapt to changing market conditions

### 💹 Risk Assessment Models
- **VaR Estimators**: Value-at-Risk models using historical simulation, parametric, and Monte Carlo methods
- **Portfolio Optimization**: Efficient frontier calculation using Modern Portfolio Theory
- **Stress Testing**: Scenario analysis for market crashes, interest rate spikes, and sector-specific shocks
- **Risk-Adjusted Return Prediction**: Models that specifically target risk-adjusted returns like Sharpe and Sortino ratios

### 🧩 Significance of Our ML Approach
- **Multi-resolution Analysis**: Separate models for different investment horizons, enabling both tactical and strategic decisions
- **Explainable AI**: Feature importance analysis and partial dependence plots to understand model decisions
- **Uncertainty Quantification**: Confidence intervals and prediction distributions rather than point estimates
- **Regime Detection**: Automatic identification of bullish, bearish, and sideways market conditions
- **Asset-Specific Optimization**: Models fine-tuned for different asset classes (equities, fixed income, commodities)

## 🛠️ Tech Stack

### Frontend
- **Framework:** Next.js 14 with TypeScript
- **Styling:** Tailwind CSS, Shadcn UI
- **State Management:** React Hooks
- **Data Visualization:** Chart.js
- **API Integration:** Axios, SWR

### Backend
- **Server:** Node.js, Express.js
- **Database:** PostgreSQL with TypeORM
- **Authentication:** JWT, OAuth2
- **API Documentation:** Swagger/OpenAPI
- **Caching:** Redis

### AI/ML Components
- **NLP:** TensorFlow.js, Transformers, BERT
- **Market Analysis:** Python, Scikit-learn, PyTorch
- **Data Processing:** Pandas, NumPy, Dask for large-scale processing
- **Model Deployment:** TensorFlow Serving, ONNX Runtime
- **Feature Store:** Feast for feature management and serving
- **Model Monitoring:** MLflow for tracking experiments and model versioning
- **Distributed Training:** Horovod for multi-GPU training of large models
- **AutoML:** Auto-Keras and TPOT for automated architecture search

## 🖥️ Screenshots

### Compound Calculator
![Compound Calculator](./docs/compound-calculator.png)
*Interactive compound interest calculator with visualization*

### Market Sentiment Analysis
![Market Sentiment](./docs/sentiment-analysis.png)
*Real-time market sentiment analysis dashboard*

### Report Generator
![Report Generator](./docs/report-generator.png)
*AI-powered investment report generation*

### AI Research Dashboard
![AI Research](./docs/ai-research.png)
*RAG-based investment research assistant*

## 🚀 Getting Started

### Prerequisites
- Node.js 18+
- PostgreSQL 14+
- Python 3.8+ (for ML models)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/investment-recommendation-system.git
cd investment-recommendation-system
```

2. Install dependencies:
```bash
# Frontend
cd frontend-new
npm install

# Backend
cd ../backend
npm install
```

3. Set up environment variables:
```bash
# Frontend (.env.local)
NEXT_PUBLIC_API_URL=http://localhost:3001

# Backend (.env)
DATABASE_URL=postgresql://user:password@localhost:5432/investsage
JWT_SECRET=your_jwt_secret
```

4. Start the development servers:
```bash
# Frontend (http://localhost:3000)
npm run dev

# Backend (http://localhost:3001)
npm run dev
```

## 📁 Project Structure

```
investment-recommendation-system/
├── frontend-new/           # Next.js frontend
│   ├── src/
│   │   ├── app/           # Pages and layouts
│   │   ├── components/    # React components
│   │   └── lib/          # Utilities and hooks
├── backend/               # Node.js backend
│   ├── src/
│   │   ├── controllers/  # Route controllers
│   │   ├── models/      # Database models
│   │   └── services/    # Business logic
├── ml/                   # Machine learning models
│   ├── sentiment/       # Market sentiment analysis
│   └── prediction/      # Price prediction models
└── docs/                # Documentation
    ├── images/         # Screenshots
    └── videos/         # Demo videos
```

## 🔑 Key Implementation Details

### Frontend
- Implemented responsive UI with Tailwind CSS and Shadcn UI
- Real-time data updates using SWR
- Client-side form validation
- Optimized performance with Next.js 14 features

### Backend
- RESTful API design
- Database optimization with TypeORM
- Caching layer for improved performance
- Comprehensive error handling
- Unit and integration tests

### AI/ML
- Sentiment analysis using TensorFlow.js
- Market trend prediction models
- RAG implementation for research queries
- Data preprocessing pipeline

## 📈 Future Enhancements

- Portfolio optimization algorithms
- Real-time market alerts
- Social trading features
- Mobile app development
- Advanced ML model integration

## 📸 Complete Screenshot Gallery

### Main Features
![Compound Calculator](./docs/compound-calculator.png)
*Interactive compound interest calculator with visualization*

![Sentiment Analysis](./docs/sentiment-analysis.png)
*Real-time market sentiment analysis dashboard*

![Report Generator](./docs/report-generator.png)
*AI-powered investment report generation*

![RAG Research](./docs/ai-research.png)
*RAG-based investment research assistant*

### Interface Components
![Landing Page](./docs/landing.png)
*Main landing page showcasing key application features*

![Dashboard Preview](./docs/images/dashboard_preview.png)
*Comprehensive dashboard view for monitoring investments*

### Additional Screenshots
![Dashboard View 1](./docs/images/Screenshot_2025-03-26_at_4.51.25_AM.png)
*Portfolio performance tracking with interactive charts*

![Dashboard View 2](./docs/images/Screenshot_2025-03-26_at_4.51.32_AM.png)
*Stock comparison tool with technical indicators*

![Dashboard View 3](./docs/images/Screenshot_2025-03-26_at_4.51.43_AM.png)
*Risk analysis view with volatility metrics*

![Dashboard View 4](./docs/images/Screenshot_2025-03-26_at_4.51.53_AM.png)
*Market sentiment trends across different sectors*

![Dashboard View 5](./docs/images/Screenshot_2025-03-26_at_4.52.01_AM.png)
*Historical performance analytics with custom date ranges*

![Dashboard View 6](./docs/images/Screenshot_2025-03-26_at_4.52.09_AM.png)
*Portfolio allocation visualization with asset breakdown*

![Dashboard View 7](./docs/images/Screenshot_2025-03-26_at_4.52.17_AM.png)
*Customizable watchlist with real-time updates*

![Dashboard View 8](./docs/images/Screenshot_2025-03-26_at_4.52.26_AM.png)
*Technical analysis tools with pattern recognition*

![Dashboard View 9](./docs/images/Screenshot_2025-03-26_at_4.52.45_AM.png)
*News sentiment integration with market impact scoring*

![Dashboard View 10](./docs/images/Screenshot_2025-03-26_at_4.52.55_AM.png)
*Recommendation engine with personalized investment insights*

> **Note:** These screenshots showcase the current state of the application. Additional screenshots are being continuously added as new features are developed. Check back regularly for updates.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Alpha Vantage API](https://www.alphavantage.co/) for financial data
- [TensorFlow](https://www.tensorflow.org/) for ML capabilities
- [Shadcn UI](https://ui.shadcn.com/) for beautiful components

