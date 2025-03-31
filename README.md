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
- **NLP:** TensorFlow.js
- **Market Analysis:** Python, Scikit-learn
- **Data Processing:** Pandas, NumPy
- **Model Deployment:** TensorFlow Serving

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

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Alpha Vantage API](https://www.alphavantage.co/) for financial data
- [TensorFlow](https://www.tensorflow.org/) for ML capabilities
- [Shadcn UI](https://ui.shadcn.com/) for beautiful components

