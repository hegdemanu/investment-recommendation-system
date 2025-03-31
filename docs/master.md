# 📊 Master Documentation

This document maintains the current requirements, technology stack, and software used in the Investment Recommendation System.

## 🛠️ Technology Stack

### Frontend
- **Framework:** React.js v19.x + Next.js v15.x
- **Language:** TypeScript v5.3+
- **Styling:** Tailwind CSS v4.x + Radix UI v2.x
- **State Management:** React Query v5.x, Zustand v5.x
- **Visualization:** Chart.js v5.x, TradingView Lightweight Charts v4.x
- **Forms:** React Hook Form v8.x, Zod v4.x

### Backend
- **Node.js:** v20.x LTS
- **Express.js:** v5.x
- **Python:** v3.12+
- **FastAPI:** v1.x
- **WebSockets:** Socket.io v5.x

### Database
- **PostgreSQL:** v16.x
- **TimescaleDB:** v3.x (PostgreSQL extension)
- **MongoDB:** v7.x (optional for flexible schema requirements)
- **Redis:** v8.x (for caching)

### Authentication
- **NextAuth.js:** v5.x / Auth0

### DevOps
- **Docker:** v25.x
- **Docker Compose:** v3.x
- **GitHub Actions:** CI/CD pipeline

### AI/ML
- **Python Libraries:**
  - NumPy v2.0+
  - Pandas v3.0+
  - Scikit-learn v1.5+
  - TensorFlow v3.0+ / PyTorch v2.5+
  - Prophet v1.2+
  - Statsmodels v0.15+ (for ARIMA)
- **Hugging Face Transformers:** v5.0+

## 📈 Data Sources

- **Alpha Vantage:** Financial data API
- **Yahoo Finance:** Market data and historical prices
- **NewsAPI:** News articles for sentiment analysis
- **FRED API:** Economic indicators
- **Finnhub:** Real-time market data

## 🔐 Security Requirements

- **Authentication:** OAuth 2.0 / JWT based
- **Authorization:** Role-Based Access Control (RBAC)
- **Data Encryption:** End-to-end for sensitive user data
- **API Security:** Rate limiting, HTTPS-only

## 📝 Compliance Requirements

- **GDPR:** Data privacy and protection
- **SEBI Guidelines:** Regulatory compliance for financial trading
- **PCI-DSS:** If handling payment data

## 🧪 Testing Framework

- **Backend:**
  - Jest for Node.js
  - Pytest for Python
- **Frontend:**
  - Jest + React Testing Library
  - Cypress for E2E testing

---

## 🔄 Version Control
**Version:** 2.0.0
**Last Updated:** 2025-05-28 