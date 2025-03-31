import { useState, useEffect } from 'react';
import { useParams } from 'react-router-dom';
import axios from 'axios';

export default function StockDetails() {
  const { symbol } = useParams();
  const [stockData, setStockData] = useState(null);
  const [technicalAnalysis, setTechnicalAnalysis] = useState(null);
  const [forecastData, setForecastData] = useState(null);
  const [sentimentData, setSentimentData] = useState(null);
  const [recommendation, setRecommendation] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [activeTab, setActiveTab] = useState('overview');

  useEffect(() => {
    async function fetchStockData() {
      try {
        setLoading(true);
        
        // Fetch basic stock data
        const stockResponse = await axios.get(`/api/stocks/${symbol}`);
        setStockData(stockResponse.data);
        
        // Fetch technical analysis
        const technicalResponse = await axios.get(`/api/analysis/technical/${symbol}`);
        setTechnicalAnalysis(technicalResponse.data);
        
        // Fetch forecast data
        const forecastResponse = await axios.get(`/api/analysis/forecast/${symbol}`);
        setForecastData(forecastResponse.data);
        
        // Fetch sentiment data
        const sentimentResponse = await axios.get(`/api/analysis/sentiment/${symbol}`);
        setSentimentData(sentimentResponse.data);
        
        // Fetch integrated recommendation
        const recommendationResponse = await axios.get(`/api/recommendation/${symbol}`);
        setRecommendation(recommendationResponse.data);
        
        setError('');
      } catch (err) {
        setError(`Failed to load data for ${symbol}`);
        console.error(err);
      } finally {
        setLoading(false);
      }
    }

    fetchStockData();
  }, [symbol]);

  // Helper function to render the recommendation status
  const renderRecommendationStatus = () => {
    if (!recommendation) return null;
    
    const { recommendation_type, confidence_score } = recommendation;
    let bgColor = 'bg-gray-200';
    let textColor = 'text-gray-800';
    
    if (recommendation_type === 'BUY') {
      bgColor = 'bg-green-100';
      textColor = 'text-green-800';
    } else if (recommendation_type === 'SELL') {
      bgColor = 'bg-red-100';
      textColor = 'text-red-800';
    } else if (recommendation_type === 'HOLD') {
      bgColor = 'bg-yellow-100';
      textColor = 'text-yellow-800';
    }
    
    return (
      <div className={`${bgColor} ${textColor} rounded-md px-4 py-2 inline-block font-semibold`}>
        {recommendation_type} ({confidence_score.toFixed(2)}%)
      </div>
    );
  };

  if (loading) {
    return (
      <div className="flex justify-center items-center h-64">
        <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-blue-500"></div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="text-center py-10">
        <h2 className="text-2xl font-bold text-red-600 mb-4">Error</h2>
        <p className="text-gray-700">{error}</p>
      </div>
    );
  }

  return (
    <div className="max-w-6xl mx-auto">
      {/* Stock Header */}
      {stockData && (
        <div className="bg-white rounded-lg shadow-md p-6 mb-6">
          <div className="flex flex-wrap justify-between items-start">
            <div>
              <h1 className="text-3xl font-bold text-gray-800">{stockData.name} ({stockData.symbol})</h1>
              <p className="text-gray-600 mt-1">{stockData.exchange}</p>
            </div>
            <div className="text-right">
              <p className="text-3xl font-bold">${stockData.price.toFixed(2)}</p>
              <p className={`font-semibold ${stockData.change >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                {stockData.change >= 0 ? '+' : ''}{stockData.change.toFixed(2)}% 
                <span className="text-gray-500 text-sm ml-1">Today</span>
              </p>
            </div>
          </div>
          
          <div className="mt-6">
            <h2 className="text-xl font-semibold mb-2">Recommendation</h2>
            {renderRecommendationStatus()}
          </div>
        </div>
      )}

      {/* Tabs */}
      <div className="mb-6 border-b border-gray-200">
        <div className="flex flex-wrap -mb-px">
          <button
            className={`mr-2 py-2 px-4 font-medium ${
              activeTab === 'overview' 
                ? 'text-blue-600 border-b-2 border-blue-600' 
                : 'text-gray-500 hover:text-gray-700'
            }`}
            onClick={() => setActiveTab('overview')}
          >
            Overview
          </button>
          <button
            className={`mr-2 py-2 px-4 font-medium ${
              activeTab === 'technical' 
                ? 'text-blue-600 border-b-2 border-blue-600' 
                : 'text-gray-500 hover:text-gray-700'
            }`}
            onClick={() => setActiveTab('technical')}
          >
            Technical Analysis
          </button>
          <button
            className={`mr-2 py-2 px-4 font-medium ${
              activeTab === 'forecast' 
                ? 'text-blue-600 border-b-2 border-blue-600' 
                : 'text-gray-500 hover:text-gray-700'
            }`}
            onClick={() => setActiveTab('forecast')}
          >
            Price Forecast
          </button>
          <button
            className={`mr-2 py-2 px-4 font-medium ${
              activeTab === 'sentiment' 
                ? 'text-blue-600 border-b-2 border-blue-600' 
                : 'text-gray-500 hover:text-gray-700'
            }`}
            onClick={() => setActiveTab('sentiment')}
          >
            Market Sentiment
          </button>
        </div>
      </div>

      {/* Tab Content */}
      <div className="bg-white rounded-lg shadow-md p-6">
        {activeTab === 'overview' && recommendation && (
          <div>
            <h2 className="text-xl font-semibold mb-4">Investment Overview</h2>
            <div className="mb-6">
              <h3 className="text-lg font-medium mb-2">Summary</h3>
              <p className="text-gray-700">{recommendation.summary}</p>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="bg-gray-50 p-4 rounded-lg">
                <h3 className="font-medium text-gray-800 mb-2">Technical Signals</h3>
                <p className={`font-semibold ${recommendation.technical_score >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                  {recommendation.technical_score >= 0 ? 'Positive' : 'Negative'} 
                  ({Math.abs(recommendation.technical_score).toFixed(2)})
                </p>
              </div>
              <div className="bg-gray-50 p-4 rounded-lg">
                <h3 className="font-medium text-gray-800 mb-2">Price Forecast</h3>
                <p className={`font-semibold ${recommendation.forecast_trend === 'UP' ? 'text-green-600' : 'text-red-600'}`}>
                  {recommendation.forecast_trend} 
                  ({recommendation.forecast_confidence.toFixed(2)}%)
                </p>
              </div>
              <div className="bg-gray-50 p-4 rounded-lg">
                <h3 className="font-medium text-gray-800 mb-2">Market Sentiment</h3>
                <p className={`font-semibold ${recommendation.sentiment_score >= 0.5 ? 'text-green-600' : 'text-red-600'}`}>
                  {recommendation.sentiment_score >= 0.5 ? 'Positive' : 'Negative'} 
                  ({(recommendation.sentiment_score * 100).toFixed(2)}%)
                </p>
              </div>
            </div>
          </div>
        )}

        {activeTab === 'technical' && technicalAnalysis && (
          <div>
            <h2 className="text-xl font-semibold mb-4">Technical Analysis</h2>
            <div className="mb-6">
              <h3 className="text-lg font-medium mb-2">Indicators</h3>
              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
                {Object.entries(technicalAnalysis.indicators).map(([key, value]) => (
                  <div key={key} className="bg-gray-50 p-4 rounded-lg">
                    <h4 className="font-medium text-gray-800">{key.replace('_', ' ').toUpperCase()}</h4>
                    <p className="text-gray-700">{value.toFixed(2)}</p>
                  </div>
                ))}
              </div>
              
              <h3 className="text-lg font-medium mb-2">Signals</h3>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                {Object.entries(technicalAnalysis.signals).map(([key, value]) => {
                  let bgColor = 'bg-gray-100';
                  let textColor = 'text-gray-800';
                  
                  if (value === 'BUY') {
                    bgColor = 'bg-green-100';
                    textColor = 'text-green-800';
                  } else if (value === 'SELL') {
                    bgColor = 'bg-red-100';
                    textColor = 'text-red-800';
                  } else if (value === 'HOLD') {
                    bgColor = 'bg-yellow-100';
                    textColor = 'text-yellow-800';
                  }
                  
                  return (
                    <div key={key} className={`${bgColor} p-4 rounded-lg`}>
                      <h4 className="font-medium text-gray-800">{key.replace('_', ' ').toUpperCase()}</h4>
                      <p className={`font-semibold ${textColor}`}>{value}</p>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>
        )}

        {activeTab === 'forecast' && forecastData && (
          <div>
            <h2 className="text-xl font-semibold mb-4">Price Forecast</h2>
            <div className="mb-6">
              <h3 className="text-lg font-medium mb-3">Forecast Summary</h3>
              <p className="text-gray-700 mb-4">{forecastData.summary}</p>
              
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
                <div className="bg-blue-50 p-4 rounded-lg">
                  <h4 className="font-medium text-gray-800">Expected Price (7 days)</h4>
                  <p className="text-xl font-semibold text-blue-600">${forecastData.forecast_price_7d.toFixed(2)}</p>
                  <p className={`text-sm ${forecastData.price_change_7d >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                    {forecastData.price_change_7d >= 0 ? '+' : ''}{forecastData.price_change_7d.toFixed(2)}%
                  </p>
                </div>
                <div className="bg-blue-50 p-4 rounded-lg">
                  <h4 className="font-medium text-gray-800">Expected Price (30 days)</h4>
                  <p className="text-xl font-semibold text-blue-600">${forecastData.forecast_price_30d.toFixed(2)}</p>
                  <p className={`text-sm ${forecastData.price_change_30d >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                    {forecastData.price_change_30d >= 0 ? '+' : ''}{forecastData.price_change_30d.toFixed(2)}%
                  </p>
                </div>
                <div className="bg-blue-50 p-4 rounded-lg">
                  <h4 className="font-medium text-gray-800">Forecast Confidence</h4>
                  <p className="text-xl font-semibold text-blue-600">{forecastData.confidence.toFixed(2)}%</p>
                </div>
              </div>
              
              <div className="bg-gray-50 p-4 rounded-lg">
                <h4 className="font-medium text-gray-800 mb-2">Factors Influencing Forecast</h4>
                <ul className="list-disc pl-5 text-gray-700">
                  {forecastData.factors.map((factor, index) => (
                    <li key={index}>{factor}</li>
                  ))}
                </ul>
              </div>
            </div>
          </div>
        )}

        {activeTab === 'sentiment' && sentimentData && (
          <div>
            <h2 className="text-xl font-semibold mb-4">Market Sentiment Analysis</h2>
            
            <div className="mb-6">
              <h3 className="text-lg font-medium mb-2">Sentiment Overview</h3>
              <div className="flex items-center mb-4">
                <div className="w-full bg-gray-200 rounded-full h-5">
                  <div 
                    className="bg-blue-600 h-5 rounded-full" 
                    style={{ width: `${sentimentData.sentiment_score * 100}%` }}
                  ></div>
                </div>
                <span className="ml-4 font-semibold">{(sentimentData.sentiment_score * 100).toFixed(2)}%</span>
              </div>
              
              <p className="text-gray-700 mb-6">{sentimentData.summary}</p>
              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
                <div className="bg-green-50 p-4 rounded-lg">
                  <h4 className="font-medium text-gray-800">Positive Mentions</h4>
                  <p className="text-xl font-semibold text-green-600">{sentimentData.positive_mentions}</p>
                </div>
                <div className="bg-red-50 p-4 rounded-lg">
                  <h4 className="font-medium text-gray-800">Negative Mentions</h4>
                  <p className="text-xl font-semibold text-red-600">{sentimentData.negative_mentions}</p>
                </div>
              </div>
            </div>
            
            <div>
              <h3 className="text-lg font-medium mb-3">Recent News Analysis</h3>
              {sentimentData.recent_news.map((news, index) => (
                <div key={index} className="bg-gray-50 p-4 rounded-lg mb-3">
                  <h4 className="font-medium">{news.title}</h4>
                  <p className="text-sm text-gray-500 mb-2">{news.source} - {news.date}</p>
                  <p className="text-gray-700 mb-2">{news.summary}</p>
                  <div 
                    className={`inline-block px-2 py-1 rounded-full text-xs font-semibold ${
                      news.sentiment >= 0.5 ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'
                    }`}
                  >
                    {news.sentiment >= 0.5 ? 'Positive' : 'Negative'} 
                    ({(news.sentiment * 100).toFixed(2)}%)
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
} 