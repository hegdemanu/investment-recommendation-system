import React, { useState, useEffect } from 'react';
import { Star, TrendingUp, TrendingDown, Percent, DollarSign, BarChart2, Info } from 'lucide-react';
import * as Tabs from '@radix-ui/react-tabs';
import * as Tooltip from '@radix-ui/react-tooltip';

const stockRecommendations = [
  {
    id: 1,
    symbol: 'NVDA',
    name: 'NVIDIA Corporation',
    price: 830.15,
    change: 3.2,
    rating: 'Strong Buy',
    targetPrice: 950,
    potentialReturn: 14.4,
    recommendationStrength: 95,
    industry: 'Semiconductors',
    reasons: [
      'Market leader in AI GPU hardware',
      'Strong growth in data center revenue',
      'Expanding into new markets',
      'Continued innovation in chip architecture'
    ]
  },
  {
    id: 2,
    symbol: 'AAPL',
    name: 'Apple Inc.',
    price: 178.85,
    change: 0.5,
    rating: 'Buy',
    targetPrice: 210,
    potentialReturn: 17.4,
    recommendationStrength: 80,
    industry: 'Consumer Electronics',
    reasons: [
      'Stable ecosystem and loyal customer base',
      'Expanding services revenue',
      'Strong balance sheet',
      'New AI features expected to drive upgrades'
    ]
  },
  {
    id: 3,
    symbol: 'META',
    name: 'Meta Platforms, Inc.',
    price: 486.18,
    change: 2.1,
    rating: 'Buy',
    targetPrice: 550,
    potentialReturn: 13.1,
    recommendationStrength: 85,
    industry: 'Internet Content & Information',
    reasons: [
      'Successful cost-cutting measures',
      'AI investments showing promising results',
      'Strong advertising business recovery',
      'Metaverse strategy beginning to show potential'
    ]
  },
  {
    id: 4,
    symbol: 'INTC',
    name: 'Intel Corporation',
    price: 31.85,
    change: -1.2,
    rating: 'Hold',
    targetPrice: 35,
    potentialReturn: 9.9,
    recommendationStrength: 60,
    industry: 'Semiconductors',
    reasons: [
      'Turnaround efforts showing mixed results',
      'Foundry business gaining traction',
      'Competition remains intense',
      'Dividend yield provides support'
    ]
  },
  {
    id: 5,
    symbol: 'AMZN',
    name: 'Amazon.com, Inc.',
    price: 142.60,
    change: 1.8,
    rating: 'Strong Buy',
    targetPrice: 180,
    potentialReturn: 26.2,
    recommendationStrength: 90,
    industry: 'Internet Retail',
    reasons: [
      'AWS growth acceleration expected',
      'Improving e-commerce margins',
      'AI capabilities enhancing multiple business units',
      'Advertising business showing strong growth'
    ]
  }
];

// Portfolio optimization suggestions
const optimizationSuggestions = [
  {
    id: 1,
    title: 'Diversify Tech Exposure',
    description: 'Your portfolio is heavily weighted in technology stocks (70%). Consider reducing exposure to balance risk.',
    impact: 'Medium',
    difficulty: 'Medium',
    potentialBenefit: 'Reduced volatility and sector risk',
    actions: [
      'Reduce AAPL position by 20%',
      'Consider adding exposure to healthcare or consumer staples'
    ]
  },
  {
    id: 2,
    title: 'Increase International Exposure',
    description: '95% of your portfolio is in US stocks. Adding international stocks could improve diversification.',
    impact: 'High',
    difficulty: 'Low',
    potentialBenefit: 'Better geographic diversification and currency exposure',
    actions: [
      'Add 5-10% allocation to European or Asian markets',
      'Consider international ETFs like VXUS or EFA'
    ]
  },
  {
    id: 3,
    title: 'Add Fixed Income Component',
    description: 'Your portfolio is 100% equities. Adding bonds could reduce overall portfolio volatility.',
    impact: 'Medium',
    difficulty: 'Low',
    potentialBenefit: 'Lower portfolio volatility and income generation',
    actions: [
      'Consider 10-20% allocation to bond ETFs',
      'Look at short-term treasury ETFs for current high yields'
    ]
  }
];

const RecommendationCard = ({ recommendation }: { recommendation: any }) => {
  const [expanded, setExpanded] = useState(false);

  return (
    <div className="bg-card p-6 rounded-lg shadow-sm border border-border mb-4">
      <div className="flex flex-col md:flex-row md:items-center justify-between mb-4">
        <div className="flex items-start mb-4 md:mb-0">
          <div className="mr-4">
            <div className={`w-12 h-12 flex items-center justify-center rounded-full ${
              recommendation.rating === 'Strong Buy' ? 'bg-green-100 dark:bg-green-900' : 
              recommendation.rating === 'Buy' ? 'bg-emerald-100 dark:bg-emerald-900' :
              recommendation.rating === 'Hold' ? 'bg-yellow-100 dark:bg-yellow-900' :
              'bg-red-100 dark:bg-red-900'
            }`}>
              <Star className={`h-6 w-6 ${
                recommendation.rating === 'Strong Buy' ? 'text-green-600 dark:text-green-400' : 
                recommendation.rating === 'Buy' ? 'text-emerald-600 dark:text-emerald-400' :
                recommendation.rating === 'Hold' ? 'text-yellow-600 dark:text-yellow-400' :
                'text-red-600 dark:text-red-400'
              }`} />
            </div>
          </div>
          <div>
            <div className="flex items-center">
              <h3 className="text-lg font-semibold">{recommendation.symbol}</h3>
              <span className="text-muted-foreground ml-2 text-sm">{recommendation.industry}</span>
            </div>
            <p className="text-sm text-muted-foreground">{recommendation.name}</p>
          </div>
        </div>
        <div className="flex space-x-4">
          <div>
            <p className="text-xs text-muted-foreground">Current Price</p>
            <p className="text-sm font-medium">${recommendation.price.toFixed(2)}</p>
          </div>
          <div>
            <p className="text-xs text-muted-foreground">Daily Change</p>
            <div className="flex items-center">
              {recommendation.change > 0 ? (
                <TrendingUp className="h-3 w-3 mr-1 text-green-500" />
              ) : (
                <TrendingDown className="h-3 w-3 mr-1 text-red-500" />
              )}
              <p className={`text-sm font-medium ${
                recommendation.change > 0 ? 'text-green-500' : 'text-red-500'
              }`}>
                {recommendation.change > 0 ? '+' : ''}{recommendation.change}%
              </p>
            </div>
          </div>
          <div>
            <p className="text-xs text-muted-foreground">Target Price</p>
            <p className="text-sm font-medium">${recommendation.targetPrice}</p>
          </div>
          <div>
            <p className="text-xs text-muted-foreground">Potential Return</p>
            <p className="text-sm font-medium text-green-500">+{recommendation.potentialReturn}%</p>
          </div>
        </div>
      </div>

      <div className="mb-4">
        <div className="flex justify-between text-sm mb-1">
          <span>Recommendation Strength</span>
          <span className="font-medium">{recommendation.recommendationStrength}%</span>
        </div>
        <div className="w-full bg-muted rounded-full h-2">
          <div 
            className={`h-2 rounded-full ${
              recommendation.recommendationStrength > 80 ? 'bg-green-500' : 
              recommendation.recommendationStrength > 60 ? 'bg-emerald-500' : 'bg-yellow-500'
            }`} 
            style={{ width: `${recommendation.recommendationStrength}%` }}
          ></div>
        </div>
      </div>

      <button 
        className="text-primary text-sm hover:underline"
        onClick={() => setExpanded(!expanded)}
      >
        {expanded ? 'Show less' : 'Show more'}
      </button>

      {expanded && (
        <div className="mt-4 space-y-4">
          <div>
            <h4 className="text-sm font-medium mb-2">Why We Recommend</h4>
            <ul className="list-disc pl-5 text-sm space-y-1">
              {recommendation.reasons.map((reason: string, index: number) => (
                <li key={index} className="text-muted-foreground">{reason}</li>
              ))}
            </ul>
          </div>
          
          <div className="pt-3 border-t border-border">
            <button 
              className="btn btn-primary"
            >
              Add to Watchlist
            </button>
            <button className="btn btn-secondary ml-3">
              View Analysis
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

const OptimizationSuggestionCard = ({ suggestion }: { suggestion: any }) => {
  const [expanded, setExpanded] = useState(false);

  const getImpactColor = (impact: string) => {
    switch (impact) {
      case 'High': return 'text-green-500';
      case 'Medium': return 'text-yellow-500';
      case 'Low': return 'text-blue-500';
      default: return 'text-muted-foreground';
    }
  };

  return (
    <div className="bg-card p-6 rounded-lg shadow-sm border border-border mb-4">
      <div className="flex justify-between items-start mb-4">
        <div>
          <h3 className="text-lg font-semibold">{suggestion.title}</h3>
          <p className="text-sm text-muted-foreground mt-1">{suggestion.description}</p>
        </div>
        <Tooltip.Provider>
          <Tooltip.Root>
            <Tooltip.Trigger asChild>
              <button className="h-8 w-8 inline-flex items-center justify-center rounded-full border border-border hover:bg-muted">
                <Info className="h-4 w-4 text-muted-foreground" />
                <span className="sr-only">More info</span>
              </button>
            </Tooltip.Trigger>
            <Tooltip.Portal>
              <Tooltip.Content
                className="bg-popover text-popover-foreground px-3 py-1.5 text-sm rounded-md shadow-md max-w-xs"
                sideOffset={5}
              >
                {suggestion.potentialBenefit}
                <Tooltip.Arrow className="fill-popover" />
              </Tooltip.Content>
            </Tooltip.Portal>
          </Tooltip.Root>
        </Tooltip.Provider>
      </div>

      <div className="grid grid-cols-2 gap-4 mb-4">
        <div>
          <p className="text-xs text-muted-foreground">Impact</p>
          <p className={`text-sm font-medium ${getImpactColor(suggestion.impact)}`}>
            {suggestion.impact}
          </p>
        </div>
        <div>
          <p className="text-xs text-muted-foreground">Difficulty</p>
          <p className="text-sm font-medium">
            {suggestion.difficulty}
          </p>
        </div>
      </div>

      <button 
        className="text-primary text-sm hover:underline"
        onClick={() => setExpanded(!expanded)}
      >
        {expanded ? 'Show less' : 'Show more'}
      </button>

      {expanded && (
        <div className="mt-4 space-y-4">
          <div>
            <h4 className="text-sm font-medium mb-2">Suggested Actions</h4>
            <ul className="list-disc pl-5 text-sm space-y-1">
              {suggestion.actions.map((action: string, index: number) => (
                <li key={index} className="text-muted-foreground">{action}</li>
              ))}
            </ul>
          </div>
          
          <div className="pt-3 border-t border-border">
            <button className="btn btn-primary">
              Apply Recommendation
            </button>
            <button className="btn btn-secondary ml-3">
              Customize
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

const Recommendations = () => {
  const [isLoading, setIsLoading] = useState(true);
  const [activeTab, setActiveTab] = useState('stocks');
  
  useEffect(() => {
    const timer = setTimeout(() => {
      setIsLoading(false);
    }, 1000);

    return () => clearTimeout(timer);
  }, []);

  return (
    <div>
      <h1 className="text-2xl font-semibold mb-6">Recommendations</h1>
      
      {isLoading ? (
        <div className="flex justify-center items-center h-64">
          <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-primary"></div>
        </div>
      ) : (
        <>
          {/* Strategy Summary */}
          <div className="bg-card p-6 rounded-lg shadow-sm border border-border mb-8">
            <h2 className="text-lg font-medium mb-4">Investment Strategy Overview</h2>
            <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
              <div className="flex items-center">
                <div className="p-3 bg-blue-500/10 rounded-full mr-3">
                  <BarChart2 className="h-6 w-6 text-blue-500" />
                </div>
                <div>
                  <p className="text-muted-foreground text-sm">Risk Profile</p>
                  <p className="font-medium">Growth</p>
                </div>
              </div>
              
              <div className="flex items-center">
                <div className="p-3 bg-green-500/10 rounded-full mr-3">
                  <TrendingUp className="h-6 w-6 text-green-500" />
                </div>
                <div>
                  <p className="text-muted-foreground text-sm">Target Return</p>
                  <p className="font-medium">12-15%</p>
                </div>
              </div>
              
              <div className="flex items-center">
                <div className="p-3 bg-yellow-500/10 rounded-full mr-3">
                  <Percent className="h-6 w-6 text-yellow-500" />
                </div>
                <div>
                  <p className="text-muted-foreground text-sm">Asset Allocation</p>
                  <p className="font-medium">80% Equity / 20% Fixed</p>
                </div>
              </div>
              
              <div className="flex items-center">
                <div className="p-3 bg-purple-500/10 rounded-full mr-3">
                  <DollarSign className="h-6 w-6 text-purple-500" />
                </div>
                <div>
                  <p className="text-muted-foreground text-sm">Investment Horizon</p>
                  <p className="font-medium">5+ Years</p>
                </div>
              </div>
            </div>
          </div>
          
          <Tabs.Root value={activeTab} onValueChange={setActiveTab}>
            <Tabs.List className="flex border-b border-border mb-6">
              <Tabs.Trigger 
                value="stocks" 
                className="px-4 py-2 font-medium data-[state=active]:text-primary data-[state=active]:border-b-2 data-[state=active]:border-primary"
              >
                Stock Recommendations
              </Tabs.Trigger>
              <Tabs.Trigger 
                value="optimization" 
                className="px-4 py-2 font-medium data-[state=active]:text-primary data-[state=active]:border-b-2 data-[state=active]:border-primary"
              >
                Portfolio Optimization
              </Tabs.Trigger>
            </Tabs.List>
            
            <Tabs.Content value="stocks">
              <div className="space-y-4">
                {stockRecommendations.map(recommendation => (
                  <RecommendationCard key={recommendation.id} recommendation={recommendation} />
                ))}
              </div>
            </Tabs.Content>
            
            <Tabs.Content value="optimization">
              <div className="space-y-4">
                {optimizationSuggestions.map(suggestion => (
                  <OptimizationSuggestionCard key={suggestion.id} suggestion={suggestion} />
                ))}
              </div>
            </Tabs.Content>
          </Tabs.Root>
        </>
      )}
    </div>
  );
};

export default Recommendations; 