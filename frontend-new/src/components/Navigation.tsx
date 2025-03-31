import React from 'react';
import { Button } from '@/components/ui/button';

interface NavigationProps {
  activeTab: string;
  onTabChange: (tab: string) => void;
}

export function Navigation({ activeTab, onTabChange }: NavigationProps) {
  const tabs = [
    {
      id: 'compound',
      label: 'Compound Calculator',
      description: 'Calculate compound interest and investment growth',
      icon: (
        <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <path d="M2 20h.01"></path>
          <path d="M7 20v-4"></path>
          <path d="M12 20v-8"></path>
          <path d="M17 20V8"></path>
          <path d="M22 4v16"></path>
        </svg>
      ),
    },
    {
      id: 'sentiment',
      label: 'Market Sentiment',
      description: 'Analyze market sentiment and trends',
      icon: (
        <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <path d="M14.5 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7.5L14.5 2z"></path>
          <polyline points="14 2 14 8 20 8"></polyline>
          <line x1="16" y1="13" x2="8" y2="13"></line>
          <line x1="16" y1="17" x2="8" y2="17"></line>
          <line x1="10" y1="9" x2="8" y2="9"></line>
        </svg>
      ),
    },
    {
      id: 'report',
      label: 'Report Generator',
      description: 'Generate detailed investment reports',
      icon: (
        <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <path d="M21.21 15.89A10 10 0 1 1 8 2.83"></path>
          <path d="M22 12A10 10 0 0 0 12 2v10z"></path>
        </svg>
      ),
    },
    {
      id: 'rag',
      label: 'AI Research',
      description: 'AI-powered investment research and analysis',
      icon: (
        <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <circle cx="11" cy="11" r="8"></circle>
          <line x1="21" y1="21" x2="16.65" y2="16.65"></line>
        </svg>
      ),
    },
  ];

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
      {tabs.map((tab) => (
        <Button
          key={tab.id}
          variant={activeTab === tab.id ? 'default' : 'outline'}
          className={`h-auto py-4 px-6 flex flex-col items-center gap-2 text-center transition-all duration-200 ${
            activeTab === tab.id
              ? 'bg-gradient-to-r from-purple-600 to-blue-500 text-white transform scale-105'
              : 'hover:bg-gradient-to-r hover:from-purple-600/10 hover:to-blue-500/10'
          }`}
          onClick={() => onTabChange(tab.id)}
        >
          <div className={`w-12 h-12 rounded-lg flex items-center justify-center ${
            activeTab === tab.id
              ? 'bg-white/20'
              : 'bg-gradient-to-r from-purple-600/20 to-blue-500/20'
          }`}>
            {tab.icon}
          </div>
          <div>
            <h3 className="font-semibold">{tab.label}</h3>
            <p className={`text-sm ${
              activeTab === tab.id ? 'text-gray-100' : 'text-gray-500 dark:text-gray-400'
            }`}>
              {tab.description}
            </p>
          </div>
        </Button>
      ))}
    </div>
  );
} 