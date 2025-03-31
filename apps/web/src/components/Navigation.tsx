import React from 'react';
import { Button } from '@/components/ui/button';

interface NavigationProps {
  activeTab: string;
  onTabChange: (tab: string) => void;
}

const tabs = [
  {
    id: 'compound',
    label: 'Compound Calculator',
    description: 'Calculate compound interest and investment growth'
  },
  {
    id: 'sentiment',
    label: 'Market Sentiment',
    description: 'Analyze market sentiment and trends'
  },
  {
    id: 'report',
    label: 'Report Generator',
    description: 'Generate detailed investment reports'
  },
  {
    id: 'rag',
    label: 'AI Research',
    description: 'AI-powered investment research'
  }
];

export function Navigation({ activeTab, onTabChange }: NavigationProps) {
  return (
    <div className="flex flex-col space-y-4 w-full max-w-4xl mx-auto p-4">
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {tabs.map((tab) => (
          <Button
            key={tab.id}
            onClick={() => onTabChange(tab.id)}
            variant={activeTab === tab.id ? "default" : "outline"}
            className="w-full h-auto flex flex-col items-start p-4 space-y-2 text-left"
          >
            <span className="text-lg font-semibold bg-gradient-to-r from-purple-600 to-blue-500 bg-clip-text text-transparent">
              {tab.label}
            </span>
            <span className="text-sm text-gray-500 dark:text-gray-400">
              {tab.description}
            </span>
          </Button>
        ))}
      </div>
    </div>
  );
} 