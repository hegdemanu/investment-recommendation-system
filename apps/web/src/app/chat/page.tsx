import React from 'react';
import ChatBot from '@/components/ChatBot';
import Navbar from '@/components/Navbar';

export const metadata = {
  title: 'AI Investment Assistant',
  description: 'Chat with our AI to get personalized investment advice and market insights',
};

export default function ChatPage() {
  return (
    <>
      <Navbar />
      <div className="container mx-auto p-6 pt-20">
        <header className="mb-8">
          <h1 className="text-3xl font-bold">AI Investment Assistant</h1>
          <p className="text-gray-600 dark:text-gray-400">
            Get personalized investment advice and market insights from our AI assistant
          </p>
        </header>
        <ChatBot />
      </div>
    </>
  );
} 