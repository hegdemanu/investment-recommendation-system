'use client';

import { useState } from 'react';
import { useRouter } from 'next/navigation';
import { useStock } from '../../hooks/useStock';

export default function StockPage() {
  return (
    <div className="container mx-auto px-4 py-8">
      <h1 className="text-3xl font-bold mb-8">Stock Analysis</h1>
      <div className="bg-white rounded-lg shadow p-6">
        <p>Stock analysis content coming soon...</p>
      </div>
    </div>
  );
} 