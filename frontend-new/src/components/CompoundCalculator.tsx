'use client';

import React, { useState } from 'react';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Tooltip } from '@/components/ui/tooltip';
import { InfoIcon } from 'lucide-react';

const CompoundCalculator = () => {
  const [principal, setPrincipal] = useState<number>(0);
  const [rate, setRate] = useState<number>(0);
  const [time, setTime] = useState<number>(0);
  const [frequency, setFrequency] = useState<number>(12);
  const [result, setResult] = useState<number | null>(null);

  const calculateCompound = () => {
    const r = rate / 100;
    const n = frequency;
    const t = time;
    // Correct compound interest formula: A = P(1 + r/n)^(nt)
    const amount = principal * Math.pow(1 + r/n, n * t);
    setResult(Number(amount.toFixed(2)));
  };

  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat('en-IN', {
      style: 'currency',
      currency: 'INR',
      maximumFractionDigits: 0
    }).format(value);
  };

  return (
    <Card className="p-6 border border-gray-200 dark:border-gray-800 shadow-md rounded-xl">
      <h2 className="text-2xl font-bold text-primary-700 dark:text-primary-400 mb-6 flex items-center">
        Compound Interest Calculator
        <Tooltip content="Calculate how your investments grow over time with compound interest">
          <InfoIcon className="ml-2 h-4 w-4 text-gray-400 cursor-help" />
        </Tooltip>
      </h2>
      <div className="space-y-4">
        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">Principal Amount (₹)</label>
          <Input
            type="number"
            value={principal || ''}
            onChange={(e) => setPrincipal(Number(e.target.value))}
            className="w-full bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100"
            placeholder="Enter amount in Rupees"
          />
        </div>
        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">Annual Interest Rate (%)</label>
          <Input
            type="number"
            value={rate || ''}
            onChange={(e) => setRate(Number(e.target.value))}
            className="w-full bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100"
            placeholder="Enter rate (e.g., 7.5)"
          />
        </div>
        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">Time (Years)</label>
          <Input
            type="number"
            value={time || ''}
            onChange={(e) => setTime(Number(e.target.value))}
            className="w-full bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100"
            placeholder="Enter time period"
          />
        </div>
        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">Compounding Frequency</label>
          <select
            value={frequency}
            onChange={(e) => setFrequency(Number(e.target.value))}
            className="w-full p-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100"
          >
            <option value={1}>Annually</option>
            <option value={2}>Semi-annually</option>
            <option value={4}>Quarterly</option>
            <option value={12}>Monthly</option>
            <option value={365}>Daily</option>
          </select>
        </div>
        <Button 
          onClick={calculateCompound} 
          className="w-full bg-primary-600 hover:bg-primary-700 text-white"
        >
          Calculate
        </Button>
        {result !== null && (
          <div className="mt-4 p-4 bg-gray-50 dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700">
            <p className="text-sm font-medium text-gray-700 dark:text-gray-300">Final Amount:</p>
            <p className="text-2xl font-bold text-primary-600 dark:text-primary-400">
              {formatCurrency(result)}
            </p>
            <div className="mt-2 text-xs text-gray-500 dark:text-gray-400">
              <p>Formula: A = P(1 + r/n)^(nt)</p>
              <p>where:</p>
              <ul className="list-disc ml-4 mt-1">
                <li>A = Final amount</li>
                <li>P = Principal (₹{principal.toLocaleString()})</li>
                <li>r = Rate ({rate}%)</li>
                <li>n = Compounding frequency ({frequency} times per year)</li>
                <li>t = Time ({time} years)</li>
              </ul>
            </div>
          </div>
        )}
      </div>
    </Card>
  );
};

export default CompoundCalculator; 