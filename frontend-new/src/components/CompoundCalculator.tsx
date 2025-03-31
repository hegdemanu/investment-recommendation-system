import React, { useState } from 'react';
import { Card } from './ui/card';
import { Button } from './ui/button';

const CompoundCalculator = () => {
  const [principal, setPrincipal] = useState<number>(0);
  const [rate, setRate] = useState<number>(0);
  const [time, setTime] = useState<number>(0);
  const [frequency, setFrequency] = useState<number>(12);
  const [result, setResult] = useState<number | null>(null);

  const calculateCompound = () => {
    const r = rate / 100;
    const amount = principal * Math.pow(1 + r/frequency, frequency * time);
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
    <Card className="p-6">
      <h2 className="text-2xl font-bold bg-gradient-to-r from-purple-600 to-blue-500 bg-clip-text text-transparent mb-6">
        Compound Interest Calculator
      </h2>
      <div className="space-y-4">
        <div>
          <label className="block text-sm font-medium mb-1">Principal Amount (₹)</label>
          <input
            type="number"
            value={principal}
            onChange={(e) => setPrincipal(Number(e.target.value))}
            className="w-full p-2 border rounded-md"
            placeholder="Enter amount in Rupees"
          />
        </div>
        <div>
          <label className="block text-sm font-medium mb-1">Annual Interest Rate (%)</label>
          <input
            type="number"
            value={rate}
            onChange={(e) => setRate(Number(e.target.value))}
            className="w-full p-2 border rounded-md"
            placeholder="Enter rate (e.g., 7.5)"
          />
        </div>
        <div>
          <label className="block text-sm font-medium mb-1">Time (Years)</label>
          <input
            type="number"
            value={time}
            onChange={(e) => setTime(Number(e.target.value))}
            className="w-full p-2 border rounded-md"
            placeholder="Enter time period"
          />
        </div>
        <div>
          <label className="block text-sm font-medium mb-1">Compounding Frequency</label>
          <select
            value={frequency}
            onChange={(e) => setFrequency(Number(e.target.value))}
            className="w-full p-2 border rounded-md"
          >
            <option value={1}>Annually</option>
            <option value={2}>Semi-annually</option>
            <option value={4}>Quarterly</option>
            <option value={12}>Monthly</option>
            <option value={365}>Daily</option>
          </select>
        </div>
        <Button onClick={calculateCompound} className="w-full">
          Calculate
        </Button>
        {result !== null && (
          <div className="mt-4 p-4 bg-gray-50 dark:bg-gray-800 rounded-md">
            <p className="text-sm font-medium">Final Amount:</p>
            <p className="text-2xl font-bold bg-gradient-to-r from-purple-600 to-blue-500 bg-clip-text text-transparent">
              {formatCurrency(result)}
            </p>
          </div>
        )}
      </div>
    </Card>
  );
};

export default CompoundCalculator; 