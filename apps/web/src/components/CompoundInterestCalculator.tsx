'use client';

import React, { useState } from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';

interface CalculatorInputs {
  principal: number;
  monthlyContribution: number;
  annualInterestRate: number;
  years: number;
}

interface DataPoint {
  year: number;
  balance: number;
  contributions: number;
  interest: number;
}

const CompoundInterestCalculator: React.FC = () => {
  const [inputs, setInputs] = useState<CalculatorInputs>({
    principal: 10000,
    monthlyContribution: 500,
    annualInterestRate: 8,
    years: 20,
  });

  const calculateCompoundInterest = (): DataPoint[] => {
    const data: DataPoint[] = [];
    let balance = inputs.principal;
    let totalContributions = inputs.principal;

    for (let year = 0; year <= inputs.years; year++) {
      const monthlyRate = inputs.annualInterestRate / 100 / 12;
      const yearlyContribution = inputs.monthlyContribution * 12;

      if (year > 0) {
        for (let month = 0; month < 12; month++) {
          balance = (balance + inputs.monthlyContribution) * (1 + monthlyRate);
        }
        totalContributions += yearlyContribution;
      }

      data.push({
        year,
        balance: Math.round(balance),
        contributions: Math.round(totalContributions),
        interest: Math.round(balance - totalContributions),
      });
    }

    return data;
  };

  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 0,
      maximumFractionDigits: 0,
    }).format(value);
  };

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target;
    setInputs(prev => ({
      ...prev,
      [name]: parseFloat(value) || 0,
    }));
  };

  const data = calculateCompoundInterest();
  const finalBalance = data[data.length - 1].balance;
  const totalContributions = data[data.length - 1].contributions;
  const totalInterest = data[data.length - 1].interest;

  return (
    <div className="space-y-6 rounded-lg bg-white p-6 shadow-lg dark:bg-gray-900 dark:shadow-2xl dark:shadow-gray-900/50">
      <h2 className="text-2xl font-semibold text-gray-900 dark:text-white">
        Compound Interest Calculator
      </h2>

      <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-4">
        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-200">
            Initial Investment
          </label>
          <input
            type="number"
            name="principal"
            value={inputs.principal}
            onChange={handleInputChange}
            className="mt-1 block w-full rounded-md border border-gray-300 bg-white px-3 py-2 shadow-sm transition-colors 
                     focus:border-emerald-500 focus:outline-none focus:ring-1 focus:ring-emerald-500 
                     dark:border-gray-600 dark:bg-gray-800 dark:text-white 
                     dark:focus:border-emerald-400 dark:focus:ring-emerald-400"
          />
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-200">
            Monthly Contribution
          </label>
          <input
            type="number"
            name="monthlyContribution"
            value={inputs.monthlyContribution}
            onChange={handleInputChange}
            className="mt-1 block w-full rounded-md border border-gray-300 bg-white px-3 py-2 shadow-sm transition-colors 
                     focus:border-emerald-500 focus:outline-none focus:ring-1 focus:ring-emerald-500 
                     dark:border-gray-600 dark:bg-gray-800 dark:text-white 
                     dark:focus:border-emerald-400 dark:focus:ring-emerald-400"
          />
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-200">
            Annual Interest Rate (%)
          </label>
          <input
            type="number"
            name="annualInterestRate"
            value={inputs.annualInterestRate}
            onChange={handleInputChange}
            className="mt-1 block w-full rounded-md border border-gray-300 bg-white px-3 py-2 shadow-sm transition-colors 
                     focus:border-emerald-500 focus:outline-none focus:ring-1 focus:ring-emerald-500 
                     dark:border-gray-600 dark:bg-gray-800 dark:text-white 
                     dark:focus:border-emerald-400 dark:focus:ring-emerald-400"
          />
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-200">
            Investment Period (Years)
          </label>
          <input
            type="number"
            name="years"
            value={inputs.years}
            onChange={handleInputChange}
            className="mt-1 block w-full rounded-md border border-gray-300 bg-white px-3 py-2 shadow-sm transition-colors 
                     focus:border-emerald-500 focus:outline-none focus:ring-1 focus:ring-emerald-500 
                     dark:border-gray-600 dark:bg-gray-800 dark:text-white 
                     dark:focus:border-emerald-400 dark:focus:ring-emerald-400"
          />
        </div>
      </div>

      <div className="grid gap-6 md:grid-cols-3">
        <div className="rounded-lg bg-emerald-50 p-4 shadow-sm transition-colors dark:bg-emerald-900/20 dark:shadow-emerald-900/10">
          <p className="text-sm font-medium text-emerald-600 dark:text-emerald-400">Final Balance</p>
          <p className="mt-2 text-2xl font-bold text-emerald-700 dark:text-emerald-300">
            {formatCurrency(finalBalance)}
          </p>
        </div>

        <div className="rounded-lg bg-sky-50 p-4 shadow-sm transition-colors dark:bg-sky-900/20 dark:shadow-sky-900/10">
          <p className="text-sm font-medium text-sky-600 dark:text-sky-400">Total Contributions</p>
          <p className="mt-2 text-2xl font-bold text-sky-700 dark:text-sky-300">
            {formatCurrency(totalContributions)}
          </p>
        </div>

        <div className="rounded-lg bg-violet-50 p-4 shadow-sm transition-colors dark:bg-violet-900/20 dark:shadow-violet-900/10">
          <p className="text-sm font-medium text-violet-600 dark:text-violet-400">Total Interest</p>
          <p className="mt-2 text-2xl font-bold text-violet-700 dark:text-violet-300">
            {formatCurrency(totalInterest)}
          </p>
        </div>
      </div>

      <div className="h-80">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data}>
            <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
            <XAxis 
              dataKey="year" 
              label={{ value: 'Years', position: 'bottom', fill: '#9CA3AF' }}
              stroke="#9CA3AF"
              tick={{ fill: '#9CA3AF' }}
            />
            <YAxis 
              tickFormatter={(value) => `$${value.toLocaleString()}`}
              label={{ value: 'Balance ($)', angle: -90, position: 'left', fill: '#9CA3AF' }}
              stroke="#9CA3AF"
              tick={{ fill: '#9CA3AF' }}
            />
            <Tooltip 
              formatter={(value: number) => [`$${value.toLocaleString()}`, '']}
              labelFormatter={(label) => `Year ${label}`}
              contentStyle={{
                backgroundColor: 'rgba(17, 24, 39, 0.8)',
                border: 'none',
                borderRadius: '0.5rem',
                color: '#F3F4F6',
              }}
            />
            <Legend wrapperStyle={{ color: '#9CA3AF' }} />
            <Line
              type="monotone"
              dataKey="balance"
              name="Total Balance"
              stroke="#10B981"
              strokeWidth={2}
              dot={{ fill: '#10B981' }}
              activeDot={{ r: 6, fill: '#10B981' }}
            />
            <Line
              type="monotone"
              dataKey="contributions"
              name="Total Contributions"
              stroke="#0EA5E9"
              strokeWidth={2}
              dot={{ fill: '#0EA5E9' }}
              activeDot={{ r: 6, fill: '#0EA5E9' }}
            />
            <Line
              type="monotone"
              dataKey="interest"
              name="Total Interest"
              stroke="#8B5CF6"
              strokeWidth={2}
              dot={{ fill: '#8B5CF6' }}
              activeDot={{ r: 6, fill: '#8B5CF6' }}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};

export default CompoundInterestCalculator; 