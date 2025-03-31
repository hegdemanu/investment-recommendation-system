import React from 'react';
import Navbar from '@/components/Navbar';
import CompoundInterestCalculator from '@/components/CompoundInterestCalculator';

export const metadata = {
  title: 'Investment Planner',
  description: 'Plan your investments and calculate potential returns',
};

export default function PlannerPage() {
  return (
    <>
      <Navbar />
      <div className="container mx-auto p-6 pt-20">
        <header className="mb-8">
          <h1 className="text-3xl font-bold">Investment Planner</h1>
          <p className="text-gray-600 dark:text-gray-400">
            Calculate potential returns and plan your investment strategy
          </p>
        </header>

        <div className="grid gap-6 lg:grid-cols-2">
          <div>
            <h2 className="mb-4 text-xl font-semibold">Investment Calculator</h2>
            <CompoundInterestCalculator />
          </div>

          <div>
            <h2 className="mb-4 text-xl font-semibold">Investment Tips</h2>
            <div className="space-y-4 rounded-lg border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-800 dark:bg-gray-900">
              <div>
                <h3 className="mb-2 font-medium text-primary">Diversification</h3>
                <p className="text-gray-600 dark:text-gray-400">
                  Spread your investments across different asset classes to reduce risk.
                </p>
              </div>

              <div>
                <h3 className="mb-2 font-medium text-primary">Regular Investing</h3>
                <p className="text-gray-600 dark:text-gray-400">
                  Consider dollar-cost averaging by investing fixed amounts regularly.
                </p>
              </div>

              <div>
                <h3 className="mb-2 font-medium text-primary">Risk Management</h3>
                <p className="text-gray-600 dark:text-gray-400">
                  Only invest what you can afford to lose and maintain an emergency fund.
                </p>
              </div>

              <div>
                <h3 className="mb-2 font-medium text-primary">Long-term Perspective</h3>
                <p className="text-gray-600 dark:text-gray-400">
                  Focus on long-term growth rather than short-term market fluctuations.
                </p>
              </div>

              <div className="rounded-lg bg-primary/5 p-4">
                <h3 className="mb-2 font-medium text-primary">Pro Tip</h3>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  The power of compound interest works best over longer time periods. 
                  Starting early, even with smaller amounts, can lead to significant growth.
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </>
  );
} 