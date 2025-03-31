import React, { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Input } from './ui/input';
import { Button } from './ui/button';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend } from 'recharts';

interface CalculatorInputs {
  principal: number;
  monthlyContribution: number;
  annualReturn: number;
  years: number;
}

const CompoundCalculator: React.FC = () => {
  const [inputs, setInputs] = useState<CalculatorInputs>({
    principal: 10000,
    monthlyContribution: 500,
    annualReturn: 8,
    years: 10,
  });
  const [results, setResults] = useState<any[]>([]);

  const calculateCompoundInterest = () => {
    const monthlyRate = inputs.annualReturn / 100 / 12;
    const totalMonths = inputs.years * 12;
    let balance = inputs.principal;
    const data = [];

    for (let month = 0; month <= totalMonths; month++) {
      balance = balance * (1 + monthlyRate) + inputs.monthlyContribution;
      if (month % 12 === 0) {
        data.push({
          year: month / 12,
          balance: Math.round(balance),
          invested: Math.round(inputs.principal + inputs.monthlyContribution * month),
        });
      }
    }

    setResults(data);
  };

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target;
    setInputs(prev => ({
      ...prev,
      [name]: parseFloat(value) || 0,
    }));
  };

  return (
    <Card className="w-full max-w-4xl mx-auto">
      <CardHeader>
        <CardTitle className="text-2xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
          Compound Interest Calculator
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium mb-1">Initial Investment ($)</label>
              <Input
                type="number"
                name="principal"
                value={inputs.principal}
                onChange={handleInputChange}
                className="w-full"
              />
            </div>
            <div>
              <label className="block text-sm font-medium mb-1">Monthly Contribution ($)</label>
              <Input
                type="number"
                name="monthlyContribution"
                value={inputs.monthlyContribution}
                onChange={handleInputChange}
                className="w-full"
              />
            </div>
            <div>
              <label className="block text-sm font-medium mb-1">Annual Return (%)</label>
              <Input
                type="number"
                name="annualReturn"
                value={inputs.annualReturn}
                onChange={handleInputChange}
                className="w-full"
              />
            </div>
            <div>
              <label className="block text-sm font-medium mb-1">Investment Period (Years)</label>
              <Input
                type="number"
                name="years"
                value={inputs.years}
                onChange={handleInputChange}
                className="w-full"
              />
            </div>
            <Button 
              onClick={calculateCompoundInterest}
              className="w-full bg-gradient-to-r from-blue-600 to-purple-600 text-white"
            >
              Calculate
            </Button>
          </div>
          <div className="min-h-[400px]">
            {results.length > 0 && (
              <>
                <div className="mb-4">
                  <h3 className="text-lg font-semibold mb-2">Results</h3>
                  <p className="text-sm">
                    Final Balance: ${results[results.length - 1].balance.toLocaleString()}
                  </p>
                  <p className="text-sm">
                    Total Invested: ${results[results.length - 1].invested.toLocaleString()}
                  </p>
                  <p className="text-sm">
                    Total Return: ${(results[results.length - 1].balance - results[results.length - 1].invested).toLocaleString()}
                  </p>
                </div>
                <LineChart width={500} height={300} data={results}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="year" />
                  <YAxis />
                  <Tooltip formatter={(value) => `$${value.toLocaleString()}`} />
                  <Legend />
                  <Line
                    type="monotone"
                    dataKey="balance"
                    stroke="#6366f1"
                    name="Balance"
                  />
                  <Line
                    type="monotone"
                    dataKey="invested"
                    stroke="#9333ea"
                    name="Invested"
                  />
                </LineChart>
              </>
            )}
          </div>
        </div>
      </CardContent>
    </Card>
  );
};

export default CompoundCalculator; 