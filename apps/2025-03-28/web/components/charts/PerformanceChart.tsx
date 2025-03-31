'use client'

import React, { useState, useEffect } from 'react'
import { Box, Text, Center, HStack, VStack, Select, useColorModeValue } from '@chakra-ui/react'
import { useSelector, useDispatch } from 'react-redux'
import { RootState, AppDispatch } from '@/lib/redux/store'
import { fetchPortfolioData } from '@/lib/redux/slices/portfolioSlice'
import { 
  LineChart, 
  Line, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  Legend, 
  ResponsiveContainer 
} from 'recharts'

// Sample data - will be replaced with real data from API
const mockPortfolioData = {
  '1D': [
    { date: '9:30 AM', value: 10000 },
    { date: '10:30 AM', value: 10120 },
    { date: '11:30 AM', value: 10050 },
    { date: '12:30 PM', value: 10200 },
    { date: '1:30 PM', value: 10180 },
    { date: '2:30 PM', value: 10250 },
    { date: '3:30 PM', value: 10400 },
    { date: '4:00 PM', value: 10450 },
  ],
  '1W': [
    { date: 'Mon', value: 10000 },
    { date: 'Tue', value: 10200 },
    { date: 'Wed', value: 10150 },
    { date: 'Thu', value: 10300 },
    { date: 'Fri', value: 10450 },
  ],
  '1M': [
    { date: 'Week 1', value: 10000 },
    { date: 'Week 2', value: 10300 },
    { date: 'Week 3', value: 10200 },
    { date: 'Week 4', value: 10500 },
  ],
  '3M': [
    { date: 'Jan', value: 10000 },
    { date: 'Feb', value: 10400 },
    { date: 'Mar', value: 10800 },
  ],
  '6M': [
    { date: 'Jan', value: 10000 },
    { date: 'Feb', value: 10400 },
    { date: 'Mar', value: 10800 },
    { date: 'Apr', value: 11200 },
    { date: 'May', value: 11000 },
    { date: 'Jun', value: 11500 },
  ],
  '1Y': [
    { date: 'Jan', value: 10000 },
    { date: 'Mar', value: 10800 },
    { date: 'May', value: 11000 },
    { date: 'Jul', value: 11200 },
    { date: 'Sep', value: 11600 },
    { date: 'Nov', value: 11400 },
    { date: 'Jan', value: 12000 },
  ],
  'All': [
    { date: '2020', value: 5000 },
    { date: '2021', value: 7500 },
    { date: '2022', value: 9000 },
    { date: '2023', value: 10500 },
    { date: '2024', value: 12000 },
  ],
}

const PerformanceChart: React.FC = () => {
  const [timeRange, setTimeRange] = useState('1M') // 1D, 1W, 1M, 3M, 6M, 1Y, All
  const dispatch = useDispatch<AppDispatch>()
  const { totalValue, changeToday, changeTodayPercent, loading, error } = useSelector(
    (state: RootState) => state.portfolio
  )
  
  useEffect(() => {
    dispatch(fetchPortfolioData())
  }, [dispatch])
  
  const handleTimeRangeChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    setTimeRange(e.target.value)
  }

  // Get data for the selected time range
  const chartData = mockPortfolioData[timeRange as keyof typeof mockPortfolioData] || []
  
  // Calculate performance metrics
  const startValue = chartData.length > 0 ? chartData[0].value : 0
  const endValue = chartData.length > 0 ? chartData[chartData.length - 1].value : 0
  const performanceValue = endValue - startValue
  const performancePercent = startValue > 0 ? (performanceValue / startValue) * 100 : 0
  
  // Colors
  const lineColor = useColorModeValue('#3182CE', '#63B3ED') // blue.500 in light mode, blue.300 in dark mode
  const gridColor = useColorModeValue('#E2E8F0', '#2D3748') // gray.200 in light mode, gray.700 in dark mode
  const tooltipBg = useColorModeValue('#FFFFFF', '#1A202C') // white in light mode, gray.800 in dark mode
  const textColor = useColorModeValue('#2D3748', '#E2E8F0') // gray.700 in light mode, gray.200 in dark mode
  
  return (
    <VStack h="300px" position="relative" spacing={4} align="stretch">
      <HStack justifyContent="space-between">
        <Text fontWeight="medium">Portfolio Performance</Text>
        <HStack spacing={4}>
          <Text 
            fontSize="sm" 
            fontWeight="medium"
            color={changeToday >= 0 ? 'green.500' : 'red.500'}
          >
            {changeToday >= 0 ? '+' : ''}{changeToday.toFixed(2)} ({changeTodayPercent >= 0 ? '+' : ''}{changeTodayPercent.toFixed(2)}%)
          </Text>
          <Select 
            value={timeRange} 
            onChange={handleTimeRangeChange} 
            size="sm" 
            width="120px"
          >
            <option value="1D">1 Day</option>
            <option value="1W">1 Week</option>
            <option value="1M">1 Month</option>
            <option value="3M">3 Months</option>
            <option value="6M">6 Months</option>
            <option value="1Y">1 Year</option>
            <option value="All">All Time</option>
          </Select>
        </HStack>
      </HStack>
      
      <Box flex="1">
        {loading ? (
          <Center h="100%">
            <Text color="gray.500">Loading performance data...</Text>
          </Center>
        ) : error ? (
          <Center h="100%">
            <Text color="red.500">Error loading data: {error}</Text>
          </Center>
        ) : chartData.length === 0 ? (
          <Center h="100%">
            <Text color="gray.500">No performance data available</Text>
          </Center>
        ) : (
          <ResponsiveContainer width="100%" height="100%">
            <LineChart
              data={chartData}
              margin={{
                top: 5,
                right: 5,
                left: 5,
                bottom: 5,
              }}
            >
              <CartesianGrid strokeDasharray="3 3" stroke={gridColor} />
              <XAxis 
                dataKey="date" 
                tick={{ fill: textColor }} 
                axisLine={{ stroke: gridColor }} 
              />
              <YAxis 
                domain={[(dataMin: number) => dataMin * 0.95, (dataMax: number) => dataMax * 1.05]} 
                tick={{ fill: textColor }}
                axisLine={{ stroke: gridColor }}
                tickFormatter={(value) => `$${value.toLocaleString()}`}
              />
              <Tooltip 
                formatter={(value: number) => [`$${value.toLocaleString()}`, 'Portfolio Value']}
                contentStyle={{ backgroundColor: tooltipBg, borderColor: gridColor }}
                labelStyle={{ color: textColor }}
              />
              <Legend wrapperStyle={{ color: textColor }} />
              <Line 
                type="monotone" 
                dataKey="value" 
                stroke={lineColor} 
                strokeWidth={2}
                dot={false}
                activeDot={{ r: 6 }}
                name="Portfolio Value"
              />
            </LineChart>
          </ResponsiveContainer>
        )}
      </Box>
    </VStack>
  )
}

export default PerformanceChart 