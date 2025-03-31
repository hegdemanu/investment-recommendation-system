'use client'

import React, { useEffect } from 'react'
import { Box, Text, Center, HStack, VStack, useColorModeValue } from '@chakra-ui/react'
import { useSelector, useDispatch } from 'react-redux'
import { RootState, AppDispatch } from '@/lib/redux/store'
import { fetchPortfolioData } from '@/lib/redux/slices/portfolioSlice'
import { 
  PieChart, 
  Pie, 
  Cell, 
  Tooltip, 
  Legend, 
  ResponsiveContainer 
} from 'recharts'

// Custom colors for each asset class
const COLORS = ['#3182CE', '#38A169', '#ECC94B', '#9F7AEA', '#E53E3E', '#DD6B20']

const AllocationPieChart: React.FC = () => {
  const dispatch = useDispatch<AppDispatch>()
  const { allocationByAsset, loading, error } = useSelector((state: RootState) => state.portfolio)
  
  useEffect(() => {
    dispatch(fetchPortfolioData())
  }, [dispatch])
  
  // Color mode values for dark/light mode compatibility
  const textColor = useColorModeValue('#2D3748', '#E2E8F0') // gray.700 in light mode, gray.200 in dark mode
  const tooltipBg = useColorModeValue('#FFFFFF', '#1A202C') // white in light mode, gray.800 in dark mode
  const borderColor = useColorModeValue('#E2E8F0', '#2D3748') // gray.200 in light mode, gray.700 in dark mode
  
  // Custom tooltip formatter
  const CustomTooltip = ({ active, payload }: any) => {
    if (active && payload && payload.length) {
      return (
        <Box 
          bg={tooltipBg} 
          p={2} 
          borderRadius="md" 
          borderWidth="1px" 
          borderColor={borderColor}
          boxShadow="md"
        >
          <Text fontWeight="bold" color={payload[0].payload.color}>
            {payload[0].name}: {payload[0].value}%
          </Text>
        </Box>
      )
    }
    return null
  }
  
  // Custom legend renderer
  const renderLegend = (props: any) => {
    const { payload } = props
    
    return (
      <VStack spacing={1} align="stretch">
        {payload.map((entry: any, index: number) => (
          <HStack key={`legend-${index}`} justify="space-between">
            <HStack>
              <Box 
                w="12px" 
                h="12px" 
                borderRadius="sm"
                bg={entry.color} 
                mr={1}
              />
              <Text fontSize="xs" color={textColor}>{entry.value}</Text>
            </HStack>
            <Text fontSize="xs" fontWeight="bold" color={textColor}>
              {entry.payload.value}%
            </Text>
          </HStack>
        ))}
      </VStack>
    )
  }

  return (
    <VStack h="300px" spacing={4} align="stretch">
      <Text fontWeight="medium">Asset Allocation</Text>
      
      <Box flex="1">
        {loading ? (
          <Center h="100%">
            <Text color="gray.500">Loading allocation data...</Text>
          </Center>
        ) : error ? (
          <Center h="100%">
            <Text color="red.500">Error loading data: {error}</Text>
          </Center>
        ) : allocationByAsset.length === 0 ? (
          <Center h="100%">
            <Text color="gray.500">No portfolio data available</Text>
          </Center>
        ) : (
          <ResponsiveContainer width="100%" height="100%">
            <PieChart>
              <Pie
                data={allocationByAsset}
                cx="50%"
                cy="50%"
                innerRadius={60}
                outerRadius={90}
                paddingAngle={2}
                dataKey="value"
                label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                labelLine={false}
              >
                {allocationByAsset.map((entry, index) => (
                  <Cell 
                    key={`cell-${index}`} 
                    fill={COLORS[index % COLORS.length]} 
                    stroke={tooltipBg}
                    strokeWidth={1}
                  />
                ))}
              </Pie>
              <Tooltip content={<CustomTooltip />} />
              <Legend 
                layout="vertical" 
                verticalAlign="middle" 
                align="right"
                content={renderLegend}
              />
            </PieChart>
          </ResponsiveContainer>
        )}
      </Box>
    </VStack>
  )
}

export default AllocationPieChart 