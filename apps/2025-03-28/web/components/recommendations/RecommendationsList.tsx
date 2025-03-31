'use client'

import React, { useEffect } from 'react'
import { useDispatch, useSelector } from 'react-redux'
import { 
  Box, 
  VStack, 
  HStack, 
  Text, 
  Badge, 
  Divider, 
  Button, 
  Icon,
  useColorModeValue
} from '@chakra-ui/react'
import { FiChevronRight, FiTrendingUp, FiTrendingDown, FiMinus } from 'react-icons/fi'
import Link from 'next/link'
import { RootState, AppDispatch } from '@/lib/redux/store'

interface RecommendationsListProps {
  limit?: number
}

const RecommendationsList: React.FC<RecommendationsListProps> = ({ limit }) => {
  const { recommendations, loading } = useSelector((state: RootState) => state.recommendations)
  const dispatch = useDispatch<AppDispatch>()
  
  // Mock recommendations (would normally be fetched from the API via Redux)
  const mockRecommendations = [
    {
      id: '1',
      title: 'Buy AAPL',
      description: 'Apple shows strong growth potential with upcoming product launches',
      action: 'Buy',
      targetPrice: 185.50,
      potentialReturn: 12.3,
      riskLevel: 'Medium',
      timeHorizon: 'Medium-term',
      stock: {
        symbol: 'AAPL',
        name: 'Apple Inc.',
        price: {
          current: 165.23,
          change: 2.45,
          changePercent: 1.5
        }
      }
    },
    {
      id: '2',
      title: 'Hold MSFT',
      description: 'Microsoft maintains stable growth with cloud services expansion',
      action: 'Hold',
      targetPrice: 340.00,
      potentialReturn: 5.8,
      riskLevel: 'Low',
      timeHorizon: 'Long-term',
      stock: {
        symbol: 'MSFT',
        name: 'Microsoft Corporation',
        price: {
          current: 320.45,
          change: -1.20,
          changePercent: -0.4
        }
      }
    },
    {
      id: '3',
      title: 'Buy AMZN',
      description: 'Amazon poised for growth with expansion in new markets',
      action: 'Buy',
      targetPrice: 145.00,
      potentialReturn: 15.2,
      riskLevel: 'Medium',
      timeHorizon: 'Medium-term',
      stock: {
        symbol: 'AMZN',
        name: 'Amazon.com Inc.',
        price: {
          current: 125.89,
          change: 3.21,
          changePercent: 2.6
        }
      }
    },
    {
      id: '4',
      title: 'Sell FB',
      description: 'Meta faces challenges with advertising revenue and market saturation',
      action: 'Sell',
      targetPrice: 285.00,
      potentialReturn: -8.5,
      riskLevel: 'High',
      timeHorizon: 'Short-term',
      stock: {
        symbol: 'META',
        name: 'Meta Platforms Inc.',
        price: {
          current: 312.45,
          change: -4.78,
          changePercent: -1.5
        }
      }
    },
    {
      id: '5',
      title: 'Buy NVDA',
      description: 'NVIDIA continues strong performance in AI and gaming markets',
      action: 'Buy',
      targetPrice: 950.00,
      potentialReturn: 20.5,
      riskLevel: 'Medium',
      timeHorizon: 'Medium-term',
      stock: {
        symbol: 'NVDA',
        name: 'NVIDIA Corporation',
        price: {
          current: 788.17,
          change: 15.24,
          changePercent: 1.9
        }
      }
    }
  ]
  
  // Use the mock data until the real API integration is complete
  const displayRecommendations = mockRecommendations.slice(0, limit || mockRecommendations.length)
  
  // Action badge colors
  const getBadgeColor = (action: string) => {
    switch (action.toLowerCase()) {
      case 'buy': return 'green'
      case 'sell': return 'red'
      case 'hold': return 'yellow'
      default: return 'gray'
    }
  }
  
  // Price change icon
  const getPriceChangeIcon = (change: number) => {
    if (change > 0) return FiTrendingUp
    if (change < 0) return FiTrendingDown
    return FiMinus
  }
  
  const cardBg = useColorModeValue('white', 'gray.800')
  const hoverBg = useColorModeValue('gray.50', 'gray.700')
  
  return (
    <VStack spacing={4} align="stretch">
      {displayRecommendations.length === 0 ? (
        <Box p={4} textAlign="center">
          <Text color="gray.500">No recommendations available</Text>
        </Box>
      ) : (
        displayRecommendations.map((rec, index) => (
          <Box 
            key={rec.id}
            p={4}
            borderRadius="md"
            bg={cardBg}
            borderWidth="1px"
            borderColor={useColorModeValue('gray.200', 'gray.700')}
            _hover={{
              bg: hoverBg,
              transform: 'translateY(-2px)',
              transition: 'all 0.2s',
              boxShadow: 'md'
            }}
          >
            <HStack justifyContent="space-between" mb={2}>
              <HStack>
                <Text fontWeight="bold">{rec.stock.symbol}</Text>
                <Badge colorScheme={getBadgeColor(rec.action)}>
                  {rec.action}
                </Badge>
              </HStack>
              <HStack>
                <Text>${rec.stock.price.current.toFixed(2)}</Text>
                <Icon 
                  as={getPriceChangeIcon(rec.stock.price.change)}
                  color={rec.stock.price.change >= 0 ? 'green.500' : 'red.500'}
                />
                <Text 
                  color={rec.stock.price.change >= 0 ? 'green.500' : 'red.500'}
                >
                  {rec.stock.price.change >= 0 ? '+' : ''}{rec.stock.price.changePercent.toFixed(2)}%
                </Text>
              </HStack>
            </HStack>
            
            <Text fontSize="sm" color="gray.500" noOfLines={2} mb={2}>
              {rec.description}
            </Text>
            
            <HStack justifyContent="space-between" fontSize="sm">
              <Text>Target: ${rec.targetPrice.toFixed(2)}</Text>
              <Text 
                color={rec.potentialReturn >= 0 ? 'green.500' : 'red.500'}
                fontWeight="medium"
              >
                Potential: {rec.potentialReturn >= 0 ? '+' : ''}{rec.potentialReturn.toFixed(1)}%
              </Text>
              <Link href={`/recommendations/${rec.id}`} passHref>
                <Button 
                  size="xs" 
                  variant="ghost" 
                  rightIcon={<Icon as={FiChevronRight} />}
                >
                  Details
                </Button>
              </Link>
            </HStack>
            
            {index < displayRecommendations.length - 1 && (
              <Divider mt={4} />
            )}
          </Box>
        ))
      )}
      
      {limit && mockRecommendations.length > limit && (
        <Box textAlign="center" pt={2}>
          <Link href="/recommendations" passHref>
            <Button variant="ghost" size="sm" rightIcon={<Icon as={FiChevronRight} />}>
              View all recommendations
            </Button>
          </Link>
        </Box>
      )}
    </VStack>
  )
}

export default RecommendationsList 