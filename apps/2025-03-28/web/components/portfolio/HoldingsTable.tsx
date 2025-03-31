'use client'

import React, { useState } from 'react'
import {
  Table,
  Thead,
  Tbody,
  Tr,
  Th,
  Td,
  Box,
  Text,
  Badge,
  Flex,
  Icon,
  Spinner,
  Center,
  Button,
  useColorModeValue,
  useDisclosure,
} from '@chakra-ui/react'
import { FiTrendingUp, FiTrendingDown, FiPlus } from 'react-icons/fi'
import { useSelector } from 'react-redux'
import { RootState } from '@/lib/redux/store'
import { PortfolioHolding } from '@/lib/redux/slices/portfolioSlice'
import AddHoldingModal from './AddHoldingModal'

const HoldingsTable: React.FC = () => {
  const { holdings, loading, error } = useSelector((state: RootState) => state.portfolio)
  const { isOpen, onOpen, onClose } = useDisclosure()
  
  const getBadgeColor = (assetClass: string) => {
    switch (assetClass) {
      case 'stock': return 'blue'
      case 'etf': return 'green'
      case 'mutual_fund': return 'purple'
      case 'bond': return 'yellow'
      case 'cash': return 'gray'
      default: return 'gray'
    }
  }
  
  const formatAssetClass = (assetClass: string) => {
    switch (assetClass) {
      case 'mutual_fund': return 'Mutual Fund'
      default: return assetClass.toUpperCase()
    }
  }
  
  const borderColor = useColorModeValue('gray.200', 'gray.700')
  const hoverBg = useColorModeValue('gray.50', 'gray.700')
  
  if (loading) {
    return (
      <Center p={8}>
        <Spinner thickness="4px" speed="0.65s" size="xl" />
      </Center>
    )
  }
  
  if (error) {
    return (
      <Center p={8}>
        <Text color="red.500">{error}</Text>
      </Center>
    )
  }
  
  return (
    <>
      <Flex justifyContent="flex-end" mb={4}>
        <Button 
          leftIcon={<Icon as={FiPlus} />} 
          colorScheme="blue" 
          size="sm"
          onClick={onOpen}
        >
          Add Holding
        </Button>
      </Flex>
      
      {holdings.length === 0 ? (
        <Center p={8}>
          <Text color="gray.500">No holdings in portfolio</Text>
        </Center>
      ) : (
        <Box overflowX="auto">
          <Table variant="simple" size="sm">
            <Thead>
              <Tr>
                <Th>Symbol</Th>
                <Th>Name</Th>
                <Th>Type</Th>
                <Th isNumeric>Quantity</Th>
                <Th isNumeric>Avg. Price</Th>
                <Th isNumeric>Current Price</Th>
                <Th isNumeric>Total Value</Th>
                <Th isNumeric>Change</Th>
              </Tr>
            </Thead>
            <Tbody>
              {holdings.map((holding: PortfolioHolding) => (
                <Tr 
                  key={holding.symbol} 
                  _hover={{ bg: hoverBg, transition: 'all 0.2s' }}
                  cursor="pointer"
                >
                  <Td fontWeight="bold">{holding.symbol}</Td>
                  <Td>{holding.name}</Td>
                  <Td>
                    <Badge colorScheme={getBadgeColor(holding.assetClass)} borderRadius="full" px={2}>
                      {formatAssetClass(holding.assetClass)}
                    </Badge>
                  </Td>
                  <Td isNumeric>{holding.quantity}</Td>
                  <Td isNumeric>${holding.avgPrice.toFixed(2)}</Td>
                  <Td isNumeric>${holding.currentPrice.toFixed(2)}</Td>
                  <Td isNumeric fontWeight="bold">${holding.value.toFixed(2)}</Td>
                  <Td isNumeric>
                    <Flex alignItems="center" justifyContent="flex-end">
                      <Icon 
                        as={holding.change >= 0 ? FiTrendingUp : FiTrendingDown} 
                        color={holding.change >= 0 ? 'green.500' : 'red.500'} 
                        mr={1}
                      />
                      <Text color={holding.change >= 0 ? 'green.500' : 'red.500'}>
                        {holding.changePercent >= 0 ? '+' : ''}{holding.changePercent.toFixed(2)}%
                      </Text>
                    </Flex>
                  </Td>
                </Tr>
              ))}
            </Tbody>
          </Table>
        </Box>
      )}
      
      <AddHoldingModal isOpen={isOpen} onClose={onClose} />
    </>
  )
}

export default HoldingsTable 