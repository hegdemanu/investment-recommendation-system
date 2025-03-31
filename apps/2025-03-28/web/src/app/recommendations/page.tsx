"use client";

import React from 'react';
import { 
  Box, 
  Heading, 
  Text, 
  Container, 
  Flex, 
  Button, 
  VStack,
  HStack,
  Icon,
  Select,
  Spacer,
  useColorModeValue
} from '@chakra-ui/react';
import { FaFilter } from 'react-icons/fa';
import RecommendationsGrid from '@/components/recommendations/RecommendationsGrid';
import DashboardLayout from '@/components/layout/DashboardLayout';

export default function RecommendationsPage() {
  const bgColor = useColorModeValue('white', 'gray.800');
  const borderColor = useColorModeValue('gray.200', 'gray.700');

  return (
    <DashboardLayout>
      <Container maxW="container.xl" py={6}>
        {/* Header Section */}
        <Box 
          mb={6} 
          p={6} 
          borderRadius="lg" 
          bg={bgColor} 
          boxShadow="sm"
          borderWidth="1px"
          borderColor={borderColor}
        >
          <VStack align="start" spacing={2}>
            <Heading as="h1" size="xl">Investment Recommendations</Heading>
            <Text fontSize="md" color="gray.600">
              Personalized investment suggestions based on your portfolio and market trends.
            </Text>
          </VStack>
        </Box>
        
        {/* Filter Section */}
        <Box 
          mb={6} 
          p={4} 
          borderRadius="lg" 
          bg={bgColor} 
          boxShadow="sm"
          borderWidth="1px"
          borderColor={borderColor}
        >
          <Flex 
            direction={{ base: 'column', md: 'row' }} 
            align={{ base: 'stretch', md: 'center' }}
            gap={4}
          >
            <HStack spacing={4}>
              <Icon as={FaFilter} />
              <Text fontWeight="medium">Filter By:</Text>
            </HStack>
            
            <HStack 
              spacing={4} 
              flex="1"
              flexWrap={{ base: 'wrap', md: 'nowrap' }}
              gap={{ base: 2, md: 0 }}
            >
              <Select placeholder="Action" size="sm" w={{ base: 'full', md: 'auto' }}>
                <option value="buy">Buy</option>
                <option value="sell">Sell</option>
                <option value="hold">Hold</option>
              </Select>
              
              <Select placeholder="Risk Level" size="sm" w={{ base: 'full', md: 'auto' }}>
                <option value="low">Low Risk</option>
                <option value="medium">Medium Risk</option>
                <option value="high">High Risk</option>
              </Select>
              
              <Select placeholder="Time Horizon" size="sm" w={{ base: 'full', md: 'auto' }}>
                <option value="short">Short-term</option>
                <option value="medium">Medium-term</option>
                <option value="long">Long-term</option>
              </Select>
            </HStack>
            
            <Spacer display={{ base: 'none', md: 'block' }} />
            
            <Button 
              size="sm" 
              colorScheme="blue" 
              w={{ base: 'full', md: 'auto' }}
            >
              Apply Filters
            </Button>
          </Flex>
        </Box>
        
        {/* Recommendations Grid */}
        <Box 
          p={6} 
          borderRadius="lg" 
          bg={bgColor} 
          boxShadow="sm"
          borderWidth="1px"
          borderColor={borderColor}
        >
          <RecommendationsGrid />
        </Box>
      </Container>
    </DashboardLayout>
  );
} 