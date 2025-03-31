"use client";

import React from 'react';
import {
  Box,
  Flex,
  Heading,
  Text,
  SimpleGrid,
  Stat,
  StatLabel,
  StatNumber,
  StatHelpText,
  StatArrow,
  Card,
  CardHeader,
  CardBody,
  Button,
  Icon,
  Divider,
  useColorModeValue,
  HStack,
} from '@chakra-ui/react';
import { FiArrowUpRight, FiArrowDownRight, FiAward, FiTrendingUp, FiPieChart, FiCompass } from 'react-icons/fi';
import Link from 'next/link';
import DashboardLayout from '@/components/layout/DashboardLayout';

export default function Dashboard() {
  const bgColor = useColorModeValue('white', 'gray.800');
  const borderColor = useColorModeValue('gray.200', 'gray.700');

  return (
    <DashboardLayout>
      <Box py={6}>
        {/* Welcome Section */}
        <Box mb={6} p={6} borderRadius="lg" bg={bgColor} boxShadow="sm" borderWidth="1px" borderColor={borderColor}>
          <Flex direction={{ base: 'column', md: 'row' }} justify="space-between" align={{ base: 'start', md: 'center' }}>
            <Box mb={{ base: 4, md: 0 }}>
              <Heading as="h1" size="xl">Welcome, John</Heading>
              <Text mt={1} color="gray.600">
                Here's what's happening with your investments today.
              </Text>
            </Box>
            <HStack spacing={4}>
              <Button 
                as={Link} 
                href="/recommendations" 
                colorScheme="blue" 
                size="md" 
                rightIcon={<Icon as={FiCompass} />}
              >
                View Recommendations
              </Button>
              <Button 
                as={Link} 
                href="/portfolio" 
                variant="outline" 
                size="md" 
                rightIcon={<Icon as={FiPieChart} />}
              >
                Portfolio
              </Button>
            </HStack>
          </Flex>
        </Box>

        {/* Stats Section */}
        <SimpleGrid columns={{ base: 1, md: 3 }} spacing={6} mb={6}>
          <Card borderWidth="1px" borderColor={borderColor} boxShadow="sm">
            <CardBody>
              <Stat>
                <StatLabel>Portfolio Value</StatLabel>
                <StatNumber>$24,860.72</StatNumber>
                <StatHelpText>
                  <StatArrow type="increase" />
                  8.36% from last month
                </StatHelpText>
              </Stat>
            </CardBody>
          </Card>
          
          <Card borderWidth="1px" borderColor={borderColor} boxShadow="sm">
            <CardBody>
              <Stat>
                <StatLabel>Total Gain/Loss</StatLabel>
                <StatNumber color="green.500">+$1,921.43</StatNumber>
                <StatHelpText>
                  <StatArrow type="increase" />
                  12.45% all time
                </StatHelpText>
              </Stat>
            </CardBody>
          </Card>
          
          <Card borderWidth="1px" borderColor={borderColor} boxShadow="sm">
            <CardBody>
              <Stat>
                <StatLabel>Active Recommendations</StatLabel>
                <StatNumber>5</StatNumber>
                <StatHelpText>
                  3 new since last week
                </StatHelpText>
              </Stat>
            </CardBody>
          </Card>
        </SimpleGrid>

        {/* Top Recommendations Section */}
        <Box mb={6} p={6} borderRadius="lg" bg={bgColor} boxShadow="sm" borderWidth="1px" borderColor={borderColor}>
          <Flex justify="space-between" align="center" mb={4}>
            <Heading as="h2" size="lg">Top Recommendations</Heading>
            <Button
              as={Link}
              href="/recommendations"
              variant="ghost"
              colorScheme="blue"
              rightIcon={<FiArrowUpRight />}
            >
              View All
            </Button>
          </Flex>
          
          <SimpleGrid columns={{ base: 1, md: 3 }} spacing={6}>
            <Card borderLeft="4px solid" borderLeftColor="green.400" borderWidth="1px" borderColor={borderColor}>
              <CardHeader pb={0}>
                <Heading size="md">Buy AAPL</Heading>
              </CardHeader>
              <CardBody>
                <Text mb={2}>Apple shows strong growth potential with upcoming product launches</Text>
                <Flex justify="space-between" mt={4}>
                  <Text fontWeight="bold">+12.3%</Text>
                  <HStack>
                    <Icon as={FiAward} color="green.500" />
                    <Text color="green.500" fontWeight="medium">Medium Risk</Text>
                  </HStack>
                </Flex>
              </CardBody>
            </Card>
            
            <Card borderLeft="4px solid" borderLeftColor="red.400" borderWidth="1px" borderColor={borderColor}>
              <CardHeader pb={0}>
                <Heading size="md">Sell META</Heading>
              </CardHeader>
              <CardBody>
                <Text mb={2}>Meta faces challenges with advertising revenue and market saturation</Text>
                <Flex justify="space-between" mt={4}>
                  <Text fontWeight="bold">-8.5%</Text>
                  <HStack>
                    <Icon as={FiAward} color="red.500" />
                    <Text color="red.500" fontWeight="medium">High Risk</Text>
                  </HStack>
                </Flex>
              </CardBody>
            </Card>
            
            <Card borderLeft="4px solid" borderLeftColor="green.400" borderWidth="1px" borderColor={borderColor}>
              <CardHeader pb={0}>
                <Heading size="md">Buy NVDA</Heading>
              </CardHeader>
              <CardBody>
                <Text mb={2}>NVIDIA continues strong performance in AI and gaming markets</Text>
                <Flex justify="space-between" mt={4}>
                  <Text fontWeight="bold">+20.5%</Text>
                  <HStack>
                    <Icon as={FiAward} color="yellow.500" />
                    <Text color="yellow.500" fontWeight="medium">Medium Risk</Text>
                  </HStack>
                </Flex>
              </CardBody>
            </Card>
          </SimpleGrid>
        </Box>

        {/* Recent Activity Section */}
        <Box p={6} borderRadius="lg" bg={bgColor} boxShadow="sm" borderWidth="1px" borderColor={borderColor}>
          <Heading as="h2" size="lg" mb={4}>Recent Activity</Heading>
          
          <Box>
            <Flex justify="space-between" align="center" py={3}>
              <Flex align="center">
                <Icon as={FiTrendingUp} mr={3} color="green.500" />
                <Box>
                  <Text fontWeight="medium">Bought AAPL</Text>
                  <Text fontSize="sm" color="gray.500">10 shares at $165.23</Text>
                </Box>
              </Flex>
              <Text fontSize="sm" color="gray.500">2 days ago</Text>
            </Flex>
            <Divider />
            
            <Flex justify="space-between" align="center" py={3}>
              <Flex align="center">
                <Icon as={FiArrowDownRight} mr={3} color="red.500" />
                <Box>
                  <Text fontWeight="medium">Sold NFLX</Text>
                  <Text fontSize="sm" color="gray.500">5 shares at $628.75</Text>
                </Box>
              </Flex>
              <Text fontSize="sm" color="gray.500">1 week ago</Text>
            </Flex>
            <Divider />
            
            <Flex justify="space-between" align="center" py={3}>
              <Flex align="center">
                <Icon as={FiTrendingUp} mr={3} color="green.500" />
                <Box>
                  <Text fontWeight="medium">Bought MSFT</Text>
                  <Text fontSize="sm" color="gray.500">8 shares at $320.45</Text>
                </Box>
              </Flex>
              <Text fontSize="sm" color="gray.500">2 weeks ago</Text>
            </Flex>
          </Box>
        </Box>
      </Box>
    </DashboardLayout>
  );
} 