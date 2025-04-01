import React, { useState, useEffect } from 'react';
import { Box, Heading, Tabs, TabList, TabPanels, Tab, TabPanel, Flex, Button, 
  Text, Badge, Spinner, Alert, AlertIcon, VStack, HStack, Divider, 
  Table, Thead, Tbody, Tr, Th, Td, useColorModeValue, useToast } from '@chakra-ui/react';
import { LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import axiosInstance from '../../lib/utils/axios';
import { format } from 'date-fns';

const ModelAnalytics = ({ modelId, stockId }) => {
  const [activeTab, setActiveTab] = useState(0);
  const [rlData, setRlData] = useState(null);
  const [backtrackingData, setBacktrackingData] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const toast = useToast();
  
  const bgColor = useColorModeValue('white', 'gray.800');
  const borderColor = useColorModeValue('gray.200', 'gray.700');
  
  useEffect(() => {
    if (modelId && activeTab === 0) {
      fetchRLData();
    } else if (modelId && activeTab === 1) {
      fetchBacktrackingData();
    }
  }, [modelId, stockId, activeTab]);
  
  const fetchRLData = async () => {
    try {
      setIsLoading(true);
      setError(null);
      
      let url = `/api/models/${modelId}/rl-performance`;
      if (stockId) {
        url = `/api/models/${modelId}/rl-performance/${stockId}`;
      }
      
      const response = await axiosInstance.get(url);
      setRlData(response.data);
    } catch (err) {
      setError(err.response?.data?.message || 'Error fetching RL data');
      console.error('Error fetching RL data:', err);
    } finally {
      setIsLoading(false);
    }
  };
  
  const fetchBacktrackingData = async () => {
    try {
      setIsLoading(true);
      setError(null);
      
      const response = await axiosInstance.get(`/api/models/${modelId}/backtracking-results`);
      setBacktrackingData(response.data);
    } catch (err) {
      setError(err.response?.data?.message || 'Error fetching backtracking data');
      console.error('Error fetching backtracking data:', err);
    } finally {
      setIsLoading(false);
    }
  };
  
  const runRLTraining = async () => {
    try {
      setIsLoading(true);
      setError(null);
      
      if (!stockId) {
        toast({
          title: 'Stock Required',
          description: 'Please select a stock to train the RL model on.',
          status: 'warning',
          duration: 5000,
          isClosable: true,
        });
        setIsLoading(false);
        return;
      }
      
      toast({
        title: 'Starting RL Training',
        description: 'Model training has begun, this may take a few moments...',
        status: 'info',
        duration: 5000,
        isClosable: true,
      });
      
      const response = await axiosInstance.post(`/api/models/${modelId}/train-rl/${stockId}`, {
        trainingConfig: {
          // You could add custom config parameters here
        }
      });
      
      // Reload RL data after training
      await fetchRLData();
      
      toast({
        title: 'RL Training Complete',
        description: 'The model has been successfully trained with reinforcement learning.',
        status: 'success',
        duration: 5000,
        isClosable: true,
      });
    } catch (err) {
      setError(err.response?.data?.message || 'Error training RL model');
      console.error('Error training RL model:', err);
      
      toast({
        title: 'Training Error',
        description: err.response?.data?.message || 'Error training RL model',
        status: 'error',
        duration: 5000,
        isClosable: true,
      });
    } finally {
      setIsLoading(false);
    }
  };
  
  const runBacktrackingAnalysis = async () => {
    try {
      setIsLoading(true);
      setError(null);
      
      if (!stockId) {
        toast({
          title: 'Stock Required',
          description: 'Please select a stock to run backtracking analysis on.',
          status: 'warning',
          duration: 5000,
          isClosable: true,
        });
        setIsLoading(false);
        return;
      }
      
      toast({
        title: 'Starting Backtracking Analysis',
        description: 'Backtracking analysis has begun, this may take a few moments...',
        status: 'info',
        duration: 5000,
        isClosable: true,
      });
      
      const response = await axiosInstance.post(`/api/models/${modelId}/backtrack/${stockId}`, {
        windowSize: 30,
        stepSize: 1,
      });
      
      // Reload backtracking data
      await fetchBacktrackingData();
      
      toast({
        title: 'Backtracking Complete',
        description: 'Backtracking analysis has been successfully completed.',
        status: 'success',
        duration: 5000,
        isClosable: true,
      });
    } catch (err) {
      setError(err.response?.data?.message || 'Error running backtracking analysis');
      console.error('Error running backtracking analysis:', err);
      
      toast({
        title: 'Backtracking Error',
        description: err.response?.data?.message || 'Error running backtracking analysis',
        status: 'error',
        duration: 5000,
        isClosable: true,
      });
    } finally {
      setIsLoading(false);
    }
  };
  
  const renderRLActionBadge = (action) => {
    let colorScheme;
    
    switch (action) {
      case 'Buy':
        colorScheme = 'green';
        break;
      case 'Sell':
        colorScheme = 'red';
        break;
      default:
        colorScheme = 'gray';
    }
    
    return (
      <Badge colorScheme={colorScheme} mr={2}>
        {action}
      </Badge>
    );
  };
  
  const formatPercentage = (value) => {
    return `${value.toFixed(2)}%`;
  };
  
  const renderRLPanel = () => {
    if (isLoading) {
      return <Spinner size="xl" mt={5} />;
    }
    
    if (error) {
      return (
        <Alert status="error" mt={5}>
          <AlertIcon />
          {error}
        </Alert>
      );
    }
    
    if (!rlData) {
      return (
        <Alert status="info" mt={5}>
          <AlertIcon />
          No reinforcement learning data available yet. Run RL training to generate data.
        </Alert>
      );
    }
    
    // Prepare data for charts
    const actionsData = rlData.recentActions.map(action => ({
      date: format(new Date(action.date), 'MM/dd/yyyy'),
      action: action.action,
      confidence: action.confidence * 100,
      reward: action.reward,
      cumulativeReward: action.cumulativeReward || 0
    }));
    
    return (
      <Box>
        <HStack spacing={4} mt={4} mb={4} align="center">
          <Button 
            colorScheme="blue" 
            onClick={runRLTraining}
            isLoading={isLoading}
            loadingText="Training"
          >
            Run RL Training
          </Button>
          <Button onClick={fetchRLData} variant="outline">
            Refresh Data
          </Button>
        </HStack>
        
        <Divider my={4} />
        
        <Heading size="md" mb={4}>RL Model Performance</Heading>
        <Flex direction={{ base: 'column', md: 'row' }} gap={4}>
          <Box 
            p={4} 
            borderWidth="1px" 
            borderRadius="lg" 
            bg={bgColor}
            borderColor={borderColor}
            boxShadow="sm"
            flex={1}
          >
            <VStack align="start" spacing={3}>
              <Text fontWeight="bold">Total Reward</Text>
              <Text fontSize="2xl">{rlData.performance.totalReward.toFixed(2)}</Text>
              
              <Text fontWeight="bold">Total Trades</Text>
              <Text fontSize="2xl">{rlData.performance.totalTrades}</Text>
              
              <Text fontWeight="bold">Win Rate</Text>
              <Text fontSize="2xl">{formatPercentage(rlData.performance.winRate)}</Text>
            </VStack>
          </Box>
          
          <Box 
            p={4} 
            borderWidth="1px" 
            borderRadius="lg" 
            bg={bgColor}
            borderColor={borderColor}
            boxShadow="sm"
            flex={2}
          >
            <Heading size="sm" mb={3}>Cumulative Reward</Heading>
            <ResponsiveContainer width="100%" height={200}>
              <LineChart data={actionsData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="date" />
                <YAxis />
                <Tooltip />
                <Legend />
                <Line type="monotone" dataKey="cumulativeReward" stroke="#8884d8" name="Cumulative Reward" />
              </LineChart>
            </ResponsiveContainer>
          </Box>
        </Flex>
        
        <Box 
          mt={6} 
          p={4} 
          borderWidth="1px" 
          borderRadius="lg" 
          bg={bgColor}
          borderColor={borderColor}
          boxShadow="sm"
        >
          <Heading size="sm" mb={3}>RL Parameters</Heading>
          <Table variant="simple" size="sm">
            <Tbody>
              {rlData.reinforcementLearning && (
                <>
                  <Tr>
                    <Td fontWeight="bold">Learning Rate</Td>
                    <Td>{rlData.reinforcementLearning.learningRate || 'N/A'}</Td>
                    <Td fontWeight="bold">Exploration Rate</Td>
                    <Td>{rlData.reinforcementLearning.explorationRate || 'N/A'}</Td>
                  </Tr>
                  <Tr>
                    <Td fontWeight="bold">Discount Factor</Td>
                    <Td>{rlData.reinforcementLearning.discountFactor || 'N/A'}</Td>
                    <Td fontWeight="bold">Reward Function</Td>
                    <Td>{rlData.reinforcementLearning.rewardFunction || 'N/A'}</Td>
                  </Tr>
                </>
              )}
            </Tbody>
          </Table>
        </Box>
        
        <Heading size="md" mt={8} mb={4}>Recent Actions</Heading>
        <Box overflowX="auto">
          <Table variant="simple">
            <Thead>
              <Tr>
                <Th>Date</Th>
                <Th>Action</Th>
                <Th>Confidence</Th>
                <Th>Reward</Th>
              </Tr>
            </Thead>
            <Tbody>
              {rlData.recentActions.slice(-10).map((action, index) => (
                <Tr key={index}>
                  <Td>{format(new Date(action.date), 'MM/dd/yyyy hh:mm a')}</Td>
                  <Td>{renderRLActionBadge(action.action)}</Td>
                  <Td>{formatPercentage(action.confidence * 100)}</Td>
                  <Td>{action.reward ? action.reward.toFixed(2) : 'N/A'}</Td>
                </Tr>
              ))}
            </Tbody>
          </Table>
        </Box>
      </Box>
    );
  };
  
  const renderBacktrackingPanel = () => {
    if (isLoading) {
      return <Spinner size="xl" mt={5} />;
    }
    
    if (error) {
      return (
        <Alert status="error" mt={5}>
          <AlertIcon />
          {error}
        </Alert>
      );
    }
    
    if (!backtrackingData) {
      return (
        <Alert status="info" mt={5}>
          <AlertIcon />
          No backtracking data available yet. Run backtracking analysis to generate data.
        </Alert>
      );
    }
    
    // Prepare parameter set data for visualization
    const parameterSets = [];
    
    if (backtrackingData.recentBacktests?.length > 0 && 
        backtrackingData.recentBacktests[0].backtracking?.parameterSets) {
      backtrackingData.recentBacktests[0].backtracking.parameterSets.forEach((set, index) => {
        const paramKeys = Object.keys(set.parameters);
        const perfKeys = Object.keys(set.performance);
        
        const paramString = paramKeys.map(key => `${key}: ${set.parameters[key]}`).join(', ');
        
        parameterSets.push({
          id: index + 1,
          parameters: paramString,
          ...set.performance
        });
      });
    }
    
    return (
      <Box>
        <HStack spacing={4} mt={4} mb={4} align="center">
          <Button 
            colorScheme="teal" 
            onClick={runBacktrackingAnalysis}
            isLoading={isLoading}
            loadingText="Analyzing"
          >
            Run Backtracking Analysis
          </Button>
          <Button onClick={fetchBacktrackingData} variant="outline">
            Refresh Data
          </Button>
        </HStack>
        
        <Divider my={4} />
        
        <Heading size="md" mb={4}>Backtracking Results</Heading>
        
        <Flex direction={{ base: 'column', md: 'row' }} gap={4}>
          <Box 
            p={4} 
            borderWidth="1px" 
            borderRadius="lg" 
            bg={bgColor}
            borderColor={borderColor}
            boxShadow="sm"
            flex={1}
          >
            <VStack align="start" spacing={3}>
              {backtrackingData.results && (
                <>
                  <Text fontWeight="bold">Success Rate</Text>
                  <Text fontSize="2xl">
                    {formatPercentage(backtrackingData.results.successRate * 100)}
                  </Text>
                  
                  <Text fontWeight="bold">Profit/Loss</Text>
                  <Text fontSize="2xl" color={backtrackingData.results.profitLoss >= 0 ? 'green.500' : 'red.500'}>
                    {backtrackingData.results.profitLoss.toFixed(2)}%
                  </Text>
                  
                  <Text fontWeight="bold">Total Trades</Text>
                  <Text fontSize="2xl">{backtrackingData.results.trades || 'N/A'}</Text>
                  
                  <Text fontWeight="bold">Avg Holding Period</Text>
                  <Text fontSize="2xl">{backtrackingData.results.avgHoldingPeriod || 'N/A'} days</Text>
                </>
              )}
            </VStack>
          </Box>
          
          <Box 
            p={4} 
            borderWidth="1px" 
            borderRadius="lg" 
            bg={bgColor}
            borderColor={borderColor}
            boxShadow="sm"
            flex={2}
          >
            <Heading size="sm" mb={3}>Parameter Performance</Heading>
            {parameterSets.length > 0 ? (
              <ResponsiveContainer width="100%" height={200}>
                <BarChart data={parameterSets}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="id" />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  <Bar dataKey="profitLoss" fill="#8884d8" name="Profit/Loss %" />
                  <Bar dataKey="winRate" fill="#82ca9d" name="Win Rate" />
                </BarChart>
              </ResponsiveContainer>
            ) : (
              <Text>No parameter set data available</Text>
            )}
          </Box>
        </Flex>
        
        <Box 
          mt={6} 
          p={4} 
          borderWidth="1px" 
          borderRadius="lg" 
          bg={bgColor}
          borderColor={borderColor}
          boxShadow="sm"
        >
          <Heading size="sm" mb={3}>Backtracking Configuration</Heading>
          <Table variant="simple" size="sm">
            <Tbody>
              {backtrackingData.backtrackingConfig && (
                <>
                  <Tr>
                    <Td fontWeight="bold">Window Size</Td>
                    <Td>{backtrackingData.backtrackingConfig.windowSize || 'N/A'} days</Td>
                    <Td fontWeight="bold">Step Size</Td>
                    <Td>{backtrackingData.backtrackingConfig.stepSize || 'N/A'} day(s)</Td>
                  </Tr>
                  <Tr>
                    <Td fontWeight="bold">Status</Td>
                    <Td colSpan={3}>
                      <Badge colorScheme={backtrackingData.backtrackingConfig.enabled ? 'green' : 'red'}>
                        {backtrackingData.backtrackingConfig.enabled ? 'Enabled' : 'Disabled'}
                      </Badge>
                    </Td>
                  </Tr>
                </>
              )}
            </Tbody>
          </Table>
        </Box>
        
        {backtrackingData.results?.optimizedParameters && (
          <Box 
            mt={6} 
            p={4} 
            borderWidth="1px" 
            borderRadius="lg" 
            bg={bgColor}
            borderColor={borderColor}
            boxShadow="sm"
          >
            <Heading size="sm" mb={3}>Optimized Parameters</Heading>
            <Table variant="simple" size="sm">
              <Thead>
                <Tr>
                  <Th>Parameter</Th>
                  <Th>Value</Th>
                </Tr>
              </Thead>
              <Tbody>
                {Object.entries(backtrackingData.results.optimizedParameters).map(([key, value]) => (
                  <Tr key={key}>
                    <Td>{key}</Td>
                    <Td>{typeof value === 'number' ? value.toFixed(4) : value}</Td>
                  </Tr>
                ))}
              </Tbody>
            </Table>
          </Box>
        )}
      </Box>
    );
  };
  
  return (
    <Box width="100%">
      <Heading size="lg" mb={5}>Model Analytics</Heading>
      
      <Tabs 
        colorScheme="blue" 
        variant="enclosed" 
        onChange={(index) => setActiveTab(index)}
      >
        <TabList>
          <Tab>Reinforcement Learning</Tab>
          <Tab>Backtracking Analysis</Tab>
        </TabList>
        
        <TabPanels>
          <TabPanel>
            {renderRLPanel()}
          </TabPanel>
          <TabPanel>
            {renderBacktrackingPanel()}
          </TabPanel>
        </TabPanels>
      </Tabs>
    </Box>
  );
};

export default ModelAnalytics; 