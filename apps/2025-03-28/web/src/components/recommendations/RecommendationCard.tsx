import React from 'react';
import { 
  Box, 
  Card, 
  CardBody, 
  Heading, 
  Stack, 
  StackDivider, 
  Text, 
  Badge, 
  Flex,
  HStack,
  Stat,
  StatLabel,
  StatNumber,
  StatHelpText,
  StatArrow,
  IconButton,
  Tooltip
} from '@chakra-ui/react';
import { ChevronRightIcon } from '@chakra-ui/icons';
import { Recommendation } from '@/lib/redux/slices/recommendationSlice';

interface RecommendationCardProps {
  recommendation: Recommendation;
  onClick?: () => void;
}

const getActionColor = (action: 'Buy' | 'Sell' | 'Hold') => {
  switch (action) {
    case 'Buy': return 'green';
    case 'Sell': return 'red';
    case 'Hold': return 'yellow';
    default: return 'gray';
  }
};

const getRiskColor = (risk: 'Low' | 'Medium' | 'High') => {
  switch (risk) {
    case 'Low': return 'green';
    case 'Medium': return 'yellow';
    case 'High': return 'red';
    default: return 'gray';
  }
};

const RecommendationCard: React.FC<RecommendationCardProps> = ({ recommendation, onClick }) => {
  const { 
    title, 
    description, 
    action, 
    targetPrice, 
    potentialReturn, 
    riskLevel, 
    timeHorizon, 
    stock 
  } = recommendation;

  const isPositiveReturn = potentialReturn >= 0;

  return (
    <Card 
      borderLeft="4px solid" 
      borderColor={getActionColor(action)}
      maxW="sm" 
      boxShadow="md" 
      _hover={{ 
        transform: 'translateY(-5px)', 
        transition: 'transform 0.3s ease-in-out',
        boxShadow: 'lg'
      }}
      onClick={onClick}
      cursor={onClick ? 'pointer' : 'default'}
    >
      <CardBody>
        <Stack spacing="4">
          <Flex justify="space-between" align="center">
            <Badge colorScheme={getActionColor(action)} fontSize="sm" px="2">
              {action}
            </Badge>
            <Badge colorScheme="purple" fontSize="sm">
              {timeHorizon}
            </Badge>
          </Flex>
          
          <Heading size="md">{title}</Heading>
          
          <Text fontSize="sm" color="gray.600" noOfLines={2}>
            {description}
          </Text>
          
          <Stack divider={<StackDivider />} spacing="4">
            <HStack justifyContent="space-between">
              <Stat size="sm">
                <StatLabel fontSize="xs">Current Price</StatLabel>
                <StatNumber fontSize="md">${stock.price.current.toFixed(2)}</StatNumber>
                <StatHelpText fontSize="xs">
                  <StatArrow type={stock.price.change >= 0 ? 'increase' : 'decrease'} />
                  {stock.price.changePercent.toFixed(2)}%
                </StatHelpText>
              </Stat>
              
              <Stat size="sm">
                <StatLabel fontSize="xs">Target Price</StatLabel>
                <StatNumber fontSize="md">${targetPrice.toFixed(2)}</StatNumber>
                <StatHelpText fontSize="xs">
                  <StatArrow type={isPositiveReturn ? 'increase' : 'decrease'} />
                  {Math.abs(potentialReturn).toFixed(2)}%
                </StatHelpText>
              </Stat>
            </HStack>
            
            <Flex justifyContent="space-between" alignItems="center">
              <Badge colorScheme={getRiskColor(riskLevel)} size="sm">
                {riskLevel} Risk
              </Badge>
              
              <Tooltip label="View Details">
                <IconButton
                  aria-label="View recommendation details"
                  icon={<ChevronRightIcon />}
                  variant="ghost"
                  size="sm"
                  onClick={(e) => {
                    e.stopPropagation();
                    onClick && onClick();
                  }}
                />
              </Tooltip>
            </Flex>
          </Stack>
        </Stack>
      </CardBody>
    </Card>
  );
};

export default RecommendationCard; 