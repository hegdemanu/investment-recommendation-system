import React, { useEffect, useState } from 'react';
import { 
  Box, 
  SimpleGrid, 
  Heading, 
  Text, 
  Flex, 
  Spinner, 
  Alert, 
  AlertIcon,
  Button,
  useDisclosure,
  Modal,
  ModalOverlay,
  ModalContent,
  ModalHeader,
  ModalCloseButton,
  ModalBody,
  ModalFooter
} from '@chakra-ui/react';
import { useDispatch, useSelector } from 'react-redux';
import { AppDispatch, RootState } from '@/lib/redux/store';
import { fetchRecommendations, Recommendation } from '@/lib/redux/slices/recommendationSlice';
import RecommendationCard from './RecommendationCard';

const RecommendationsGrid: React.FC = () => {
  const dispatch = useDispatch<AppDispatch>();
  const { recommendations, loading, error } = useSelector((state: RootState) => state.recommendations);
  const { isOpen, onOpen, onClose } = useDisclosure();
  const [selectedRecommendation, setSelectedRecommendation] = useState<Recommendation | null>(null);

  useEffect(() => {
    dispatch(fetchRecommendations());
  }, [dispatch]);

  const handleCardClick = (recommendation: Recommendation) => {
    setSelectedRecommendation(recommendation);
    onOpen();
  };

  if (loading) {
    return (
      <Flex justify="center" align="center" minH="300px">
        <Spinner size="xl" thickness="4px" speed="0.65s" color="blue.500" />
      </Flex>
    );
  }

  if (error) {
    return (
      <Alert status="error" borderRadius="md">
        <AlertIcon />
        {error}
      </Alert>
    );
  }

  if (recommendations.length === 0) {
    return (
      <Box textAlign="center" py="8">
        <Heading size="md" mb="4">No Recommendations Available</Heading>
        <Text>We don't have any investment recommendations for you at this time.</Text>
      </Box>
    );
  }

  return (
    <Box>
      <SimpleGrid columns={{ base: 1, md: 2, lg: 3 }} spacing="6">
        {recommendations.map((recommendation) => (
          <RecommendationCard 
            key={recommendation.id} 
            recommendation={recommendation} 
            onClick={() => handleCardClick(recommendation)}
          />
        ))}
      </SimpleGrid>

      {/* Recommendation Detail Modal */}
      {selectedRecommendation && (
        <Modal isOpen={isOpen} onClose={onClose} size="lg">
          <ModalOverlay />
          <ModalContent>
            <ModalHeader>
              {selectedRecommendation.title}
            </ModalHeader>
            <ModalCloseButton />
            
            <ModalBody>
              <Box mb="4">
                <Heading size="sm" mb="2">Stock Information</Heading>
                <Text>Symbol: {selectedRecommendation.stock.symbol}</Text>
                <Text>Name: {selectedRecommendation.stock.name}</Text>
                <Text>
                  Current Price: ${selectedRecommendation.stock.price.current.toFixed(2)} 
                  ({selectedRecommendation.stock.price.changePercent >= 0 ? '+' : ''}
                  {selectedRecommendation.stock.price.changePercent.toFixed(2)}%)
                </Text>
              </Box>
              
              <Box mb="4">
                <Heading size="sm" mb="2">Recommendation Details</Heading>
                <Text>Action: {selectedRecommendation.action}</Text>
                <Text>Target Price: ${selectedRecommendation.targetPrice.toFixed(2)}</Text>
                <Text>
                  Potential Return: {selectedRecommendation.potentialReturn >= 0 ? '+' : ''}
                  {selectedRecommendation.potentialReturn.toFixed(2)}%
                </Text>
                <Text>Risk Level: {selectedRecommendation.riskLevel}</Text>
                <Text>Time Horizon: {selectedRecommendation.timeHorizon}</Text>
              </Box>
              
              <Box>
                <Heading size="sm" mb="2">Analysis</Heading>
                <Text>{selectedRecommendation.description}</Text>
              </Box>
            </ModalBody>

            <ModalFooter>
              <Button colorScheme="blue" mr={3} onClick={onClose}>
                Close
              </Button>
              <Button variant="ghost">Add to Watchlist</Button>
            </ModalFooter>
          </ModalContent>
        </Modal>
      )}
    </Box>
  );
};

export default RecommendationsGrid; 