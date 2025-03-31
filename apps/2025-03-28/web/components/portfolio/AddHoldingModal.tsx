'use client'

import React, { useState } from 'react'
import {
  Modal,
  ModalOverlay,
  ModalContent,
  ModalHeader,
  ModalFooter,
  ModalBody,
  ModalCloseButton,
  Button,
  FormControl,
  FormLabel,
  Input,
  Select,
  NumberInput,
  NumberInputField,
  NumberInputStepper,
  NumberIncrementStepper,
  NumberDecrementStepper,
  FormErrorMessage,
  VStack,
  useToast,
} from '@chakra-ui/react'
import { usePortfolioData } from '@/lib/hooks/usePortfolioData'

interface AddHoldingModalProps {
  isOpen: boolean
  onClose: () => void
}

const AddHoldingModal: React.FC<AddHoldingModalProps> = ({ isOpen, onClose }) => {
  const [symbol, setSymbol] = useState('')
  const [quantity, setQuantity] = useState(1)
  const [price, setPrice] = useState(0)
  const [assetClass, setAssetClass] = useState('stock')
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [errors, setErrors] = useState<Record<string, string>>({})
  
  const { updateHoldingData } = usePortfolioData()
  const toast = useToast()
  
  const validateForm = () => {
    const newErrors: Record<string, string> = {}
    
    if (!symbol.trim()) {
      newErrors.symbol = 'Symbol is required'
    }
    
    if (quantity <= 0) {
      newErrors.quantity = 'Quantity must be greater than 0'
    }
    
    if (price <= 0) {
      newErrors.price = 'Price must be greater than 0'
    }
    
    setErrors(newErrors)
    return Object.keys(newErrors).length === 0
  }
  
  const handleSubmit = async () => {
    if (!validateForm()) return
    
    setIsSubmitting(true)
    
    try {
      const result = await updateHoldingData(symbol.toUpperCase(), quantity, price)
      
      if (result.success) {
        toast({
          title: 'Holding added',
          description: `Successfully added ${quantity} shares of ${symbol.toUpperCase()}`,
          status: 'success',
          duration: 5000,
          isClosable: true,
        })
        
        // Reset form
        setSymbol('')
        setQuantity(1)
        setPrice(0)
        setAssetClass('stock')
        onClose()
      } else {
        toast({
          title: 'Error',
          description: result.error || 'Failed to add holding',
          status: 'error',
          duration: 5000,
          isClosable: true,
        })
      }
    } catch (error) {
      toast({
        title: 'Error',
        description: 'An unexpected error occurred',
        status: 'error',
        duration: 5000,
        isClosable: true,
      })
    } finally {
      setIsSubmitting(false)
    }
  }
  
  return (
    <Modal isOpen={isOpen} onClose={onClose}>
      <ModalOverlay />
      <ModalContent>
        <ModalHeader>Add New Holding</ModalHeader>
        <ModalCloseButton />
        <ModalBody>
          <VStack spacing={4}>
            <FormControl isInvalid={!!errors.symbol}>
              <FormLabel>Symbol</FormLabel>
              <Input 
                value={symbol}
                onChange={(e) => setSymbol(e.target.value)}
                placeholder="AAPL"
                autoFocus
              />
              {errors.symbol && <FormErrorMessage>{errors.symbol}</FormErrorMessage>}
            </FormControl>
            
            <FormControl>
              <FormLabel>Asset Type</FormLabel>
              <Select 
                value={assetClass}
                onChange={(e) => setAssetClass(e.target.value)}
              >
                <option value="stock">Stock</option>
                <option value="etf">ETF</option>
                <option value="mutual_fund">Mutual Fund</option>
                <option value="bond">Bond</option>
                <option value="cash">Cash</option>
                <option value="other">Other</option>
              </Select>
            </FormControl>
            
            <FormControl isInvalid={!!errors.quantity}>
              <FormLabel>Quantity</FormLabel>
              <NumberInput
                value={quantity}
                onChange={(_, valueAsNumber) => setQuantity(valueAsNumber)}
                min={1}
                precision={2}
              >
                <NumberInputField />
                <NumberInputStepper>
                  <NumberIncrementStepper />
                  <NumberDecrementStepper />
                </NumberInputStepper>
              </NumberInput>
              {errors.quantity && <FormErrorMessage>{errors.quantity}</FormErrorMessage>}
            </FormControl>
            
            <FormControl isInvalid={!!errors.price}>
              <FormLabel>Purchase Price</FormLabel>
              <NumberInput
                value={price}
                onChange={(_, valueAsNumber) => setPrice(valueAsNumber)}
                min={0}
                precision={2}
                format={(val) => `$${val}`}
              >
                <NumberInputField />
                <NumberInputStepper>
                  <NumberIncrementStepper />
                  <NumberDecrementStepper />
                </NumberInputStepper>
              </NumberInput>
              {errors.price && <FormErrorMessage>{errors.price}</FormErrorMessage>}
            </FormControl>
          </VStack>
        </ModalBody>
        
        <ModalFooter>
          <Button variant="ghost" mr={3} onClick={onClose}>
            Cancel
          </Button>
          <Button 
            colorScheme="blue" 
            onClick={handleSubmit} 
            isLoading={isSubmitting}
            loadingText="Adding"
          >
            Add Holding
          </Button>
        </ModalFooter>
      </ModalContent>
    </Modal>
  )
}

export default AddHoldingModal 