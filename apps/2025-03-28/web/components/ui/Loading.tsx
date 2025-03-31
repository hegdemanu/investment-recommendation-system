'use client'

import React from 'react'
import { Spinner, Center, Text, VStack } from '@chakra-ui/react'

interface LoadingProps {
  text?: string
}

const Loading: React.FC<LoadingProps> = ({ text = 'Loading...' }) => {
  return (
    <Center h="100vh">
      <VStack spacing={4}>
        <Spinner
          thickness="4px"
          speed="0.65s"
          emptyColor="gray.200"
          color="blue.500"
          size="xl"
        />
        {text && <Text>{text}</Text>}
      </VStack>
    </Center>
  )
}

export default Loading 