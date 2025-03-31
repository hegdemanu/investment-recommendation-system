'use client';

import { useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { Box, Center, Spinner, Text, VStack } from '@chakra-ui/react';

export default function Home() {
  const router = useRouter();

  useEffect(() => {
    // Redirect to dashboard after a short delay
    const redirectTimer = setTimeout(() => {
      router.push('/dashboard');
    }, 1000);

    return () => clearTimeout(redirectTimer);
  }, [router]);

  return (
    <Box minH="100vh" display="flex" alignItems="center" justifyContent="center">
      <Center>
        <VStack spacing={4}>
          <Spinner size="xl" thickness="4px" speed="0.65s" color="blue.500" />
          <Text fontSize="lg">Loading Investment Dashboard...</Text>
        </VStack>
      </Center>
    </Box>
  );
}
