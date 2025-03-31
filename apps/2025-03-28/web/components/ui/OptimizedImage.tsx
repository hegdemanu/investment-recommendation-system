'use client'

import React from 'react'
import Image from 'next/image'
import { Box, BoxProps, Skeleton } from '@chakra-ui/react'
import { useState } from 'react'

interface OptimizedImageProps extends Omit<BoxProps, 'as'> {
  src: string
  alt: string
  width: number
  height: number
  priority?: boolean
  quality?: number
  objectFit?: 'cover' | 'contain' | 'fill' | 'none'
}

export const OptimizedImage: React.FC<OptimizedImageProps> = ({
  src,
  alt,
  width,
  height,
  priority = false,
  quality = 75,
  objectFit = 'cover',
  ...boxProps
}) => {
  const [isLoading, setIsLoading] = useState(true)

  return (
    <Box position="relative" overflow="hidden" {...boxProps}>
      {isLoading && (
        <Skeleton
          position="absolute"
          top={0}
          left={0}
          width="100%"
          height="100%"
          startColor="gray.100"
          endColor="gray.300"
        />
      )}
      <Image
        src={src}
        alt={alt}
        width={width}
        height={height}
        quality={quality}
        priority={priority}
        onLoad={() => setIsLoading(false)}
        style={{
          objectFit,
          width: '100%',
          height: '100%',
        }}
      />
    </Box>
  )
}

export default OptimizedImage 