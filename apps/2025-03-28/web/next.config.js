/** @type {import('next').NextConfig} */
const nextConfig = {
  // Turn on React strict mode for better development experience
  reactStrictMode: true,
  
  // Image optimization configuration
  images: {
    // Configure domains for remote images
    domains: ['images.unsplash.com', 'cdn.example.com'],
    
    // Set image formats that should be optimized
    formats: ['image/avif', 'image/webp'],
    
    // Configure the image loader to use a CDN
    loader: 'default',
    
    // Maximum image dimension (width or height)
    imageSizes: [16, 32, 48, 64, 96, 128, 256, 384],
    deviceSizes: [640, 750, 828, 1080, 1200, 1920, 2048, 3840],
    
    // Cache images for a week (604800 seconds)
    minimumCacheTTL: 60 * 60 * 24 * 7,
  },
  
  // Set up custom headers for security and performance
  async headers() {
    return [
      {
        source: '/(.*)',
        headers: [
          {
            key: 'X-Content-Type-Options',
            value: 'nosniff',
          },
          {
            key: 'X-Frame-Options',
            value: 'DENY',
          },
          {
            key: 'X-XSS-Protection',
            value: '1; mode=block',
          },
          {
            key: 'Referrer-Policy',
            value: 'strict-origin-when-cross-origin',
          },
        ],
      },
    ]
  },
  
  // Compress the output (can be skipped if using a CDN with built-in compression)
  compress: true,
  
  // Set up environment variables (can also be loaded from .env files)
  env: {
    NEXT_PUBLIC_API_URL: process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000/api',
    NEXT_PUBLIC_ENVIRONMENT: process.env.NODE_ENV || 'development',
  },
  
  // Disable certain features for better performance in production
  ...(process.env.NODE_ENV === 'production' && {
    // Disable source maps in production for smaller bundle size
    productionBrowserSourceMaps: false,
    
    // Remove console logs in production
    compiler: {
      removeConsole: {
        exclude: ['error', 'warn'],
      },
    },
  }),
}

module.exports = nextConfig 