import type { NextConfig } from 'next';

const nextConfig: NextConfig = {
  reactStrictMode: true,
  async rewrites() {
    return [
      {
        source: '/api/:path*',
        destination: 'http://localhost:8000/api/:path*',
      },
    ];
  },
  // Fix for webpack error: Cannot read properties of undefined (reading 'call')
  webpack: (config, { isServer }) => {
    // Fix for "Cannot read properties of undefined (reading 'call')" error
    config.resolve.fallback = {
      ...config.resolve.fallback,
      fs: false,
      net: false,
      tls: false,
      crypto: require.resolve('crypto-browserify'),
      stream: require.resolve('stream-browserify'),
      path: require.resolve('path-browserify'),
      process: require.resolve('process/browser'),
    };

    return config;
  },
  // Next.js 15 features
  experimental: {
    // Enable server actions in Next.js 15
    serverActions: {
      bodySizeLimit: '2mb',
      allowedOrigins: ['localhost:3000']
    },
    // Optimize package imports
    optimizePackageImports: ['@chakra-ui/react', 'framer-motion'],
    // Turbo config for Next.js 15
    turbo: {
      loaders: {
        '.svg': ['file']
      }
    }
  },
};

export default nextConfig; 