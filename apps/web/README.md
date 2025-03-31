# Investment Recommendation System Frontend

This is the frontend application for the Investment Recommendation System, built with Next.js, React, TypeScript, and Tailwind CSS.

## Features

- Modern, responsive UI with Tailwind CSS and Radix UI
- Interactive financial dashboards and visualizations
- Authentication with NextAuth.js
- TypeScript for type safety and developer experience
- API integration with the backend services

## Tech Stack

- **Next.js**: React framework for SSR, routing, and API routes
- **React**: UI library
- **TypeScript**: Static typing
- **Tailwind CSS**: Utility-first CSS framework
- **Radix UI**: Accessible component primitives
- **Chart.js/Lightweight Charts**: Data visualization
- **SWR**: Data fetching and caching
- **NextAuth.js**: Authentication

## Getting Started

1. Install dependencies:
   ```
   npm install
   ```

2. Set up environment variables:
   - Copy `.env.example` to `.env.local`
   - Update the variables as needed

3. Run the development server:
   ```
   npm run dev
   ```

4. Open [http://localhost:3000](http://localhost:3000) in your browser

## Project Structure

- `/src/components`: Reusable UI components
- `/src/pages`: Next.js pages and API routes
- `/src/styles`: Global styles and Tailwind config
- `/src/lib`: Shared utilities and client libraries
- `/src/hooks`: Custom React hooks
- `/src/context`: React context providers
- `/src/utils`: Helper functions
- `/src/api`: API service clients
- `/public`: Static assets

## Build and Deployment

To build the application for production:

```
npm run build
```

To run the production build:

```
npm run start
```

This frontend is designed to be deployed on Vercel for optimal performance with Next.js.
