FROM node:18-alpine

WORKDIR /app

# Copy package files
COPY package*.json ./
COPY packages/api/package*.json ./packages/api/

# Install dependencies
RUN npm ci

# Copy source code
COPY packages/api ./packages/api
COPY packages/common ./packages/common

# Build the application
RUN npm run build

# Remove development dependencies
RUN npm prune --production

EXPOSE 5000

CMD ["npm", "start"] 