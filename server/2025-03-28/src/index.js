const express = require('express');
const path = require('path');
const axios = require('axios');

const app = express();
const PORT = process.env.PORT || 3000;
const BACKEND_URL = process.env.BACKEND_URL || 'http://localhost:8000';

// Middleware for parsing JSON and URL-encoded data
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// Serve static files from the client build directory
app.use(express.static(path.join(__dirname, '../../client/dist')));

// API proxy endpoints
app.use('/api/*', async (req, res) => {
  try {
    const endpoint = req.originalUrl.replace('/api', '');
    const method = req.method.toLowerCase();
    const data = method === 'get' ? { params: req.query } : { data: req.body };
    
    const response = await axios({
      method: method,
      url: `${BACKEND_URL}${endpoint}`,
      ...data,
      headers: {
        'Content-Type': 'application/json',
      },
    });
    
    return res.status(response.status).json(response.data);
  } catch (error) {
    console.error('API Proxy Error:', error.message);
    const status = error.response?.status || 500;
    const message = error.response?.data || 'Internal Server Error';
    return res.status(status).json({ error: message });
  }
});

// Serve React app for any other routes
app.get('*', (req, res) => {
  res.sendFile(path.join(__dirname, '../../client/dist/index.html'));
});

app.listen(PORT, () => {
  console.log(`Express server running on http://localhost:${PORT}`);
  console.log(`Proxying API requests to ${BACKEND_URL}`);
}); 