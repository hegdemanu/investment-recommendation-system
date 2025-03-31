const express = require('express');
const router = express.Router();
const { register, login, getMe } = require('../controllers/authController');
const auth = require('../middleware/auth');

// Route to register a new user
router.post('/register', register);

// Route to login
router.post('/login', login);

// Route to get current user (protected)
router.get('/me', auth, getMe);

module.exports = router; 