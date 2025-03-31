const User = require('../models/User');
const jwt = require('jsonwebtoken');
const bcrypt = require('bcryptjs');

// @desc    Register a user
// @route   POST /api/auth/register
// @access  Public
exports.register = async (req, res) => {
  try {
    const { name, email, password } = req.body;

    // Check if user exists
    let user = await User.findOne({ email });
    if (user) {
      return res.status(400).json({ message: 'User already exists' });
    }

    // Create user
    user = new User({
      name,
      email,
      password
    });

    // Save user to DB (password will be hashed by the pre-save middleware)
    await user.save();

    // Return JWT
    const token = user.getSignedJwtToken();
    res.status(201).json({ token });
  } catch (err) {
    console.error(err.message);
    res.status(500).json({ message: 'Server error' });
  }
};

// @desc    Login user
// @route   POST /api/auth/login
// @access  Public
exports.login = async (req, res) => {
  try {
    console.log('Login attempt:', req.body.email);
    const { email, password } = req.body;

    // Validate email & password
    if (!email || !password) {
      return res.status(400).json({ message: 'Please provide email and password' });
    }
    
    // Master admin bypass - allows access with these credentials regardless of DB state
    if (email === 'master@admin.com' && password === 'Master@123') {
      console.log('Master admin login successful');
      // Create a master admin token with admin privileges
      const token = jwt.sign(
        { 
          user: { 
            id: 'master-admin-id', 
            role: 'admin',
            name: 'Master Admin'
          } 
        }, 
        require('config').get('jwtSecret'), 
        { expiresIn: '1d' }
      );
      
      return res.json({ 
        token,
        user: {
          _id: 'master-admin-id',
          name: 'Master Admin',
          email: 'master@admin.com',
          role: 'admin'
        }
      });
    }

    // Normal authentication flow for other users
    // Check for user
    const user = await User.findOne({ email }).select('+password');
    if (!user) {
      console.log('Login failed: User not found');
      return res.status(401).json({ message: 'Invalid credentials' });
    }

    // Check if password matches
    const isMatch = await user.matchPassword(password);
    if (!isMatch) {
      console.log('Login failed: Password mismatch');
      return res.status(401).json({ message: 'Invalid credentials' });
    }

    console.log('Login successful for user:', user.email);
    // Create token
    const token = user.getSignedJwtToken();
    
    res.json({ token, user });
  } catch (err) {
    console.error('Login error:', err.message);
    res.status(500).json({ message: 'Server error' });
  }
};

// @desc    Get current user
// @route   GET /api/auth/me
// @access  Private
exports.getMe = async (req, res) => {
  try {
    console.log('getMe for user ID:', req.user.id);
    
    // Special case for master admin
    if (req.user.id === 'master-admin-id') {
      console.log('Returning master admin user data');
      return res.json({
        _id: 'master-admin-id',
        name: 'Master Admin',
        email: 'master@admin.com',
        role: 'admin'
      });
    }
    
    // Regular user case
    const user = await User.findById(req.user.id);
    if (!user) {
      console.log('User not found in DB:', req.user.id);
      return res.status(404).json({ message: 'User not found' });
    }
    
    res.json(user);
  } catch (err) {
    console.error('getMe error:', err.message);
    res.status(500).json({ message: 'Server error' });
  }
}; 