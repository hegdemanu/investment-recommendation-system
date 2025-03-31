const jwt = require('jsonwebtoken');
const config = require('config');
const User = require('../models/User');

/**
 * Authentication middleware
 * Verifies JWT token and sets req.user to decoded user
 */
module.exports = async function(req, res, next) {
  // Get token from header
  const tokenHeader = req.header('x-auth-token');
  const authHeader = req.header('Authorization');
  
  let token = tokenHeader;
  
  // Check for Bearer token in Authorization header
  if (!token && authHeader && authHeader.startsWith('Bearer ')) {
    token = authHeader.split(' ')[1];
  }

  // Check if no token
  if (!token) {
    console.log('Auth middleware: No token provided');
    return res.status(401).json({ msg: 'No token, authorization denied' });
  }

  // Verify token
  try {
    const decoded = jwt.verify(token, config.get('jwtSecret'));
    req.user = decoded.user;
    
    console.log('Auth middleware: Token valid for user:', req.user.id);
    
    // Optional: Check if user still exists in DB
    if (req.user.id !== 'master-admin-id') { // Skip DB check for master admin
      const user = await User.findById(req.user.id).select('-password');
      if (!user) {
        console.log('Auth middleware: User not found in DB');
        return res.status(401).json({ message: 'User not found, authorization denied' });
      }
    }
    
    next();
  } catch (err) {
    console.log('Auth middleware: Token verification failed -', err.message);
    res.status(401).json({ msg: 'Token is not valid' });
  }
};

// Grant access to specific roles
const authorize = (...roles) => {
  return (req, res, next) => {
    if (!req.user || !roles.includes(req.user.role)) {
      return res.status(403).json({
        success: false,
        error: 'Not authorized to access this route'
      });
    }
    next();
  };
};

module.exports.authorize = authorize; 