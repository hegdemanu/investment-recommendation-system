const express = require('express');
const router = express.Router();
const auth = require('../middleware/auth');
const { check, validationResult } = require('express-validator');

// Import User model
const User = require('../models/User');

/**
 * @route    GET /api/users/me
 * @desc     Get current user profile
 * @access   Private
 */
router.get('/me', auth, async (req, res) => {
  try {
    const user = await User.findById(req.user.id).select('-password');
    if (!user) {
      return res.status(404).json({ message: 'User not found' });
    }
    res.json(user);
  } catch (err) {
    console.error('Error fetching user profile:', err.message);
    res.status(500).json({ message: 'Server error' });
  }
});

/**
 * @route    PUT /api/users/me
 * @desc     Update user profile
 * @access   Private
 */
router.put(
  '/me',
  [
    auth,
    [
      check('name', 'Name is required').not().isEmpty(),
      check('email', 'Please include a valid email').isEmail(),
    ]
  ],
  async (req, res) => {
    const errors = validationResult(req);
    if (!errors.isEmpty()) {
      return res.status(400).json({ errors: errors.array() });
    }

    const { name, email } = req.body;

    try {
      // Check if email already exists
      const emailExists = await User.findOne({ email, _id: { $ne: req.user.id } });
      if (emailExists) {
        return res.status(400).json({ message: 'Email already in use' });
      }

      // Update user
      const user = await User.findByIdAndUpdate(
        req.user.id,
        { $set: { name, email } },
        { new: true }
      ).select('-password');

      res.json(user);
    } catch (err) {
      console.error('Error updating user profile:', err.message);
      res.status(500).json({ message: 'Server error' });
    }
  }
);

module.exports = router; 