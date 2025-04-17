import express from 'express';
import { protect, admin } from '../middleware/auth.middleware.js';
import {
  loginUser,
  registerUser,
  getUserProfile,
  updateUserProfile,
  getAllUsers,
  getUserById,
  updateUser,
  deleteUser,
  changePassword,
  refreshToken
} from '../controllers/user.controller.js';

const router = express.Router();

// @route   POST /api/users/register
// @desc    Register a new user
// @access  Public
router.post('/register', registerUser);

// @route   POST /api/users/login
// @desc    Login user & get token
// @access  Public
router.post('/login', loginUser);

// @route   POST /api/users/refresh
// @desc    Refresh access token using refresh token
// @access  Public (with refresh token)
router.post('/refresh', refreshToken);

// @route   GET /api/users/profile
// @desc    Get user profile
// @access  Private
router.get('/profile', protect, getUserProfile);

// @route   PUT /api/users/profile
// @desc    Update user profile
// @access  Private
router.put('/profile', protect, updateUserProfile);

// @route   PUT /api/users/password
// @desc    Change user password
// @access  Private
router.put('/password', protect, changePassword);

// Admin Routes
// @route   GET /api/users
// @desc    Get all users
// @access  Private/Admin
router.get('/', protect, admin, getAllUsers);

// @route   GET /api/users/:id
// @desc    Get user by id
// @access  Private/Admin
router.get('/:id', protect, admin, getUserById);

// @route   PUT /api/users/:id
// @desc    Update user
// @access  Private/Admin
router.put('/:id', protect, admin, updateUser);

// @route   DELETE /api/users/:id
// @desc    Delete user
// @access  Private/Admin
router.delete('/:id', protect, admin, deleteUser);

export default router; 