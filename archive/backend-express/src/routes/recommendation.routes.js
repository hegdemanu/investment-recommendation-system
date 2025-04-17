import express from 'express';
import { protect, admin } from '../middleware/auth.middleware.js';
import {
  getRecommendations,
  getRecommendationById,
  createRecommendation,
  updateRecommendation,
  deleteRecommendation,
  getUserRecommendations,
  getStockRecommendations,
  generateRecommendationsForUser
} from '../controllers/recommendation.controller.js';

const router = express.Router();

// @route   GET /api/recommendations
// @desc    Get all recommendations (filtered)
// @access  Private/Admin
router.get('/', protect, admin, getRecommendations);

// @route   GET /api/recommendations/user
// @desc    Get recommendations for logged in user
// @access  Private
router.get('/user', protect, getUserRecommendations);

// @route   GET /api/recommendations/generate
// @desc    Generate new recommendations for user
// @access  Private
router.get('/generate', protect, generateRecommendationsForUser);

// @route   GET /api/recommendations/stock/:symbol
// @desc    Get recommendations for a specific stock
// @access  Private
router.get('/stock/:symbol', protect, getStockRecommendations);

// @route   GET /api/recommendations/:id
// @desc    Get recommendation by ID
// @access  Private
router.get('/:id', protect, getRecommendationById);

// @route   POST /api/recommendations
// @desc    Create a new recommendation (admin only)
// @access  Private/Admin
router.post('/', protect, admin, createRecommendation);

// @route   PUT /api/recommendations/:id
// @desc    Update a recommendation (admin only)
// @access  Private/Admin
router.put('/:id', protect, admin, updateRecommendation);

// @route   DELETE /api/recommendations/:id
// @desc    Delete a recommendation (admin only)
// @access  Private/Admin
router.delete('/:id', protect, admin, deleteRecommendation);

export default router; 