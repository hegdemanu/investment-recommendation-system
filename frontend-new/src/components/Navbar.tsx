'use client';

import React, { useState, useEffect } from 'react';
import Link from 'next/link';
import { ThemeToggle } from '@/components/ThemeToggle';
import { usePathname } from 'next/navigation';

const Navbar = () => {
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const [scrolled, setScrolled] = useState(false);
  const pathname = usePathname();

  useEffect(() => {
    const handleScroll = () => {
      setScrolled(window.scrollY > 20);
    };

    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  const toggleMobileMenu = () => {
    setMobileMenuOpen(!mobileMenuOpen);
  };

  const isActive = (path: string) => pathname === path;

  return (
    <nav className={`sticky w-full top-0 z-40 transition-all duration-300 ${
      scrolled 
        ? 'bg-white/80 dark:bg-gray-900/80 backdrop-blur-md shadow-lg'
        : 'bg-white dark:bg-gray-900'
    }`}>
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between h-16">
          <div className="flex items-center">
            <div className="flex-shrink-0 flex items-center animate-fade-in">
              <div className="h-10 w-10 rounded-lg bg-gradient-to-r from-blue-600 via-indigo-600 to-purple-600 dark:from-blue-400 dark:via-indigo-400 dark:to-purple-400 flex items-center justify-center mr-2">
                <svg
                  className="h-6 w-6"
                  xmlns="http://www.w3.org/2000/svg"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="white"
                  strokeWidth="2"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                >
                  <path d="M2 12H5L8 5L14 19L16 12H22" />
                </svg>
              </div>
              <Link 
                href="/" 
                className="text-xl font-bold bg-gradient-to-r from-blue-600 via-indigo-600 to-purple-600 dark:from-blue-400 dark:via-indigo-400 dark:to-purple-400 bg-clip-text text-transparent hover:opacity-80 transition-opacity"
              >
                InvestSage AI
              </Link>
            </div>
            <div className="hidden sm:ml-10 sm:flex sm:space-x-8">
              <Link 
                href="/" 
                className={`inline-flex items-center px-1 pt-1 border-b-2 text-sm font-medium transition-colors duration-200 ${
                  isActive('/') 
                    ? 'border-primary text-primary dark:text-primary' 
                    : 'border-transparent text-gray-700 dark:text-gray-300 hover:border-primary/50 hover:text-primary dark:hover:text-primary'
                }`}
              >
                Home
              </Link>
              <Link 
                href="/stock" 
                className={`inline-flex items-center px-1 pt-1 border-b-2 text-sm font-medium transition-colors duration-200 ${
                  isActive('/stock') 
                    ? 'border-secondary text-secondary dark:text-secondary' 
                    : 'border-transparent text-gray-700 dark:text-gray-300 hover:border-secondary/50 hover:text-secondary dark:hover:text-secondary'
                }`}
              >
                Stocks
              </Link>
              <Link 
                href="/dashboard" 
                className={`inline-flex items-center px-1 pt-1 border-b-2 text-sm font-medium transition-colors duration-200 ${
                  isActive('/dashboard') 
                    ? 'border-accent text-accent dark:text-accent' 
                    : 'border-transparent text-gray-700 dark:text-gray-300 hover:border-accent/50 hover:text-accent dark:hover:text-accent'
                }`}
              >
                Dashboard
              </Link>
            </div>
          </div>
          <div className="hidden sm:ml-6 sm:flex sm:items-center space-x-4">
            <ThemeToggle />
            <button className="btn-primary text-sm">
              Login
            </button>
          </div>
          <div className="flex items-center sm:hidden">
            <ThemeToggle />
            <button 
              onClick={toggleMobileMenu}
              className="inline-flex items-center justify-center p-2 ml-2 rounded-md text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-800 focus:outline-none focus:ring-2 focus:ring-inset focus:ring-primary transition-colors"
              aria-expanded={mobileMenuOpen}
            >
              <span className="sr-only">Open main menu</span>
              {mobileMenuOpen ? (
                <svg className="h-6 w-6" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              ) : (
                <svg className="h-6 w-6" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
                </svg>
              )}
            </button>
          </div>
        </div>
      </div>

      {/* Mobile menu */}
      <div 
        className={`sm:hidden transition-all duration-300 ease-in-out ${
          mobileMenuOpen ? 'max-h-64 opacity-100' : 'max-h-0 opacity-0'
        } overflow-hidden`}
      >
        <div className="px-2 pt-2 pb-3 space-y-1">
          <Link 
            href="/" 
            className={`block px-3 py-2 rounded-md text-base font-medium transition-colors ${
              isActive('/') 
                ? 'bg-primary/10 text-primary' 
                : 'text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-800 hover:text-primary'
            }`}
          >
            Home
          </Link>
          <Link 
            href="/stock" 
            className={`block px-3 py-2 rounded-md text-base font-medium transition-colors ${
              isActive('/stock') 
                ? 'bg-secondary/10 text-secondary' 
                : 'text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-800 hover:text-secondary'
            }`}
          >
            Stocks
          </Link>
          <Link 
            href="/dashboard" 
            className={`block px-3 py-2 rounded-md text-base font-medium transition-colors ${
              isActive('/dashboard') 
                ? 'bg-accent/10 text-accent' 
                : 'text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-800 hover:text-accent'
            }`}
          >
            Dashboard
          </Link>
          <div className="px-3 py-2">
            <button className="btn-primary w-full text-sm">
              Login
            </button>
          </div>
        </div>
      </div>
    </nav>
  );
};

export default Navbar; 