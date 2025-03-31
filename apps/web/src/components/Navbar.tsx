'use client';

import React, { useState, useEffect } from 'react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';

interface NavbarProps {
  userInfo?: {
    name: string;
    email: string;
    image?: string;
  };
}

const navLinks = [
  { href: '/dashboard', label: 'Dashboard' },
  { href: '/stock', label: 'Stocks' },
  { href: '/recommendation', label: 'Recommendations' },
  { href: '/portfolio', label: 'Portfolio' },
  { href: '/analysis', label: 'Analysis' },
  { href: '/planner', label: 'Investment Planner' },
  { href: '/chat', label: 'AI Assistant' },
] as const;

const Navbar: React.FC<NavbarProps> = ({ userInfo }) => {
  const [scrolled, setScrolled] = useState(false);
  const [mode, setMode] = useState<'beginner' | 'expert'>('beginner');
  const pathname = usePathname();

  useEffect(() => {
    const handleScroll = () => {
      setScrolled(window.scrollY > 20);
    };

    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  return (
    <nav className={`fixed top-0 left-0 right-0 z-50 transition-all duration-300 ${
      scrolled 
        ? 'bg-white/80 backdrop-blur-md dark:bg-gray-900/80 shadow-lg' 
        : 'bg-transparent'
    }`}>
      <div className="container mx-auto px-4">
        <div className="flex h-16 items-center justify-between">
          <div className="flex items-center">
            <Link href="/" className="text-xl font-bold">
              InvestAI
            </Link>
            <div className="ml-10 hidden space-x-4 md:flex">
              {navLinks.map((link) => (
                <Link
                  key={link.href}
                  href={link.href}
                  className={`rounded-md px-3 py-2 text-sm font-medium transition-colors ${
                    pathname === link.href
                      ? 'bg-primary text-primary-foreground'
                      : 'text-gray-700 hover:bg-gray-100/50 dark:text-gray-200 dark:hover:bg-gray-800/50'
                  }`}
                >
                  {link.label}
                </Link>
              ))}
            </div>
          </div>

          <div className="flex items-center gap-4">
            <select
              value={mode}
              onChange={(e) => setMode(e.target.value as 'beginner' | 'expert')}
              className="rounded-md border border-gray-300 bg-white/50 px-3 py-1 text-sm backdrop-blur-sm transition-colors hover:bg-gray-50 dark:border-gray-600 dark:bg-gray-800/50"
            >
              <option value="beginner">Beginner Mode</option>
              <option value="expert">Expert Mode</option>
            </select>

            {userInfo ? (
              <div className="flex items-center gap-3">
                <span className="text-sm">{userInfo.name}</span>
                {userInfo.image ? (
                  <img
                    src={userInfo.image}
                    alt={userInfo.name}
                    className="h-8 w-8 rounded-full"
                  />
                ) : (
                  <div className="flex h-8 w-8 items-center justify-center rounded-full bg-primary text-primary-foreground">
                    {userInfo.name[0]}
                  </div>
                )}
              </div>
            ) : (
              <Link
                href="/auth/login"
                className="rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-foreground transition-colors hover:bg-primary/90"
              >
                Sign In
              </Link>
            )}
          </div>
        </div>
      </div>
    </nav>
  );
};

export default Navbar; 