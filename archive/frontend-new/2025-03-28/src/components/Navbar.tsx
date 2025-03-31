import React from 'react';
import Link from 'next/link';

type NavbarProps = {
  user?: {
    name: string;
    email: string;
  } | null;
};

const Navbar: React.FC<NavbarProps> = ({ user }) => {
  return (
    <nav className="bg-white shadow-sm">
      <div className="container mx-auto px-4 py-3 flex items-center justify-between">
        <div className="flex items-center">
          <Link href="/" className="text-xl font-bold text-primary-600">
            InvestRecommend
          </Link>
          <div className="hidden md:flex ml-10">
            <Link href="/dashboard" className="px-3 py-2 text-gray-700 hover:text-primary-600">
              Dashboard
            </Link>
            <Link href="/stocks" className="px-3 py-2 text-gray-700 hover:text-primary-600">
              Stocks
            </Link>
            <Link href="/recommendations" className="px-3 py-2 text-gray-700 hover:text-primary-600">
              Recommendations
            </Link>
            <Link href="/portfolio" className="px-3 py-2 text-gray-700 hover:text-primary-600">
              Portfolio
            </Link>
          </div>
        </div>
        <div className="flex items-center">
          {user ? (
            <div className="flex items-center">
              <span className="mr-2 text-gray-700">Hello, {user.name}</span>
              <button className="px-3 py-2 text-sm bg-gray-100 text-gray-700 rounded-md hover:bg-gray-200">
                Logout
              </button>
            </div>
          ) : (
            <div className="flex items-center">
              <Link href="/login" className="px-3 py-2 text-gray-700 hover:text-primary-600">
                Login
              </Link>
              <Link 
                href="/register"
                className="ml-2 px-3 py-2 text-sm bg-primary-600 text-white rounded-md hover:bg-primary-700"
              >
                Register
              </Link>
            </div>
          )}
        </div>
      </div>
    </nav>
  );
};

export default Navbar; 