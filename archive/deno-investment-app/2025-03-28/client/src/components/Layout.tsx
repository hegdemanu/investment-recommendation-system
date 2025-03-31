import React, { ReactNode, useState } from 'react';
import { Link, useLocation } from 'react-router-dom';
import * as DropdownMenu from '@radix-ui/react-dropdown-menu';
import { 
  LayoutDashboard, 
  Briefcase, 
  TrendingUp, 
  Settings, 
  Menu, 
  X, 
  Sun, 
  Moon, 
  Bell, 
  User
} from 'lucide-react';

interface LayoutProps {
  children: ReactNode;
  isDarkMode: boolean;
  toggleDarkMode: () => void;
}

const Layout = ({ children, isDarkMode, toggleDarkMode }: LayoutProps) => {
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);
  const location = useLocation();

  const navigation = [
    { name: 'Dashboard', href: '/', icon: LayoutDashboard },
    { name: 'Portfolio', href: '/portfolio', icon: Briefcase },
    { name: 'Recommendations', href: '/recommendations', icon: TrendingUp },
    { name: 'Settings', href: '/settings', icon: Settings },
  ];

  const toggleSidebar = () => {
    setIsSidebarOpen(!isSidebarOpen);
  };

  return (
    <div className="flex h-screen">
      {/* Mobile sidebar toggle */}
      <div className="fixed inset-0 flex z-40 lg:hidden" role="dialog" aria-modal="true">
        <div 
          className={`fixed inset-0 bg-gray-600 bg-opacity-75 transition-opacity ease-in-out duration-300 ${
            isSidebarOpen ? 'opacity-100' : 'opacity-0 pointer-events-none'
          }`} 
          aria-hidden="true"
          onClick={toggleSidebar}
        />

        {/* Sidebar */}
        <div className={`relative flex-1 flex flex-col max-w-xs w-full bg-card transform transition ease-in-out duration-300 ${
          isSidebarOpen ? 'translate-x-0' : '-translate-x-full'
        }`}>
          <div className="absolute top-0 right-0 -mr-12 pt-2">
            <button
              type="button"
              className="ml-1 flex items-center justify-center h-10 w-10 rounded-full focus:outline-none focus:ring-2 focus:ring-inset focus:ring-white"
              onClick={toggleSidebar}
            >
              <span className="sr-only">Close sidebar</span>
              <X className="h-6 w-6 text-white" aria-hidden="true" />
            </button>
          </div>

          <div className="flex-1 h-0 pt-5 pb-4 overflow-y-auto">
            <div className="flex-shrink-0 flex items-center px-4">
              <span className="text-xl font-bold">InvestSage</span>
            </div>
            <nav className="mt-5 px-2 space-y-1">
              {navigation.map((item) => (
                <Link
                  key={item.name}
                  to={item.href}
                  className={`group flex items-center px-2 py-2 text-sm font-medium rounded-md ${
                    location.pathname === item.href
                      ? 'bg-primary text-primary-foreground'
                      : 'text-muted-foreground hover:bg-muted hover:text-foreground'
                  }`}
                >
                  <item.icon
                    className={`mr-3 flex-shrink-0 h-6 w-6 ${
                      location.pathname === item.href
                        ? 'text-primary-foreground'
                        : 'text-muted-foreground group-hover:text-foreground'
                    }`}
                    aria-hidden="true"
                  />
                  {item.name}
                </Link>
              ))}
            </nav>
          </div>
        </div>
      </div>

      {/* Static sidebar for desktop */}
      <div className="hidden lg:flex lg:flex-shrink-0">
        <div className="flex flex-col w-64">
          <div className="flex-1 flex flex-col min-h-0 border-r border-border bg-card">
            <div className="flex-1 flex flex-col pt-5 pb-4 overflow-y-auto">
              <div className="flex items-center flex-shrink-0 px-4">
                <span className="text-xl font-bold">InvestSage</span>
              </div>
              <nav className="mt-5 flex-1 px-2 space-y-1">
                {navigation.map((item) => (
                  <Link
                    key={item.name}
                    to={item.href}
                    className={`group flex items-center px-2 py-2 text-sm font-medium rounded-md ${
                      location.pathname === item.href
                        ? 'bg-primary text-primary-foreground'
                        : 'text-muted-foreground hover:bg-muted hover:text-foreground'
                    }`}
                  >
                    <item.icon
                      className={`mr-3 flex-shrink-0 h-6 w-6 ${
                        location.pathname === item.href
                          ? 'text-primary-foreground'
                          : 'text-muted-foreground group-hover:text-foreground'
                      }`}
                      aria-hidden="true"
                    />
                    {item.name}
                  </Link>
                ))}
              </nav>
            </div>
          </div>
        </div>
      </div>

      <div className="flex flex-col w-0 flex-1 overflow-hidden">
        <div className="relative z-10 flex-shrink-0 flex h-16 bg-card border-b border-border">
          <button
            type="button"
            className="px-4 border-r border-border text-muted-foreground focus:outline-none focus:ring-2 focus:ring-inset focus:ring-primary lg:hidden"
            onClick={toggleSidebar}
          >
            <span className="sr-only">Open sidebar</span>
            <Menu className="h-6 w-6" aria-hidden="true" />
          </button>
          <div className="flex-1 px-4 flex justify-end">
            <div className="ml-4 flex items-center md:ml-6 space-x-4">
              <button 
                onClick={toggleDarkMode}
                className="p-2 rounded-md hover:bg-muted flex items-center justify-center focus:outline-none focus:ring-2 focus:ring-primary transition-colors"
              >
                {isDarkMode ? 
                  <Sun className="h-5 w-5 text-muted-foreground" /> : 
                  <Moon className="h-5 w-5 text-muted-foreground" />
                }
              </button>
            
              <button className="p-2 rounded-md hover:bg-muted flex items-center justify-center focus:outline-none focus:ring-2 focus:ring-primary transition-colors">
                <Bell className="h-5 w-5 text-muted-foreground" />
              </button>

              <DropdownMenu.Root>
                <DropdownMenu.Trigger asChild>
                  <button className="flex items-center max-w-xs rounded-full hover:bg-muted p-2">
                    <span className="sr-only">Open user menu</span>
                    <User className="h-5 w-5 text-muted-foreground" />
                  </button>
                </DropdownMenu.Trigger>
                <DropdownMenu.Portal>
                  <DropdownMenu.Content 
                    className="min-w-[220px] bg-popover rounded-md p-2 shadow-md" 
                    sideOffset={5}
                    align="end"
                  >
                    <DropdownMenu.Item className="text-sm cursor-pointer rounded flex items-center h-9 px-2 py-1.5 select-none outline-none data-[highlighted]:bg-accent data-[highlighted]:text-accent-foreground">
                      Your Profile
                    </DropdownMenu.Item>
                    <DropdownMenu.Item className="text-sm cursor-pointer rounded flex items-center h-9 px-2 py-1.5 select-none outline-none data-[highlighted]:bg-accent data-[highlighted]:text-accent-foreground">
                      Account Settings
                    </DropdownMenu.Item>
                    <DropdownMenu.Separator className="h-px bg-border my-1" />
                    <DropdownMenu.Item className="text-sm cursor-pointer rounded flex items-center h-9 px-2 py-1.5 select-none outline-none data-[highlighted]:bg-accent data-[highlighted]:text-accent-foreground">
                      Sign out
                    </DropdownMenu.Item>
                  </DropdownMenu.Content>
                </DropdownMenu.Portal>
              </DropdownMenu.Root>
            </div>
          </div>
        </div>

        <main className="flex-1 relative overflow-y-auto focus:outline-none">
          <div className="py-6">
            <div className="max-w-7xl mx-auto px-4 sm:px-6 md:px-8">
              {children}
            </div>
          </div>
        </main>
      </div>
    </div>
  );
};

export default Layout; 