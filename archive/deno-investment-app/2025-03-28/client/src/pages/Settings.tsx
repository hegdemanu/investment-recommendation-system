import React, { useState } from 'react';
import * as Tabs from '@radix-ui/react-tabs';
import * as Slider from '@radix-ui/react-slider';
import { User, Shield, Bell, DollarSign, Check } from 'lucide-react';

const RiskLevelSlider = ({ value, onChange }: { value: number, onChange: (value: number) => void }) => {
  return (
    <div className="mt-4">
      <Slider.Root
        className="relative flex items-center select-none touch-none w-full h-5"
        value={[value]}
        onValueChange={(values) => onChange(values[0])}
        max={10}
        step={1}
      >
        <Slider.Track className="bg-muted relative grow rounded-full h-2">
          <Slider.Range className="absolute bg-primary rounded-full h-full" />
        </Slider.Track>
        <Slider.Thumb
          className="block w-5 h-5 bg-background border-2 border-primary rounded-full focus:outline-none focus:ring-2 focus:ring-primary focus:ring-offset-2"
          aria-label="Risk level"
        />
      </Slider.Root>
      <div className="flex justify-between text-sm text-muted-foreground mt-2">
        <span>Conservative</span>
        <span>Moderate</span>
        <span>Aggressive</span>
      </div>
    </div>
  );
};

const Settings = () => {
  const [isLoading, setIsLoading] = useState(false);
  const [activeTab, setActiveTab] = useState('profile');
  const [riskLevel, setRiskLevel] = useState(7);
  const [investmentGoals, setInvestmentGoals] = useState(['growth', 'income']);
  const [emailNotifications, setEmailNotifications] = useState(true);
  const [pushNotifications, setPushNotifications] = useState(true);
  const [marketAlerts, setMarketAlerts] = useState(true);
  
  const handleFormSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    setIsLoading(true);
    
    // Simulate API call
    setTimeout(() => {
      setIsLoading(false);
      alert('Settings saved successfully!');
    }, 1000);
  };

  const handleGoalToggle = (goal: string) => {
    if (investmentGoals.includes(goal)) {
      setInvestmentGoals(investmentGoals.filter(g => g !== goal));
    } else {
      setInvestmentGoals([...investmentGoals, goal]);
    }
  };

  const [notificationSettings, setNotificationSettings] = useState({
    emailAlerts: true,
    pushNotifications: false,
    weeklyReports: true,
    marketUpdates: true
  });

  const [displaySettings, setDisplaySettings] = useState({
    darkMode: false,
    compactView: false,
    showPerformanceMetrics: true
  });

  const handleNotificationChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const { name, checked } = event.target;
    setNotificationSettings(prev => ({
      ...prev,
      [name]: checked
    }));
  };

  const handleDisplayChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const { name, checked } = event.target;
    setDisplaySettings(prev => ({
      ...prev,
      [name]: checked
    }));
  };

  return (
    <div className="container mx-auto p-4">
      <h1 className="text-2xl font-bold mb-6">Settings</h1>
      
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="card">
          <div className="card-header">
            <h2 className="card-title">Notification Preferences</h2>
            <p className="card-description">Manage how you receive updates and alerts</p>
          </div>
          <div className="card-body">
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <div>
                  <h3 className="font-medium">Email Alerts</h3>
                  <p className="text-sm text-muted-foreground">Get important updates via email</p>
                </div>
                <label className="relative inline-flex items-center cursor-pointer">
                  <input 
                    type="checkbox" 
                    name="emailAlerts"
                    checked={notificationSettings.emailAlerts}
                    onChange={handleNotificationChange}
                    className="sr-only peer"
                  />
                  <div className="w-11 h-6 bg-muted rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-primary"></div>
                </label>
              </div>
              
              <div className="flex items-center justify-between">
                <div>
                  <h3 className="font-medium">Push Notifications</h3>
                  <p className="text-sm text-muted-foreground">Receive alerts on your device</p>
                </div>
                <label className="relative inline-flex items-center cursor-pointer">
                  <input 
                    type="checkbox" 
                    name="pushNotifications"
                    checked={notificationSettings.pushNotifications}
                    onChange={handleNotificationChange}
                    className="sr-only peer"
                  />
                  <div className="w-11 h-6 bg-muted rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-primary"></div>
                </label>
              </div>
              
              <div className="flex items-center justify-between">
                <div>
                  <h3 className="font-medium">Weekly Reports</h3>
                  <p className="text-sm text-muted-foreground">Get a weekly summary of your portfolio</p>
                </div>
                <label className="relative inline-flex items-center cursor-pointer">
                  <input 
                    type="checkbox" 
                    name="weeklyReports"
                    checked={notificationSettings.weeklyReports}
                    onChange={handleNotificationChange}
                    className="sr-only peer"
                  />
                  <div className="w-11 h-6 bg-muted rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-primary"></div>
                </label>
              </div>
              
              <div className="flex items-center justify-between">
                <div>
                  <h3 className="font-medium">Market Updates</h3>
                  <p className="text-sm text-muted-foreground">Get notified about significant market events</p>
                </div>
                <label className="relative inline-flex items-center cursor-pointer">
                  <input 
                    type="checkbox" 
                    name="marketUpdates"
                    checked={notificationSettings.marketUpdates}
                    onChange={handleNotificationChange}
                    className="sr-only peer"
                  />
                  <div className="w-11 h-6 bg-muted rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-primary"></div>
                </label>
              </div>
            </div>
          </div>
          <div className="card-footer">
            <button className="btn btn-primary">Save Preferences</button>
          </div>
        </div>
        
        <div className="card">
          <div className="card-header">
            <h2 className="card-title">Display Settings</h2>
            <p className="card-description">Customize your interface</p>
          </div>
          <div className="card-body">
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <div>
                  <h3 className="font-medium">Dark Mode</h3>
                  <p className="text-sm text-muted-foreground">Switch to dark theme</p>
                </div>
                <label className="relative inline-flex items-center cursor-pointer">
                  <input 
                    type="checkbox" 
                    name="darkMode"
                    checked={displaySettings.darkMode}
                    onChange={handleDisplayChange}
                    className="sr-only peer"
                  />
                  <div className="w-11 h-6 bg-muted rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-primary"></div>
                </label>
              </div>
              
              <div className="flex items-center justify-between">
                <div>
                  <h3 className="font-medium">Compact View</h3>
                  <p className="text-sm text-muted-foreground">Show more content with less spacing</p>
                </div>
                <label className="relative inline-flex items-center cursor-pointer">
                  <input 
                    type="checkbox" 
                    name="compactView"
                    checked={displaySettings.compactView}
                    onChange={handleDisplayChange}
                    className="sr-only peer"
                  />
                  <div className="w-11 h-6 bg-muted rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-primary"></div>
                </label>
              </div>
              
              <div className="flex items-center justify-between">
                <div>
                  <h3 className="font-medium">Performance Metrics</h3>
                  <p className="text-sm text-muted-foreground">Show detailed performance analytics</p>
                </div>
                <label className="relative inline-flex items-center cursor-pointer">
                  <input 
                    type="checkbox" 
                    name="showPerformanceMetrics"
                    checked={displaySettings.showPerformanceMetrics}
                    onChange={handleDisplayChange}
                    className="sr-only peer"
                  />
                  <div className="w-11 h-6 bg-muted rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-primary"></div>
                </label>
              </div>
            </div>
          </div>
          <div className="card-footer flex justify-between">
            <button className="btn btn-outline">Reset to Default</button>
            <button className="btn btn-primary">Save Settings</button>
          </div>
        </div>
        
        <div className="card">
          <div className="card-header">
            <h2 className="card-title">Account Information</h2>
            <p className="card-description">Manage your profile and account settings</p>
          </div>
          <div className="card-body">
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium mb-1" htmlFor="name">Full Name</label>
                <input 
                  type="text" 
                  id="name" 
                  className="input"
                  defaultValue="John Doe"
                />
              </div>
              
              <div>
                <label className="block text-sm font-medium mb-1" htmlFor="email">Email Address</label>
                <input 
                  type="email" 
                  id="email" 
                  className="input"
                  defaultValue="john.doe@example.com"
                />
              </div>
              
              <div>
                <label className="block text-sm font-medium mb-1" htmlFor="phone">Phone Number</label>
                <input 
                  type="tel" 
                  id="phone" 
                  className="input"
                  defaultValue="+1 (555) 123-4567"
                />
              </div>
            </div>
          </div>
          <div className="card-footer">
            <button className="btn btn-primary">Update Profile</button>
          </div>
        </div>
        
        <div className="card">
          <div className="card-header">
            <h2 className="card-title">Security</h2>
            <p className="card-description">Manage your password and security settings</p>
          </div>
          <div className="card-body">
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium mb-1" htmlFor="current-password">Current Password</label>
                <input 
                  type="password" 
                  id="current-password" 
                  className="input"
                />
              </div>
              
              <div>
                <label className="block text-sm font-medium mb-1" htmlFor="new-password">New Password</label>
                <input 
                  type="password" 
                  id="new-password" 
                  className="input"
                />
              </div>
              
              <div>
                <label className="block text-sm font-medium mb-1" htmlFor="confirm-password">Confirm New Password</label>
                <input 
                  type="password" 
                  id="confirm-password" 
                  className="input"
                />
              </div>
            </div>
          </div>
          <div className="card-footer">
            <button className="btn btn-primary">Change Password</button>
          </div>
        </div>
      </div>
      
      <div className="mt-8">
        <div className="card bg-destructive/5 border-destructive/20">
          <div className="card-header">
            <h2 className="card-title text-destructive">Danger Zone</h2>
            <p className="card-description">Irreversible actions</p>
          </div>
          <div className="card-body">
            <p className="text-sm mb-4">Once you delete your account, there is no going back. Please be certain.</p>
            <button className="btn btn-destructive">Delete Account</button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Settings; 