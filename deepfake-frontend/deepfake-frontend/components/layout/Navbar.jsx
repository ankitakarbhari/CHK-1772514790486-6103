import React, { useState, useEffect } from 'react';
import Link from 'next/link';
import { useRouter } from 'next/router';
import { motion } from 'framer-motion';  // ← THIS WAS MISSING
import { 
  FiMenu, 
  FiBell, 
  FiUser, 
  FiSearch,
  FiSettings,
  FiLogOut,
  FiMoon,
  FiSun,
  FiChevronDown
} from 'react-icons/fi';

const Navbar = ({ sidebarOpen, setSidebarOpen, isMobile }) => {
  const router = useRouter();
  const [scrolled, setScrolled] = useState(false);
  const [showNotifications, setShowNotifications] = useState(false);
  const [showProfile, setShowProfile] = useState(false);
  const [isDark, setIsDark] = useState(true);

  // Sample notifications
  const notifications = [
    {
      id: 1,
      title: 'Deepfake Detected',
      message: 'Video analysis completed - Fake detected',
      time: '2 min ago',
      read: false,
      type: 'warning'
    },
    {
      id: 2,
      title: 'Analysis Complete',
      message: 'Your image has been processed',
      time: '15 min ago',
      read: false,
      type: 'success'
    },
    {
      id: 3,
      title: 'Model Updated',
      message: 'New deepfake detection model available',
      time: '1 hour ago',
      read: true,
      type: 'info'
    }
  ];

  useEffect(() => {
    const handleScroll = () => {
      setScrolled(window.scrollY > 20);
    };
    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  const unreadCount = notifications.filter(n => !n.read).length;

  const toggleTheme = () => {
    setIsDark(!isDark);
    document.documentElement.classList.toggle('dark');
  };

  const getNotificationIcon = (type) => {
    switch(type) {
      case 'warning': return '🔴';
      case 'success': return '🟢';
      case 'info': return '🔵';
      default: return '⚪';
    }
  };

  return (
    <nav
      className={`fixed top-0 z-40 w-full transition-all duration-300 ${
        scrolled
          ? 'glass border-b border-white/10 py-2'
          : 'bg-transparent py-4'
      }`}
    >
      <div className="container mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between">
          {/* Left Section */}
          <div className="flex items-center gap-4">
            {/* Mobile menu button */}
            {isMobile && (
              <button
                onClick={() => setSidebarOpen(!sidebarOpen)}
                className="p-2 glass rounded-xl hover:bg-white/20 transition-all duration-300"
              >
                <FiMenu className="w-5 h-5 text-white" />
              </button>
            )}

            {/* Logo */}
            <Link href="/dashboard" className="flex items-center gap-2">
              <motion.div
                whileHover={{ rotate: 360 }}
                transition={{ duration: 0.5 }}
                className="w-10 h-10 bg-gradient-to-r from-primary-600 to-primary-400 
                           rounded-xl flex items-center justify-center"
              >
                <span className="text-xl font-bold text-white">D</span>
              </motion.div>
              <span className="text-xl font-bold text-white hidden sm:block">
                Deep<span className="text-primary-400">Shield</span>
              </span>
            </Link>

            {/* Search Bar */}
            <div className="hidden lg:flex items-center glass rounded-xl px-4 py-2 w-96">
              <FiSearch className="w-5 h-5 text-gray-400 mr-2" />
              <input
                type="text"
                placeholder="Search scans, files, or URLs..."
                className="bg-transparent border-none focus:outline-none text-white w-full text-sm"
              />
            </div>
          </div>

          {/* Right Section */}
          <div className="flex items-center gap-3">
            {/* Theme Toggle */}
            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={toggleTheme}
              className="p-2 glass rounded-xl hover:bg-white/20 transition-all duration-300"
            >
              {isDark ? (
                <FiSun className="w-5 h-5 text-yellow-400" />
              ) : (
                <FiMoon className="w-5 h-5 text-gray-400" />
              )}
            </motion.button>

            {/* Notifications */}
            <div className="relative">
              <button
                onClick={() => setShowNotifications(!showNotifications)}
                className="p-2 glass rounded-xl hover:bg-white/20 transition-all duration-300 relative"
              >
                <FiBell className="w-5 h-5 text-white" />
                {unreadCount > 0 && (
                  <span className="absolute top-1 right-1 w-2 h-2 bg-red-500 rounded-full" />
                )}
              </button>

              {/* Notifications Dropdown */}
              {showNotifications && (
                <motion.div
                  initial={{ opacity: 0, y: -10 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -10 }}
                  className="absolute right-0 mt-2 w-80 glass rounded-xl shadow-xl z-50"
                >
                  <div className="p-4 border-b border-white/10">
                    <div className="flex items-center justify-between">
                      <h3 className="text-white font-semibold">Notifications</h3>
                      <span className="text-xs text-gray-400">{unreadCount} new</span>
                    </div>
                  </div>
                  <div className="max-h-96 overflow-y-auto">
                    {notifications.map((notif) => (
                      <div
                        key={notif.id}
                        className={`p-4 border-b border-white/10 hover:bg-white/5 cursor-pointer transition-colors ${
                          !notif.read ? 'bg-primary-500/5' : ''
                        }`}
                      >
                        <div className="flex items-start gap-3">
                          <span className="text-lg">{getNotificationIcon(notif.type)}</span>
                          <div className="flex-1">
                            <p className="text-white text-sm font-medium">{notif.title}</p>
                            <p className="text-gray-400 text-xs mt-1">{notif.message}</p>
                            <p className="text-gray-500 text-xs mt-2">{notif.time}</p>
                          </div>
                          {!notif.read && (
                            <span className="w-2 h-2 bg-primary-500 rounded-full" />
                          )}
                        </div>
                      </div>
                    ))}
                  </div>
                  <div className="p-3 text-center border-t border-white/10">
                    <button className="text-xs text-primary-400 hover:text-primary-300">
                      Mark all as read
                    </button>
                  </div>
                </motion.div>
              )}
            </div>

            {/* Profile */}
            <div className="relative">
              <button
                onClick={() => setShowProfile(!showProfile)}
                className="flex items-center gap-2 p-2 glass rounded-xl hover:bg-white/20 transition-all duration-300"
              >
                <div className="w-8 h-8 bg-gradient-to-r from-primary-600 to-primary-400 
                              rounded-lg flex items-center justify-center">
                  <FiUser className="w-4 h-4 text-white" />
                </div>
                <span className="hidden lg:block text-white">Admin</span>
                <FiChevronDown className="w-4 h-4 text-gray-400 hidden lg:block" />
              </button>

              {/* Profile Dropdown */}
              {showProfile && (
                <motion.div
                  initial={{ opacity: 0, y: -10 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -10 }}
                  className="absolute right-0 mt-2 w-48 glass rounded-xl shadow-xl z-50"
                >
                  <div className="p-2">
                    <div className="p-3 border-b border-white/10">
                      <p className="text-white text-sm font-medium">Admin User</p>
                      <p className="text-gray-400 text-xs">admin@deepshield.com</p>
                    </div>
                    <Link href="/profile">
                      <div className="flex items-center gap-2 p-3 text-gray-300 hover:bg-white/10 rounded-lg cursor-pointer">
                        <FiUser className="w-4 h-4" />
                        <span className="text-sm">Profile</span>
                      </div>
                    </Link>
                    <Link href="/settings">
                      <div className="flex items-center gap-2 p-3 text-gray-300 hover:bg-white/10 rounded-lg cursor-pointer">
                        <FiSettings className="w-4 h-4" />
                        <span className="text-sm">Settings</span>
                      </div>
                    </Link>
                    <div className="border-t border-white/10 my-2" />
                    <button
                      onClick={() => router.push('/')}
                      className="flex items-center gap-2 p-3 text-red-400 hover:bg-white/10 rounded-lg w-full"
                    >
                      <FiLogOut className="w-4 h-4" />
                      <span className="text-sm">Logout</span>
                    </button>
                  </div>
                </motion.div>
              )}
            </div>
          </div>
        </div>
      </div>
    </nav>
  );
};

export default Navbar;