import React, { useState } from 'react';
import Link from 'next/link';
import { useRouter } from 'next/router';
import { motion } from 'framer-motion';
import { 
  FiHome,
  FiImage,
  FiVideo,
  FiMusic,
  FiFileText,
  FiLink,
  FiCamera,
  FiSettings,
  FiHelpCircle,
  FiLogOut,
  FiChevronRight,
  FiBarChart2,
  FiShield
} from 'react-icons/fi';

/**
 * Sidebar Component - Main navigation sidebar
 */

const Sidebar = ({ isMobile, setSidebarOpen }) => {
  const router = useRouter();
  const [collapsed, setCollapsed] = useState(false);

  const menuItems = [
    {
      name: 'Dashboard',
      icon: FiHome,
      path: '/dashboard',
      color: 'blue'
    },
    {
      name: 'Image Detection',
      icon: FiImage,
      path: '/image',
      color: 'purple'
    },
    {
      name: 'Video Detection',
      icon: FiVideo,
      path: '/video',
      color: 'red'
    },
    {
      name: 'Audio Detection',
      icon: FiMusic,
      path: '/audio',
      color: 'green'
    },
    {
      name: 'Text Analysis',
      icon: FiFileText,
      path: '/text',
      color: 'yellow'
    },
    {
      name: 'URL Scanner',
      icon: FiLink,
      path: '/url',
      color: 'pink'
    },
    {
      name: 'Live Detection',
      icon: FiCamera,
      path: '/live',
      color: 'orange'
    }
  ];

  const bottomItems = [
    {
      name: 'Settings',
      icon: FiSettings,
      path: '/settings',
      color: 'gray'
    },
    {
      name: 'Help',
      icon: FiHelpCircle,
      path: '/help',
      color: 'gray'
    }
  ];

  const getItemColor = (color, isActive) => {
    const colors = {
      blue: isActive ? 'bg-blue-500/20 text-blue-400 border-blue-500' : 'text-gray-400',
      purple: isActive ? 'bg-purple-500/20 text-purple-400 border-purple-500' : 'text-gray-400',
      red: isActive ? 'bg-red-500/20 text-red-400 border-red-500' : 'text-gray-400',
      green: isActive ? 'bg-green-500/20 text-green-400 border-green-500' : 'text-gray-400',
      yellow: isActive ? 'bg-yellow-500/20 text-yellow-400 border-yellow-500' : 'text-gray-400',
      pink: isActive ? 'bg-pink-500/20 text-pink-400 border-pink-500' : 'text-gray-400',
      orange: isActive ? 'bg-orange-500/20 text-orange-400 border-orange-500' : 'text-gray-400',
      gray: isActive ? 'bg-gray-500/20 text-gray-400 border-gray-500' : 'text-gray-400',
    };
    return colors[color] || colors.blue;
  };

  return (
    <motion.aside
      initial={{ width: collapsed ? 80 : 256 }}
      animate={{ width: collapsed ? 80 : 256 }}
      transition={{ duration: 0.3 }}
      className="h-full glass border-r border-white/10 flex flex-col"
    >
      {/* Collapse Toggle (Desktop only) */}
      {!isMobile && (
        <button
          onClick={() => setCollapsed(!collapsed)}
          className="absolute -right-3 top-20 w-6 h-6 glass rounded-full flex items-center justify-center hover:bg-white/20 z-10"
        >
          <FiChevronRight className={`w-4 h-4 text-white transition-transform ${collapsed ? 'rotate-180' : ''}`} />
        </button>
      )}

      {/* Logo Area (Collapsed) */}
      {collapsed && !isMobile && (
        <div className="py-6 flex justify-center">
          <div className="w-10 h-10 bg-gradient-to-r from-primary-600 to-primary-400 rounded-xl flex items-center justify-center">
            <FiShield className="w-5 h-5 text-white" />
          </div>
        </div>
      )}

      {/* Menu Items */}
      <div className="flex-1 overflow-y-auto py-6 px-3">
        <div className="space-y-1">
          {menuItems.map((item) => {
            const Icon = item.icon;
            const isActive = router.pathname === item.path;
            const colorClass = getItemColor(item.color, isActive);

            return (
              <Link
                key={item.path}
                href={item.path}
                onClick={() => isMobile && setSidebarOpen(false)}
              >
                <motion.div
                  whileHover={{ x: 5 }}
                  whileTap={{ scale: 0.95 }}
                  className={`
                    flex items-center gap-3 px-3 py-3 rounded-xl
                    transition-all duration-300 cursor-pointer relative
                    ${isActive ? colorClass : 'hover:bg-white/10 text-gray-400 hover:text-white'}
                  `}
                >
                  <Icon className={`w-5 h-5 flex-shrink-0 ${isActive ? colorClass : ''}`} />
                  
                  {!collapsed && (
                    <>
                      <span className="text-sm font-medium whitespace-nowrap">{item.name}</span>
                      {isActive && (
                        <motion.div
                          layoutId="activeIndicator"
                          className="absolute right-3 w-1.5 h-6 rounded-full bg-current"
                        />
                      )}
                    </>
                  )}

                  {collapsed && isActive && (
                    <motion.div
                      layoutId="activeIndicatorCollapsed"
                      className="absolute left-0 w-1 h-8 rounded-r-full bg-current"
                    />
                  )}
                </motion.div>
              </Link>
            );
          })}
        </div>

        {/* Bottom Items */}
        <div className="absolute bottom-6 left-3 right-3 space-y-1">
          {bottomItems.map((item) => {
            const Icon = item.icon;
            const isActive = router.pathname === item.path;

            return (
              <Link key={item.path} href={item.path}>
                <motion.div
                  whileHover={{ x: 5 }}
                  whileTap={{ scale: 0.95 }}
                  className={`
                    flex items-center gap-3 px-3 py-3 rounded-xl
                    transition-all duration-300 cursor-pointer
                    ${isActive 
                      ? 'bg-gray-500/20 text-gray-400' 
                      : 'hover:bg-white/10 text-gray-400 hover:text-white'
                    }
                  `}
                >
                  <Icon className="w-5 h-5 flex-shrink-0" />
                  {!collapsed && <span className="text-sm font-medium whitespace-nowrap">{item.name}</span>}
                </motion.div>
              </Link>
            );
          })}

          {/* Logout Button */}
          <motion.button
            whileHover={{ x: 5 }}
            whileTap={{ scale: 0.95 }}
            onClick={() => router.push('/')}
            className="w-full flex items-center gap-3 px-3 py-3 rounded-xl
                       hover:bg-white/10 text-red-400 hover:text-red-300
                       transition-all duration-300 cursor-pointer"
          >
            <FiLogOut className="w-5 h-5 flex-shrink-0" />
            {!collapsed && <span className="text-sm font-medium whitespace-nowrap">Logout</span>}
          </motion.button>

          {/* System Status (when expanded) */}
          {!collapsed && (
            <div className="mt-6 p-4 glass rounded-xl">
              <div className="flex items-center gap-2 mb-3">
                <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse" />
                <span className="text-xs text-gray-400">System Online</span>
              </div>
              <div className="space-y-2">
                <div className="flex justify-between text-xs">
                  <span className="text-gray-400">CPU</span>
                  <span className="text-white">45%</span>
                </div>
                <div className="progress-bar">
                  <div className="progress-fill" style={{ width: '45%' }} />
                </div>
                <div className="flex justify-between text-xs">
                  <span className="text-gray-400">Memory</span>
                  <span className="text-white">6.2/16 GB</span>
                </div>
                <div className="progress-bar">
                  <div className="progress-fill" style={{ width: '38%' }} />
                </div>
              </div>
            </div>
          )}

          {/* Collapsed Status */}
          {collapsed && (
            <div className="mt-6 flex justify-center">
              <div className="w-8 h-8 glass rounded-xl flex items-center justify-center">
                <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse" />
              </div>
            </div>
          )}
        </div>
      </div>
    </motion.aside>
  );
};

export default Sidebar;