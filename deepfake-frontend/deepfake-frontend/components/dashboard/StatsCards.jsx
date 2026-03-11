import React from 'react';
import { motion } from 'framer-motion';
import { 
  FiShield, 
  FiAlertTriangle, 
  FiCheckCircle, 
  FiClock,
  FiEye,
  FiFileText,
  FiVideo,
  FiMusic 
} from 'react-icons/fi';

/**
 * StatsCards Component - Displays statistics in card format
 * 
 * @param {Object} props
 * @param {Array} props.stats - Array of stat objects
 * @param {boolean} props.loading - Loading state
 */

const StatsCards = ({ stats = [], loading = false }) => {
  
  // Default stats if none provided
  const defaultStats = [
    {
      title: 'Total Scans',
      value: '1,247',
      change: '+12.5%',
      icon: FiShield,
      color: 'blue',
      trend: 'up'
    },
    {
      title: 'Deepfakes Detected',
      value: '324',
      change: '+8.2%',
      icon: FiAlertTriangle,
      color: 'red',
      trend: 'up'
    },
    {
      title: 'Authentic Media',
      value: '923',
      change: '-3.1%',
      icon: FiCheckCircle,
      color: 'green',
      trend: 'down'
    },
    {
      title: 'Avg Response',
      value: '0.8s',
      change: '-0.2s',
      icon: FiClock,
      color: 'purple',
      trend: 'down'
    }
  ];

  const displayStats = stats.length > 0 ? stats : defaultStats;

  // Color mappings
  const colorClasses = {
    blue: {
      bg: 'bg-blue-500/10',
      text: 'text-blue-400',
      border: 'border-blue-500/20',
      gradient: 'from-blue-600 to-blue-400'
    },
    red: {
      bg: 'bg-red-500/10',
      text: 'text-red-400',
      border: 'border-red-500/20',
      gradient: 'from-red-600 to-red-400'
    },
    green: {
      bg: 'bg-green-500/10',
      text: 'text-green-400',
      border: 'border-green-500/20',
      gradient: 'from-green-600 to-green-400'
    },
    purple: {
      bg: 'bg-purple-500/10',
      text: 'text-purple-400',
      border: 'border-purple-500/20',
      gradient: 'from-purple-600 to-purple-400'
    },
    yellow: {
      bg: 'bg-yellow-500/10',
      text: 'text-yellow-400',
      border: 'border-yellow-500/20',
      gradient: 'from-yellow-600 to-yellow-400'
    }
  };

  // Loading skeletons
  if (loading) {
    return (
      <div className="stats-grid">
        {[1, 2, 3, 4].map((i) => (
          <div key={i} className="glass-card p-6 animate-pulse">
            <div className="flex items-center justify-between mb-4">
              <div className="w-12 h-12 bg-white/10 rounded-xl" />
              <div className="w-16 h-4 bg-white/10 rounded" />
            </div>
            <div className="w-24 h-4 bg-white/10 rounded mb-2" />
            <div className="w-32 h-8 bg-white/10 rounded" />
          </div>
        ))}
      </div>
    );
  }

  return (
    <div className="stats-grid">
      {displayStats.map((stat, index) => {
        const Icon = stat.icon;
        const colors = colorClasses[stat.color] || colorClasses.blue;
        const trendColor = stat.trend === 'up' ? 'text-green-400' : 'text-red-400';
        const trendIcon = stat.trend === 'up' ? '↑' : '↓';

        return (
          <motion.div
            key={index}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: index * 0.1 }}
            whileHover={{ y: -5 }}
            className={`stat-card ${colors.bg} ${colors.border} border`}
          >
            <div className="flex items-center justify-between mb-4">
              <div className={`w-12 h-12 rounded-xl ${colors.bg} flex items-center justify-center`}>
                <Icon className={`w-6 h-6 ${colors.text}`} />
              </div>
              <span className={`text-sm ${trendColor}`}>
                {trendIcon} {stat.change}
              </span>
            </div>
            
            <h3 className="text-gray-400 text-sm">{stat.title}</h3>
            <p className="text-3xl font-bold text-white mt-1">{stat.value}</p>
            
            {/* Mini progress bar */}
            <div className="mt-4 progress-bar">
              <div 
                className={`progress-fill bg-gradient-to-r ${colors.gradient}`}
                style={{ width: `${Math.random() * 40 + 60}%` }}
              />
            </div>
          </motion.div>
        );
      })}
    </div>
  );
};

// ============================================
// Detailed Stats Card Component
// ============================================

export const DetailedStatsCard = ({ title, stats, icon: Icon, color = 'blue' }) => {
  const colorClasses = {
    blue: 'from-blue-600 to-blue-400',
    red: 'from-red-600 to-red-400',
    green: 'from-green-600 to-green-400',
    purple: 'from-purple-600 to-purple-400',
  };

  return (
    <motion.div
      whileHover={{ y: -5 }}
      className="glass-card p-6"
    >
      <div className="flex items-center gap-3 mb-6">
        <div className={`w-10 h-10 rounded-xl bg-gradient-to-r ${colorClasses[color]} p-2`}>
          <Icon className="w-6 h-6 text-white" />
        </div>
        <h3 className="text-white font-semibold">{title}</h3>
      </div>

      <div className="space-y-4">
        {stats.map((stat, index) => (
          <div key={index} className="flex items-center justify-between">
            <span className="text-gray-400 text-sm">{stat.label}</span>
            <div className="flex items-center gap-3">
              <span className="text-white font-semibold">{stat.value}</span>
              {stat.change && (
                <span className={stat.change > 0 ? 'text-green-400' : 'text-red-400'}>
                  {stat.change > 0 ? '+' : ''}{stat.change}%
                </span>
              )}
            </div>
          </div>
        ))}
      </div>
    </motion.div>
  );
};

// ============================================
// Detection Type Stats Component
// ============================================

export const DetectionTypeStats = ({ stats = [] }) => {
  const types = [
    { name: 'Images', icon: FiEye, color: 'blue', value: 45 },
    { name: 'Videos', icon: FiVideo, color: 'purple', value: 25 },
    { name: 'Audio', icon: FiMusic, color: 'green', value: 15 },
    { name: 'Text', icon: FiFileText, color: 'yellow', value: 10 },
    { name: 'URLs', icon: FiShield, color: 'red', value: 5 },
  ];

  const displayStats = stats.length > 0 ? stats : types;

  return (
    <div className="glass-card p-6">
      <h3 className="text-white font-semibold mb-4">Detection by Type</h3>
      <div className="space-y-4">
        {displayStats.map((type, index) => {
          const Icon = type.icon;
          const percentage = type.value;

          return (
            <div key={index} className="space-y-2">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Icon className={`w-4 h-4 text-${type.color}-400`} />
                  <span className="text-gray-300 text-sm">{type.name}</span>
                </div>
                <span className="text-white font-medium">{percentage}%</span>
              </div>
              <div className="progress-bar">
                <div 
                  className={`progress-fill bg-${type.color}-500`}
                  style={{ width: `${percentage}%` }}
                />
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};

export default StatsCards;