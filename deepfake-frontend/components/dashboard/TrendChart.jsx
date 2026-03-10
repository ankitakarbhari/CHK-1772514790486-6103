import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend
} from 'recharts';
import { FiCalendar, FiDownload, FiRefreshCw } from 'react-icons/fi';
import Button from '@/components/common/Button';

/**
 * TrendChart Component - Displays detection trends over time
 * 
 * @param {Object} props
 * @param {Array} props.data - Chart data
 * @param {string} props.title - Chart title
 * @param {boolean} props.loading - Loading state
 */

const TrendChart = ({ 
  data = [], 
  title = 'Detection Trends',
  loading = false,
  onRefresh,
  onDownload 
}) => {
  const [timeRange, setTimeRange] = useState('week');
  const [chartData, setChartData] = useState([]);

  // Sample data
  const sampleData = [
    { name: 'Mon', deepfakes: 12, authentic: 45, suspicious: 8 },
    { name: 'Tue', deepfakes: 19, authentic: 52, suspicious: 11 },
    { name: 'Wed', deepfakes: 15, authentic: 48, suspicious: 9 },
    { name: 'Thu', deepfakes: 17, authentic: 51, suspicious: 13 },
    { name: 'Fri', deepfakes: 14, authentic: 49, suspicious: 7 },
    { name: 'Sat', deepfakes: 13, authentic: 53, suspicious: 10 },
    { name: 'Sun', deepfakes: 18, authentic: 47, suspicious: 12 },
  ];

  useEffect(() => {
    setChartData(data.length > 0 ? data : sampleData);
  }, [data, timeRange]);

  const timeRanges = [
    { value: 'day', label: 'Today' },
    { value: 'week', label: 'This Week' },
    { value: 'month', label: 'This Month' },
    { value: 'year', label: 'This Year' },
  ];

  // Custom tooltip
  const CustomTooltip = ({ active, payload, label }) => {
    if (active && payload && payload.length) {
      return (
        <div className="glass p-3 rounded-lg border border-white/10">
          <p className="text-white text-sm font-medium mb-2">{label}</p>
          {payload.map((entry, index) => (
            <div key={index} className="flex items-center gap-2 text-xs">
              <div 
                className="w-2 h-2 rounded-full" 
                style={{ backgroundColor: entry.color }}
              />
              <span className="text-gray-400">{entry.name}:</span>
              <span className="text-white font-medium">{entry.value}</span>
            </div>
          ))}
        </div>
      );
    }
    return null;
  };

  if (loading) {
    return (
      <div className="chart-card animate-pulse">
        <div className="h-8 w-48 bg-white/10 rounded mb-6" />
        <div className="h-64 bg-white/5 rounded" />
      </div>
    );
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="chart-card"
    >
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between mb-6 gap-4">
        <div>
          <h3 className="text-white font-semibold">{title}</h3>
          <p className="text-sm text-gray-400 mt-1">
            Deepfake detection activity over time
          </p>
        </div>

        <div className="flex items-center gap-2">
          {/* Time range selector */}
          <div className="flex glass rounded-lg p-1">
            {timeRanges.map((range) => (
              <button
                key={range.value}
                onClick={() => setTimeRange(range.value)}
                className={`
                  px-3 py-1.5 text-xs font-medium rounded-md transition-all
                  ${timeRange === range.value 
                    ? 'bg-primary-500 text-white' 
                    : 'text-gray-400 hover:text-white hover:bg-white/10'
                  }
                `}
              >
                {range.label}
              </button>
            ))}
          </div>

          {/* Action buttons */}
          <Button
            variant="ghost"
            size="sm"
            icon={<FiRefreshCw className="w-4 h-4" />}
            onClick={onRefresh}
          />
          <Button
            variant="ghost"
            size="sm"
            icon={<FiDownload className="w-4 h-4" />}
            onClick={onDownload}
          />
        </div>
      </div>

      {/* Chart */}
      <div className="h-80">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart
            data={chartData}
            margin={{ top: 10, right: 30, left: 0, bottom: 0 }}
          >
            <defs>
              <linearGradient id="deepfakesGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#ef4444" stopOpacity={0.3}/>
                <stop offset="95%" stopColor="#ef4444" stopOpacity={0}/>
              </linearGradient>
              <linearGradient id="authenticGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#10b981" stopOpacity={0.3}/>
                <stop offset="95%" stopColor="#10b981" stopOpacity={0}/>
              </linearGradient>
              <linearGradient id="suspiciousGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#f59e0b" stopOpacity={0.3}/>
                <stop offset="95%" stopColor="#f59e0b" stopOpacity={0}/>
              </linearGradient>
            </defs>

            <CartesianGrid 
              strokeDasharray="3 3" 
              stroke="rgba(255,255,255,0.1)"
              vertical={false}
            />

            <XAxis 
              dataKey="name" 
              stroke="#9ca3af"
              tick={{ fill: '#9ca3af', fontSize: 12 }}
              axisLine={{ stroke: 'rgba(255,255,255,0.1)' }}
            />

            <YAxis 
              stroke="#9ca3af"
              tick={{ fill: '#9ca3af', fontSize: 12 }}
              axisLine={{ stroke: 'rgba(255,255,255,0.1)' }}
            />

            <Tooltip content={<CustomTooltip />} />

            <Legend 
              wrapperStyle={{ color: '#9ca3af', paddingTop: 20 }}
              iconType="circle"
            />

            <Area
              type="monotone"
              dataKey="deepfakes"
              name="Deepfakes"
              stroke="#ef4444"
              fill="url(#deepfakesGradient)"
              strokeWidth={2}
            />

            <Area
              type="monotone"
              dataKey="authentic"
              name="Authentic"
              stroke="#10b981"
              fill="url(#authenticGradient)"
              strokeWidth={2}
            />

            <Area
              type="monotone"
              dataKey="suspicious"
              name="Suspicious"
              stroke="#f59e0b"
              fill="url(#suspiciousGradient)"
              strokeWidth={2}
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>

      {/* Summary stats */}
      <div className="grid grid-cols-3 gap-4 mt-6 pt-6 border-t border-white/10">
        <div className="text-center">
          <p className="text-2xl font-bold text-white">
            {chartData.reduce((acc, curr) => acc + curr.deepfakes, 0)}
          </p>
          <p className="text-xs text-gray-400">Total Deepfakes</p>
        </div>
        <div className="text-center">
          <p className="text-2xl font-bold text-white">
            {chartData.reduce((acc, curr) => acc + curr.authentic, 0)}
          </p>
          <p className="text-xs text-gray-400">Total Authentic</p>
        </div>
        <div className="text-center">
          <p className="text-2xl font-bold text-white">
            {chartData.reduce((acc, curr) => acc + curr.suspicious, 0)}
          </p>
          <p className="text-xs text-gray-400">Total Suspicious</p>
        </div>
      </div>
    </motion.div>
  );
};

// ============================================
// Mini Trend Chart (Small version for cards)
// ============================================

export const MiniTrendChart = ({ data = [], color = 'primary' }) => {
  const colors = {
    primary: '#3b82f6',
    red: '#ef4444',
    green: '#10b981',
    purple: '#8b5cf6',
  };

  return (
    <div className="h-16">
      <ResponsiveContainer width="100%" height="100%">
        <AreaChart data={data}>
          <Area
            type="monotone"
            dataKey="value"
            stroke={colors[color]}
            fill={colors[color]}
            fillOpacity={0.2}
            strokeWidth={2}
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
};

export default TrendChart;