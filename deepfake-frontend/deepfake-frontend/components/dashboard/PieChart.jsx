import React from 'react';
import { motion } from 'framer-motion';
import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip, Legend } from 'recharts';
import { FiPieChart } from 'react-icons/fi';

/**
 * DetectionPieChart Component - Displays detection distribution
 * 
 * @param {Object} props
 * @param {Array} props.data - Chart data
 * @param {string} props.title - Chart title
 * @param {boolean} props.loading - Loading state
 */

const DetectionPieChart = ({ 
  data = [], 
  title = 'Detection Distribution',
  loading = false 
}) => {
  
  // Sample data
  const sampleData = [
    { name: 'Images', value: 45, color: '#3b82f6' },
    { name: 'Videos', value: 25, color: '#8b5cf6' },
    { name: 'Audio', value: 15, color: '#10b981' },
    { name: 'Text', value: 10, color: '#f59e0b' },
    { name: 'URLs', value: 5, color: '#ef4444' },
  ];

  const chartData = data.length > 0 ? data : sampleData;

  const RADIAN = Math.PI / 180;
  const renderCustomizedLabel = ({
    cx,
    cy,
    midAngle,
    innerRadius,
    outerRadius,
    percent,
    index,
  }) => {
    const radius = innerRadius + (outerRadius - innerRadius) * 0.5;
    const x = cx + radius * Math.cos(-midAngle * RADIAN);
    const y = cy + radius * Math.sin(-midAngle * RADIAN);

    return (
      <text
        x={x}
        y={y}
        fill="white"
        textAnchor={x > cx ? 'start' : 'end'}
        dominantBaseline="central"
        fontSize={12}
        fontWeight="bold"
      >
        {`${(percent * 100).toFixed(0)}%`}
      </text>
    );
  };

  // Custom tooltip
  const CustomTooltip = ({ active, payload }) => {
    if (active && payload && payload.length) {
      return (
        <div className="glass p-3 rounded-lg border border-white/10">
          <p className="text-white text-sm font-medium mb-1">
            {payload[0].name}
          </p>
          <p className="text-gray-400 text-xs">
            Count: <span className="text-white font-medium">{payload[0].value}</span>
          </p>
          <p className="text-gray-400 text-xs">
            Percentage: <span className="text-white font-medium">
              {((payload[0].value / chartData.reduce((a, b) => a + b.value, 0)) * 100).toFixed(1)}%
            </span>
          </p>
        </div>
      );
    }
    return null;
  };

  if (loading) {
    return (
      <div className="chart-card animate-pulse">
        <div className="h-8 w-48 bg-white/10 rounded mb-6" />
        <div className="h-64 bg-white/5 rounded-full w-64 mx-auto" />
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
      <div className="flex items-center gap-3 mb-6">
        <div className="w-10 h-10 rounded-xl bg-gradient-to-r from-primary-600 to-primary-400 p-2">
          <FiPieChart className="w-6 h-6 text-white" />
        </div>
        <div>
          <h3 className="text-white font-semibold">{title}</h3>
          <p className="text-sm text-gray-400">
            Total: {chartData.reduce((a, b) => a + b.value, 0)} detections
          </p>
        </div>
      </div>

      {/* Chart */}
      <div className="h-64">
        <ResponsiveContainer width="100%" height="100%">
          <PieChart>
            <Pie
              data={chartData}
              cx="50%"
              cy="50%"
              labelLine={false}
              label={renderCustomizedLabel}
              outerRadius={80}
              fill="#8884d8"
              dataKey="value"
            >
              {chartData.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={entry.color} />
              ))}
            </Pie>
            <Tooltip content={<CustomTooltip />} />
            <Legend 
              layout="vertical" 
              align="right"
              verticalAlign="middle"
              wrapperStyle={{
                color: '#9ca3af',
                fontSize: '12px',
              }}
            />
          </PieChart>
        </ResponsiveContainer>
      </div>

      {/* Legend List (Mobile friendly) */}
      <div className="grid grid-cols-2 gap-4 mt-6 pt-6 border-t border-white/10 lg:hidden">
        {chartData.map((item, index) => (
          <div key={index} className="flex items-center gap-2">
            <div 
              className="w-3 h-3 rounded-full" 
              style={{ backgroundColor: item.color }}
            />
            <span className="text-sm text-gray-300">{item.name}</span>
            <span className="text-sm text-white font-medium ml-auto">
              {item.value}
            </span>
          </div>
        ))}
      </div>
    </motion.div>
  );
};

// ============================================
// Donut Chart Variant
// ============================================

export const DonutChart = ({ data = [], title, centerText }) => {
  const total = data.reduce((a, b) => a + b.value, 0);

  return (
    <div className="relative">
      <ResponsiveContainer width="100%" height={200}>
        <PieChart>
          <Pie
            data={data}
            cx="50%"
            cy="50%"
            innerRadius={60}
            outerRadius={80}
            fill="#8884d8"
            paddingAngle={5}
            dataKey="value"
          >
            {data.map((entry, index) => (
              <Cell key={`cell-${index}`} fill={entry.color} />
            ))}
          </Pie>
          <Tooltip />
        </PieChart>
      </ResponsiveContainer>
      
      {/* Center text */}
      <div className="absolute inset-0 flex items-center justify-center">
        <div className="text-center">
          <p className="text-2xl font-bold text-white">{total}</p>
          <p className="text-xs text-gray-400">Total</p>
        </div>
      </div>
    </div>
  );
};

// ============================================
// Progress Circle Chart
// ============================================

export const ProgressCircle = ({ percentage, color = 'primary', size = 120 }) => {
  const colors = {
    primary: '#3b82f6',
    red: '#ef4444',
    green: '#10b981',
    yellow: '#f59e0b',
    purple: '#8b5cf6',
  };

  return (
    <div className="relative" style={{ width: size, height: size }}>
      <svg className="transform -rotate-90" width={size} height={size}>
        {/* Background circle */}
        <circle
          cx={size / 2}
          cy={size / 2}
          r={size * 0.4}
          fill="none"
          stroke="rgba(255,255,255,0.1)"
          strokeWidth="8"
        />
        
        {/* Progress circle */}
        <circle
          cx={size / 2}
          cy={size / 2}
          r={size * 0.4}
          fill="none"
          stroke={colors[color]}
          strokeWidth="8"
          strokeLinecap="round"
          strokeDasharray={`${2 * Math.PI * size * 0.4}`}
          strokeDashoffset={`${2 * Math.PI * size * 0.4 * (1 - percentage / 100)}`}
          style={{ transition: 'stroke-dashoffset 0.5s' }}
        />
      </svg>
      
      {/* Percentage text */}
      <div className="absolute inset-0 flex items-center justify-center">
        <div className="text-center">
          <p className="text-2xl font-bold text-white">{percentage}%</p>
        </div>
      </div>
    </div>
  );
};

export default DetectionPieChart;