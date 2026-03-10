import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import StatsCards from '../components/dashboard/StatsCards';
import TrendChart from '../components/dashboard/TrendChart';
import PieChart from '../components/dashboard/PieChart';
import RecentDetections from '../components/dashboard/RecentDetections';
import { useAlert } from '../components/common/Alert';
import Loader from '../components/common/Loader';

export default function DashboardPage() {
  const [loading, setLoading] = useState(false);
  const { error } = useAlert();

  // Sample data - replace with real API calls later
  const stats = [
    { title: 'Total Scans', value: '1,247', change: '+12.5%' },
    { title: 'Deepfakes Detected', value: '324', change: '+8.2%' },
    { title: 'Authentic Media', value: '923', change: '-3.1%' },
    { title: 'Avg Response', value: '0.8s', change: '-0.2s' },
  ];

  const recentDetections = [
    { id: 'DET-001', type: 'Image', filename: 'profile_photo.jpg', result: 'REAL', confidence: 98.5, date: '2024-01-15' },
    { id: 'DET-002', type: 'Video', filename: 'interview.mp4', result: 'FAKE', confidence: 94.2, date: '2024-01-14' },
    { id: 'DET-003', type: 'Audio', filename: 'voice_message.wav', result: 'FAKE', confidence: 87.3, date: '2024-01-14' },
  ];

  if (loading) {
    return <Loader fullScreen text="Loading dashboard..." />;
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="space-y-8 p-6"
    >
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-white">Dashboard</h1>
          <p className="text-gray-400 mt-1">
            Welcome back! Here's your deepfake detection overview
          </p>
        </div>
        <div className="glass px-4 py-2 rounded-xl">
          <span className="text-gray-400">Last updated:</span>
          <span className="text-white ml-2">{new Date().toLocaleDateString()}</span>
        </div>
      </div>

      {/* Stats Cards - Using your component */}
      <StatsCards stats={stats} />

      {/* Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <TrendChart />
        <PieChart />
      </div>

      {/* Recent Detections - Using your component */}
      <RecentDetections detections={recentDetections} />
    </motion.div>
  );
}