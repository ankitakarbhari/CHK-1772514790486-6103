import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import StatsCards from '@/components/dashboard/StatsCards';
import TrendChart from '@/components/dashboard/TrendChart';
import PieChart from '@/components/dashboard/PieChart';
import RecentDetections from '@/components/dashboard/RecentDetections';
import { getRecentDetections, getStats } from '@/utils/api';
import { useAlert } from '@/components/common/Alert';
import Loader from '@/components/common/Loader';

export default function Dashboard() {
  const [detections, setDetections] = useState([]);
  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(true);
  const { error } = useAlert();

  useEffect(() => {
    fetchData();
  }, []);

  const fetchData = async () => {
    try {
      const [detectionsData, statsData] = await Promise.all([
        getRecentDetections(20),
        getStats()
      ]);
      setDetections(detectionsData);
      setStats(statsData);
    } catch (err) {
      error('Failed to load dashboard data');
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return <Loader fullScreen text="Loading dashboard..." />;
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="space-y-8"
    >
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-white">Dashboard</h1>
          <p className="text-gray-400 mt-1">
            Welcome back! Here's your deepfake detection overview
          </p>
        </div>
        <div className="flex items-center gap-3">
          <div className="glass px-4 py-2 rounded-xl">
            <span className="text-gray-400 mr-2">Last updated:</span>
            <span className="text-white">{new Date().toLocaleDateString()}</span>
          </div>
          <button
            onClick={fetchData}
            className="btn-icon"
            disabled={loading}
          >
            <svg className={`w-5 h-5 ${loading ? 'animate-spin' : ''}`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
            </svg>
          </button>
        </div>
      </div>

      {/* Stats Cards */}
      <StatsCards stats={stats} />

      {/* Charts */}
      <div className="charts-grid">
        <TrendChart />
        <PieChart />
      </div>

      {/* Recent Detections */}
      <RecentDetections 
        detections={detections} 
        onView={(det) => console.log('View:', det)}
        onDownload={(det) => console.log('Download:', det)}
        onDelete={(det) => console.log('Delete:', det)}
      />
    </motion.div>
  );
}