import React from 'react';
import { motion } from 'framer-motion';
import LiveDetection from '@/components/detection/LiveDetection';
import { FiInfo } from 'react-icons/fi';

export default function LivePage() {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="space-y-6"
    >
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-white">Live Deepfake Detection</h1>
          <p className="text-gray-400 mt-1">
            Real-time analysis from camera, screen share, or microphone
          </p>
        </div>
        <div className="relative group">
          <FiInfo className="w-5 h-5 text-gray-400 cursor-help" />
          <div className="absolute right-0 mt-2 w-64 glass p-3 rounded-xl 
                        opacity-0 invisible group-hover:opacity-100 group-hover:visible 
                        transition-all duration-300 z-10 text-sm text-gray-300">
            Real-time detection using WebSocket connection for instant alerts.
          </div>
        </div>
      </div>

      {/* Quick Stats */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="glass p-4 rounded-xl">
          <div className="text-2xl font-bold text-white mb-1">30 FPS</div>
          <div className="text-sm text-gray-400">Processing Speed</div>
        </div>
        <div className="glass p-4 rounded-xl">
          <div className="text-2xl font-bold text-white mb-1">&lt; 100ms</div>
          <div className="text-sm text-gray-400">Latency</div>
        </div>
        <div className="glass p-4 rounded-xl">
          <div className="text-2xl font-bold text-white mb-1">3</div>
          <div className="text-sm text-gray-400">Detection Modes</div>
        </div>
      </div>

      {/* Live Detection Component */}
      <LiveDetection />
    </motion.div>
  );
}