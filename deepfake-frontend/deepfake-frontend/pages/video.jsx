import React from 'react';
import { motion } from 'framer-motion';
import VideoUploader from '@/components/detection/VideoUploader';
import { FiInfo } from 'react-icons/fi';

export default function VideoPage() {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="space-y-6"
    >
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-white">Video Deepfake Detection</h1>
          <p className="text-gray-400 mt-1">
            Upload a video to detect face swaps, lip-sync deepfakes, and manipulations
          </p>
        </div>
        <div className="relative group">
          <FiInfo className="w-5 h-5 text-gray-400 cursor-help" />
          <div className="absolute right-0 mt-2 w-64 glass p-3 rounded-xl 
                        opacity-0 invisible group-hover:opacity-100 group-hover:visible 
                        transition-all duration-300 z-10 text-sm text-gray-300">
            Our AI analyzes video frame by frame to detect inconsistencies and manipulations.
          </div>
        </div>
      </div>

      {/* Quick Stats */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="glass p-4 rounded-xl">
          <div className="text-2xl font-bold text-white mb-1">96.8%</div>
          <div className="text-sm text-gray-400">Detection Accuracy</div>
        </div>
        <div className="glass p-4 rounded-xl">
          <div className="text-2xl font-bold text-white mb-1">100 MB</div>
          <div className="text-sm text-gray-400">Max File Size</div>
        </div>
        <div className="glass p-4 rounded-xl">
          <div className="text-2xl font-bold text-white mb-1">5+</div>
          <div className="text-sm text-gray-400">Video Formats</div>
        </div>
      </div>

      {/* Uploader Component */}
      <VideoUploader />
    </motion.div>
  );
}