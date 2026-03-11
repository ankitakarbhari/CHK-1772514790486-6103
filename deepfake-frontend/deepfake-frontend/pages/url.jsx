import React from 'react';
import { motion } from 'framer-motion';
import UrlAnalyzer from '@/components/detection/UrlAnalyzer';
import { FiInfo } from 'react-icons/fi';

export default function UrlPage() {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="space-y-6"
    >
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-white">URL Security Analyzer</h1>
          <p className="text-gray-400 mt-1">
            Check URLs for phishing, malware, scams, and deepfake content
          </p>
        </div>
        <div className="relative group">
          <FiInfo className="w-5 h-5 text-gray-400 cursor-help" />
          <div className="absolute right-0 mt-2 w-64 glass p-3 rounded-xl 
                        opacity-0 invisible group-hover:opacity-100 group-hover:visible 
                        transition-all duration-300 z-10 text-sm text-gray-300">
            Comprehensive URL analysis including phishing detection, SSL validation, and content scanning.
          </div>
        </div>
      </div>

      {/* Quick Stats */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="glass p-4 rounded-xl">
          <div className="text-2xl font-bold text-white mb-1">97.8%</div>
          <div className="text-sm text-gray-400">Detection Accuracy</div>
        </div>
        <div className="glass p-4 rounded-xl">
          <div className="text-2xl font-bold text-white mb-1">&lt; 2s</div>
          <div className="text-sm text-gray-400">Analysis Time</div>
        </div>
        <div className="glass p-4 rounded-xl">
          <div className="text-2xl font-bold text-white mb-1">50+</div>
          <div className="text-sm text-gray-400">Threat Indicators</div>
        </div>
      </div>

      {/* Analyzer Component */}
      <UrlAnalyzer />
    </motion.div>
  );
}