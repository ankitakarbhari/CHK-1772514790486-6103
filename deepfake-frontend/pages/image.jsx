import React from 'react';
import { motion } from 'framer-motion';
import ImageUploader from '@/components/detection/ImageUploader';
import { FiInfo } from 'react-icons/fi';

export default function ImagePage() {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="space-y-6"
    >
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-white">Image Deepfake Detection</h1>
          <p className="text-gray-400 mt-1">
            Upload an image to check if it's AI-generated or manipulated
          </p>
        </div>
        <div className="relative group">
          <FiInfo className="w-5 h-5 text-gray-400 cursor-help" />
          <div className="absolute right-0 mt-2 w-64 glass p-3 rounded-xl 
                        opacity-0 invisible group-hover:opacity-100 group-hover:visible 
                        transition-all duration-300 z-10 text-sm text-gray-300">
            Our AI analyzes images for signs of manipulation, GAN artifacts, 
            and inconsistencies that indicate deepfakes.
          </div>
        </div>
      </div>

      {/* Quick Stats */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="glass p-4 rounded-xl">
          <div className="text-2xl font-bold text-white mb-1">98.3%</div>
          <div className="text-sm text-gray-400">Detection Accuracy</div>
        </div>
        <div className="glass p-4 rounded-xl">
          <div className="text-2xl font-bold text-white mb-1">&lt; 1s</div>
          <div className="text-sm text-gray-400">Processing Time</div>
        </div>
        <div className="glass p-4 rounded-xl">
          <div className="text-2xl font-bold text-white mb-1">10+</div>
          <div className="text-sm text-gray-400">Supported Formats</div>
        </div>
      </div>

      {/* Uploader Component */}
      <ImageUploader />

      {/* Features */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mt-8">
        {[
          {
            title: 'Face Detection',
            description: 'Identifies and analyzes faces in images',
            icon: '👤',
          },
          {
            title: 'Heatmap Visualization',
            description: 'Shows manipulated regions with color coding',
            icon: '🔥',
          },
          {
            title: 'GAN Detection',
            description: 'Detects AI-generated synthetic images',
            icon: '🤖',
          },
          {
            title: 'Metadata Analysis',
            description: 'Checks image metadata for inconsistencies',
            icon: '📊',
          },
        ].map((feature, index) => (
          <motion.div
            key={index}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 * index }}
            className="glass p-4 rounded-xl hover:border-primary-500/50 transition-all duration-300"
          >
            <div className="text-3xl mb-2">{feature.icon}</div>
            <h3 className="text-white font-semibold mb-1">{feature.title}</h3>
            <p className="text-gray-400 text-sm">{feature.description}</p>
          </motion.div>
        ))}
      </div>
    </motion.div>
  );
}