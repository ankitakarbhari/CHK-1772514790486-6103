import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { 
  FiMaximize2,
  FiMinimize2,
  FiDownload,
  FiLayers,
  FiTarget,
  FiGrid
} from 'react-icons/fi';
import Button from '@/components/common/Button';
import Modal from '@/components/common/Modal';

/**
 * HeatmapViewer Component - Display manipulation heatmap overlay
 * 
 * @param {Object} props
 * @param {string} props.originalImage - Original image URL/base64
 * @param {string} props.heatmapImage - Heatmap overlay image URL/base64
 * @param {Array} props.regions - Manipulated regions data
 */

const HeatmapViewer = ({ originalImage, heatmapImage, regions = [] }) => {
  const [viewMode, setViewMode] = useState('overlay'); // 'overlay', 'split', 'original'
  const [opacity, setOpacity] = useState(0.6);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [selectedRegion, setSelectedRegion] = useState(null);

  if (!originalImage || !heatmapImage) return null;

  const handleDownload = () => {
    const link = document.createElement('a');
    link.href = heatmapImage;
    link.download = `heatmap-${Date.now()}.png`;
    link.click();
  };

  const renderView = () => {
    switch(viewMode) {
      case 'overlay':
        return (
          <div className="relative">
            <img src={originalImage} alt="Original" className="w-full rounded-lg" />
            <div className="absolute inset-0">
              <img 
                src={heatmapImage} 
                alt="Heatmap" 
                className="w-full rounded-lg"
                style={{ opacity }}
              />
            </div>
          </div>
        );

      case 'split':
        return (
          <div className="grid grid-cols-2 gap-4">
            <div>
              <p className="text-sm text-gray-400 mb-2">Original</p>
              <img src={originalImage} alt="Original" className="w-full rounded-lg" />
            </div>
            <div>
              <p className="text-sm text-gray-400 mb-2">Heatmap</p>
              <img src={heatmapImage} alt="Heatmap" className="w-full rounded-lg" />
            </div>
          </div>
        );

      case 'original':
        return (
          <img src={originalImage} alt="Original" className="w-full rounded-lg" />
        );

      default:
        return null;
    }
  };

  return (
    <>
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="glass-card p-6"
      >
        {/* Header */}
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-white font-semibold">Heatmap Analysis</h3>
          <div className="flex gap-2">
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setIsFullscreen(true)}
              icon={<FiMaximize2 className="w-4 h-4" />}
            />
            <Button
              variant="ghost"
              size="sm"
              onClick={handleDownload}
              icon={<FiDownload className="w-4 h-4" />}
            />
          </div>
        </div>

        {/* View Mode Toggle */}
        <div className="flex gap-2 mb-4">
          <button
            onClick={() => setViewMode('overlay')}
            className={`flex items-center gap-2 px-3 py-2 rounded-lg transition-all ${
              viewMode === 'overlay' ? 'bg-primary-500 text-white' : 'glass text-gray-400 hover:text-white'
            }`}
          >
            <FiLayers className="w-4 h-4" />
            <span className="text-sm">Overlay</span>
          </button>
          <button
            onClick={() => setViewMode('split')}
            className={`flex items-center gap-2 px-3 py-2 rounded-lg transition-all ${
              viewMode === 'split' ? 'bg-primary-500 text-white' : 'glass text-gray-400 hover:text-white'
            }`}
          >
            <FiGrid className="w-4 h-4" />
            <span className="text-sm">Split</span>
          </button>
          <button
            onClick={() => setViewMode('original')}
            className={`flex items-center gap-2 px-3 py-2 rounded-lg transition-all ${
              viewMode === 'original' ? 'bg-primary-500 text-white' : 'glass text-gray-400 hover:text-white'
            }`}
          >
            <FiTarget className="w-4 h-4" />
            <span className="text-sm">Original</span>
          </button>
        </div>

        {/* Opacity Slider (for overlay mode) */}
        {viewMode === 'overlay' && (
          <div className="mb-4">
            <label className="block text-gray-400 text-sm mb-2">
              Heatmap Opacity: {Math.round(opacity * 100)}%
            </label>
            <input
              type="range"
              min="0"
              max="1"
              step="0.1"
              value={opacity}
              onChange={(e) => setOpacity(parseFloat(e.target.value))}
              className="w-full"
            />
          </div>
        )}

        {/* Image Display */}
        <div className="rounded-lg overflow-hidden border border-white/10 bg-black">
          {renderView()}
        </div>

        {/* Manipulated Regions */}
        {regions.length > 0 && (
          <div className="mt-4">
            <p className="text-white text-sm font-medium mb-3">Manipulated Regions</p>
            <div className="grid grid-cols-2 gap-3">
              {regions.map((region, idx) => (
                <button
                  key={idx}
                  onClick={() => setSelectedRegion(region)}
                  className={`p-3 glass rounded-xl text-left transition-all ${
                    selectedRegion === region ? 'border-primary-500 border' : ''
                  }`}
                >
                  <p className="text-xs text-gray-400">Region {idx + 1}</p>
                  <p className="text-white text-sm font-medium">
                    Confidence: {(region.confidence * 100).toFixed(1)}%
                  </p>
                  {region.bounding_box && (
                    <p className="text-xs text-gray-400 mt-1">
                      {region.bounding_box.width}x{region.bounding_box.height}
                    </p>
                  )}
                </button>
              ))}
            </div>
          </div>
        )}
      </motion.div>

      {/* Fullscreen Modal */}
      <Modal
        isOpen={isFullscreen}
        onClose={() => setIsFullscreen(false)}
        title="Heatmap Viewer"
        size="xl"
      >
        <div className="space-y-4">
          <div className="flex justify-end gap-2">
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setIsFullscreen(false)}
              icon={<FiMinimize2 className="w-4 h-4" />}
            />
          </div>
          <div className="grid grid-cols-2 gap-4">
            <div>
              <p className="text-gray-400 text-sm mb-2">Original</p>
              <img src={originalImage} alt="Original" className="w-full rounded-lg" />
            </div>
            <div>
              <p className="text-gray-400 text-sm mb-2">Heatmap</p>
              <img src={heatmapImage} alt="Heatmap" className="w-full rounded-lg" />
            </div>
          </div>
        </div>
      </Modal>
    </>
  );
};

export default HeatmapViewer;