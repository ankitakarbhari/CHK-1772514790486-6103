import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { 
  FiClock,
  FiFilm,
  FiAlertTriangle,
  FiCheckCircle,
  FiZoomIn,
  FiDownload
} from 'react-icons/fi';
import Button from '@/components/common/Button';

/**
 * TimelineView Component - Display video detection timeline
 * 
 * @param {Object} props
 * @param {Array} props.timeline - Timeline data array
 * @param {number} props.totalFrames - Total frames analyzed
 * @param {Function} props.onFrameSelect - Frame selection handler
 */

const TimelineView = ({ timeline = [], totalFrames = 0, onFrameSelect }) => {
  const [selectedFrame, setSelectedFrame] = useState(null);
  const [zoomLevel, setZoomLevel] = useState(1);

  if (!timeline || timeline.length === 0) return null;

  const handleFrameClick = (frame) => {
    setSelectedFrame(frame);
    onFrameSelect?.(frame);
  };

  const getFrameColor = (frame) => {
    if (frame.is_fake) return 'bg-red-500';
    if (frame.confidence > 0.5) return 'bg-yellow-500';
    return 'bg-green-500';
  };

  const getConfidenceColor = (confidence) => {
    if (confidence > 0.8) return 'text-red-400';
    if (confidence > 0.6) return 'text-yellow-400';
    return 'text-green-400';
  };

  // Calculate stats
  const fakeFrames = timeline.filter(f => f.is_fake).length;
  const suspiciousFrames = timeline.filter(f => !f.is_fake && f.confidence > 0.5).length;
  const realFrames = timeline.filter(f => !f.is_fake && f.confidence <= 0.5).length;

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="glass-card p-6"
    >
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div>
          <h3 className="text-white font-semibold">Detection Timeline</h3>
          <p className="text-sm text-gray-400 mt-1">
            {timeline.length} frames analyzed • {totalFrames} total
          </p>
        </div>
        <div className="flex gap-2">
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setZoomLevel(Math.min(zoomLevel + 0.5, 3))}
            icon={<FiZoomIn className="w-4 h-4" />}
          />
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setZoomLevel(Math.max(zoomLevel - 0.5, 1))}
            icon={<FiZoomIn className="w-4 h-4 transform rotate-180" />}
          />
        </div>
      </div>

      {/* Stats Summary */}
      <div className="grid grid-cols-3 gap-4 mb-6">
        <div className="glass p-3 rounded-xl text-center">
          <div className="flex items-center justify-center gap-1 mb-1">
            <FiAlertTriangle className="w-4 h-4 text-red-400" />
            <span className="text-xs text-gray-400">Fake</span>
          </div>
          <p className="text-white font-semibold">{fakeFrames}</p>
        </div>
        <div className="glass p-3 rounded-xl text-center">
          <div className="flex items-center justify-center gap-1 mb-1">
            <FiAlertTriangle className="w-4 h-4 text-yellow-400" />
            <span className="text-xs text-gray-400">Suspicious</span>
          </div>
          <p className="text-white font-semibold">{suspiciousFrames}</p>
        </div>
        <div className="glass p-3 rounded-xl text-center">
          <div className="flex items-center justify-center gap-1 mb-1">
            <FiCheckCircle className="w-4 h-4 text-green-400" />
            <span className="text-xs text-gray-400">Real</span>
          </div>
          <p className="text-white font-semibold">{realFrames}</p>
        </div>
      </div>

      {/* Timeline Visualization */}
      <div className="space-y-4">
        {/* Timeline Bars */}
        <div 
          className="flex h-16 gap-0.5 overflow-x-auto pb-2"
          style={{ minWidth: `${timeline.length * (4 * zoomLevel)}px` }}
        >
          {timeline.map((frame, idx) => (
            <motion.div
              key={idx}
              initial={{ height: 0 }}
              animate={{ height: '100%' }}
              transition={{ delay: idx * 0.01 }}
              className={`flex-1 min-w-[4px] ${getFrameColor(frame)} cursor-pointer hover:opacity-80 transition-opacity`}
              style={{ height: `${frame.confidence * 100}%` }}
              onClick={() => handleFrameClick(frame)}
              title={`Frame ${frame.frame_index}: ${frame.is_fake ? 'FAKE' : 'REAL'} (${(frame.confidence * 100).toFixed(1)}%)`}
            />
          ))}
        </div>

        {/* Time Labels */}
        <div className="flex justify-between text-xs text-gray-400">
          <span>Start</span>
          <span>Time →</span>
          <span>End</span>
        </div>
      </div>

      {/* Selected Frame Details */}
      {selectedFrame && (
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          className="mt-6 p-4 glass rounded-xl"
        >
          <div className="flex items-center justify-between mb-3">
            <h4 className="text-white font-medium">Frame Details</h4>
            <span className="text-xs text-gray-400">
              Index: {selectedFrame.frame_index}
            </span>
          </div>
          
          <div className="grid grid-cols-2 gap-4">
            <div>
              <p className="text-xs text-gray-400 mb-1">Timestamp</p>
              <p className="text-white text-sm">
                {selectedFrame.timestamp?.toFixed(2)}s
              </p>
            </div>
            <div>
              <p className="text-xs text-gray-400 mb-1">Result</p>
              <p className={`text-sm font-medium ${
                selectedFrame.is_fake ? 'text-red-400' : 'text-green-400'
              }`}>
                {selectedFrame.is_fake ? 'FAKE' : 'REAL'}
              </p>
            </div>
            <div>
              <p className="text-xs text-gray-400 mb-1">Confidence</p>
              <p className={`text-sm ${getConfidenceColor(selectedFrame.confidence)}`}>
                {(selectedFrame.confidence * 100).toFixed(1)}%
              </p>
            </div>
            <div>
              <p className="text-xs text-gray-400 mb-1">Faces Detected</p>
              <p className="text-white text-sm">{selectedFrame.num_faces || 0}</p>
            </div>
          </div>

          {selectedFrame.face_boxes?.length > 0 && (
            <div className="mt-3">
              <p className="text-xs text-gray-400 mb-2">Face Detections</p>
              <div className="space-y-2">
                {selectedFrame.face_boxes.map((face, idx) => (
                  <div key={idx} className="flex items-center justify-between text-xs">
                    <span className="text-gray-300">Face {idx + 1}</span>
                    <span className="text-white">
                      {face.width}x{face.height}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </motion.div>
      )}

      {/* Export Options */}
      <div className="mt-6 pt-4 border-t border-white/10 flex justify-end gap-2">
        <Button
          variant="ghost"
          size="sm"
          onClick={() => {
            const data = JSON.stringify(timeline, null, 2);
            const blob = new Blob([data], { type: 'application/json' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `timeline-${Date.now()}.json`;
            a.click();
          }}
          icon={<FiDownload className="w-4 h-4" />}
        >
          Export JSON
        </Button>
      </div>
    </motion.div>
  );
};

export default TimelineView;