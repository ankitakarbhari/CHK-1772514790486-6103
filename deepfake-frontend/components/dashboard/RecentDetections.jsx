import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { 
  FiEye, 
  FiDownload, 
  FiTrash2, 
  FiMoreVertical,
  FiFilter,
  FiSearch,
  FiImage,
  FiVideo,
  FiMusic,
  FiFileText,
  FiLink,
  FiCheckCircle,
  FiXCircle,
  FiAlertCircle
} from 'react-icons/fi';
import Button from '@/components/common/Button';
import { formatDistanceToNow } from 'date-fns';

/**
 * RecentDetections Component - Displays recent detection history
 * 
 * @param {Object} props
 * @param {Array} props.detections - Array of detection objects
 * @param {boolean} props.loading - Loading state
 * @param {Function} props.onView - View detection handler
 * @param {Function} props.onDownload - Download report handler
 * @param {Function} props.onDelete - Delete detection handler
 */

const RecentDetections = ({ 
  detections = [], 
  loading = false,
  onView,
  onDownload,
  onDelete 
}) => {
  const [filter, setFilter] = useState('all');
  const [search, setSearch] = useState('');
  const [selectedRows, setSelectedRows] = useState([]);

  // Sample detections
  const sampleDetections = [
    {
      id: 'DET-001',
      type: 'image',
      filename: 'profile_photo.jpg',
      result: 'REAL',
      confidence: 98.5,
      date: new Date(Date.now() - 2 * 60 * 60 * 1000).toISOString(),
      thumbnail: 'https://via.placeholder.com/40',
    },
    {
      id: 'DET-002',
      type: 'video',
      filename: 'interview.mp4',
      result: 'FAKE',
      confidence: 94.2,
      date: new Date(Date.now() - 5 * 60 * 60 * 1000).toISOString(),
    },
    {
      id: 'DET-003',
      type: 'audio',
      filename: 'voice_message.wav',
      result: 'FAKE',
      confidence: 87.3,
      date: new Date(Date.now() - 1 * 24 * 60 * 60 * 1000).toISOString(),
    },
    {
      id: 'DET-004',
      type: 'text',
      filename: 'article.txt',
      result: 'REAL',
      confidence: 96.8,
      date: new Date(Date.now() - 2 * 24 * 60 * 60 * 1000).toISOString(),
    },
    {
      id: 'DET-005',
      type: 'url',
      filename: 'example.com',
      result: 'SUSPICIOUS',
      confidence: 76.5,
      date: new Date(Date.now() - 3 * 24 * 60 * 60 * 1000).toISOString(),
    },
  ];

  const displayDetections = detections.length > 0 ? detections : sampleDetections;

  // Filter and search
  const filteredDetections = displayDetections.filter(det => {
    if (filter !== 'all' && det.result.toLowerCase() !== filter) return false;
    if (search && !det.filename.toLowerCase().includes(search.toLowerCase())) return false;
    return true;
  });

  // Icons for types
  const typeIcons = {
    image: FiImage,
    video: FiVideo,
    audio: FiMusic,
    text: FiFileText,
    url: FiLink,
  };

  // Result colors and icons
  const resultConfig = {
    REAL: {
      icon: FiCheckCircle,
      color: 'text-green-400',
      bg: 'bg-green-500/10',
      border: 'border-green-500/20',
    },
    FAKE: {
      icon: FiXCircle,
      color: 'text-red-400',
      bg: 'bg-red-500/10',
      border: 'border-red-500/20',
    },
    SUSPICIOUS: {
      icon: FiAlertCircle,
      color: 'text-yellow-400',
      bg: 'bg-yellow-500/10',
      border: 'border-yellow-500/20',
    },
  };

  // Handle row selection
  const toggleRow = (id) => {
    setSelectedRows(prev =>
      prev.includes(id) ? prev.filter(rowId => rowId !== id) : [...prev, id]
    );
  };

  const toggleAll = () => {
    if (selectedRows.length === filteredDetections.length) {
      setSelectedRows([]);
    } else {
      setSelectedRows(filteredDetections.map(d => d.id));
    }
  };

  // Loading skeleton
  if (loading) {
    return (
      <div className="table-container">
        <div className="p-6 border-b border-white/10">
          <div className="h-8 w-48 bg-white/10 rounded" />
        </div>
        <div className="p-6 space-y-4">
          {[1, 2, 3, 4, 5].map(i => (
            <div key={i} className="flex items-center space-x-4 animate-pulse">
              <div className="w-10 h-10 bg-white/10 rounded" />
              <div className="flex-1 h-10 bg-white/10 rounded" />
            </div>
          ))}
        </div>
      </div>
    );
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="table-container"
    >
      {/* Header */}
      <div className="p-6 border-b border-white/10">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <div>
            <h3 className="text-white font-semibold">Recent Detections</h3>
            <p className="text-sm text-gray-400 mt-1">
              Showing {filteredDetections.length} of {displayDetections.length} detections
            </p>
          </div>

          {/* Search and Filter */}
          <div className="flex items-center gap-3">
            {/* Search */}
            <div className="relative">
              <FiSearch className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-4 h-4" />
              <input
                type="text"
                placeholder="Search files..."
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                className="input-field pl-10 py-2 text-sm"
              />
            </div>

            {/* Filter */}
            <div className="relative">
              <select
                value={filter}
                onChange={(e) => setFilter(e.target.value)}
                className="input-field py-2 pr-8 text-sm appearance-none"
              >
                <option value="all">All Results</option>
                <option value="real">Real Only</option>
                <option value="fake">Fake Only</option>
                <option value="suspicious">Suspicious Only</option>
              </select>
              <FiFilter className="absolute right-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-4 h-4 pointer-events-none" />
            </div>
          </div>
        </div>
      </div>

      {/* Table */}
      <div className="overflow-x-auto">
        <table className="w-full">
          <thead>
            <tr className="border-b border-white/10">
              <th className="table-header w-10">
                <input
                  type="checkbox"
                  checked={selectedRows.length === filteredDetections.length && filteredDetections.length > 0}
                  onChange={toggleAll}
                  className="w-4 h-4 rounded border-white/10 bg-white/5 text-primary-500"
                />
              </th>
              <th className="table-header">ID</th>
              <th className="table-header">Type</th>
              <th className="table-header">Filename</th>
              <th className="table-header">Result</th>
              <th className="table-header">Confidence</th>
              <th className="table-header">Date</th>
              <th className="table-header text-right">Actions</th>
            </tr>
          </thead>
          <tbody>
            {filteredDetections.map((det, index) => {
              const TypeIcon = typeIcons[det.type] || FiImage;
              const result = resultConfig[det.result] || resultConfig.REAL;
              const isSelected = selectedRows.includes(det.id);

              return (
                <motion.tr
                  key={det.id}
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  transition={{ delay: index * 0.05 }}
                  className={`
                    border-b border-white/5 hover:bg-white/5 transition-colors
                    ${isSelected ? 'bg-primary-500/5' : ''}
                  `}
                >
                  <td className="table-cell">
                    <input
                      type="checkbox"
                      checked={isSelected}
                      onChange={() => toggleRow(det.id)}
                      className="w-4 h-4 rounded border-white/10 bg-white/5 text-primary-500"
                    />
                  </td>
                  <td className="table-cell font-mono text-xs">{det.id}</td>
                  <td className="table-cell">
                    <div className="flex items-center gap-2">
                      <TypeIcon className="w-4 h-4 text-gray-400" />
                      <span className="text-sm capitalize">{det.type}</span>
                    </div>
                  </td>
                  <td className="table-cell">
                    <div className="flex items-center gap-3">
                      {det.thumbnail && (
                        <img 
                          src={det.thumbnail} 
                          alt="" 
                          className="w-8 h-8 rounded-lg object-cover"
                        />
                      )}
                      <span className="text-sm text-gray-300">{det.filename}</span>
                    </div>
                  </td>
                  <td className="table-cell">
                    <span className={`
                      inline-flex items-center gap-1 px-2 py-1 rounded-full text-xs
                      ${result.bg} ${result.color} border ${result.border}
                    `}>
                      <result.icon className="w-3 h-3" />
                      {det.result}
                    </span>
                  </td>
                  <td className="table-cell">
                    <div className="flex items-center gap-2">
                      <span className="text-sm text-white">{det.confidence}%</span>
                      <div className="progress-bar w-16">
                        <div 
                          className={`progress-fill ${
                            det.confidence > 80 ? 'bg-green-500' :
                            det.confidence > 60 ? 'bg-yellow-500' : 'bg-red-500'
                          }`}
                          style={{ width: `${det.confidence}%` }}
                        />
                      </div>
                    </div>
                  </td>
                  <td className="table-cell text-sm text-gray-400">
                    {formatDistanceToNow(new Date(det.date), { addSuffix: true })}
                  </td>
                  <td className="table-cell text-right">
                    <div className="flex items-center justify-end gap-2">
                      <button
                        onClick={() => onView?.(det)}
                        className="p-1.5 rounded-lg hover:bg-white/10 text-gray-400 hover:text-white transition-colors"
                        title="View details"
                      >
                        <FiEye className="w-4 h-4" />
                      </button>
                      <button
                        onClick={() => onDownload?.(det)}
                        className="p-1.5 rounded-lg hover:bg-white/10 text-gray-400 hover:text-white transition-colors"
                        title="Download report"
                      >
                        <FiDownload className="w-4 h-4" />
                      </button>
                      <button
                        onClick={() => onDelete?.(det)}
                        className="p-1.5 rounded-lg hover:bg-white/10 text-gray-400 hover:text-red-400 transition-colors"
                        title="Delete"
                      >
                        <FiTrash2 className="w-4 h-4" />
                      </button>
                      <button className="p-1.5 rounded-lg hover:bg-white/10 text-gray-400 hover:text-white transition-colors">
                        <FiMoreVertical className="w-4 h-4" />
                      </button>
                    </div>
                  </td>
                </motion.tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {/* Empty state */}
      {filteredDetections.length === 0 && (
        <div className="p-12 text-center">
          <div className="w-16 h-16 mx-auto bg-white/5 rounded-full flex items-center justify-center mb-4">
            <FiSearch className="w-6 h-6 text-gray-400" />
          </div>
          <h4 className="text-white font-medium mb-2">No detections found</h4>
          <p className="text-sm text-gray-400">
            {search || filter !== 'all' 
              ? 'Try adjusting your filters' 
              : 'Upload media to start detecting deepfakes'}
          </p>
        </div>
      )}

      {/* Footer */}
      {filteredDetections.length > 0 && (
        <div className="p-4 border-t border-white/10 flex items-center justify-between">
          <div className="text-sm text-gray-400">
            {selectedRows.length} of {filteredDetections.length} selected
          </div>
          
          {selectedRows.length > 0 && (
            <div className="flex items-center gap-3">
              <Button size="sm" variant="ghost" onClick={() => setSelectedRows([])}>
                Clear
              </Button>
              <Button size="sm" variant="danger" onClick={() => {
                onDelete?.(selectedRows);
                setSelectedRows([]);
              }}>
                Delete Selected
              </Button>
            </div>
          )}
        </div>
      )}
    </motion.div>
  );
};

// ============================================
// Detection Card (Mobile/Grid View)
// ============================================

export const DetectionCard = ({ detection, onView, onDownload }) => {
  const TypeIcon = typeIcons[detection.type] || FiImage;
  const result = resultConfig[detection.result] || resultConfig.REAL;

  return (
    <motion.div
      whileHover={{ y: -5 }}
      className="glass-card p-4"
    >
      <div className="flex items-start justify-between mb-3">
        <div className="flex items-center gap-3">
          <div className={`w-10 h-10 rounded-xl ${result.bg} flex items-center justify-center`}>
            <TypeIcon className={`w-5 h-5 ${result.color}`} />
          </div>
          <div>
            <h4 className="text-white font-medium text-sm">{detection.filename}</h4>
            <p className="text-xs text-gray-400">{detection.id}</p>
          </div>
        </div>
        <span className={`px-2 py-1 rounded-full text-xs ${result.bg} ${result.color}`}>
          {detection.result}
        </span>
      </div>

      <div className="space-y-2 mb-4">
        <div className="flex justify-between text-sm">
          <span className="text-gray-400">Confidence</span>
          <span className="text-white">{detection.confidence}%</span>
        </div>
        <div className="progress-bar">
          <div 
            className={`progress-fill ${
              detection.confidence > 80 ? 'bg-green-500' :
              detection.confidence > 60 ? 'bg-yellow-500' : 'bg-red-500'
            }`}
            style={{ width: `${detection.confidence}%` }}
          />
        </div>
      </div>

      <div className="flex items-center justify-between text-xs text-gray-400">
        <span>{formatDistanceToNow(new Date(detection.date), { addSuffix: true })}</span>
        <div className="flex items-center gap-2">
          <button
            onClick={() => onView?.(detection)}
            className="p-1.5 rounded-lg hover:bg-white/10 text-gray-400 hover:text-white"
          >
            <FiEye className="w-4 h-4" />
          </button>
          <button
            onClick={() => onDownload?.(detection)}
            className="p-1.5 rounded-lg hover:bg-white/10 text-gray-400 hover:text-white"
          >
            <FiDownload className="w-4 h-4" />
          </button>
        </div>
      </div>
    </motion.div>
  );
};

export default RecentDetections;