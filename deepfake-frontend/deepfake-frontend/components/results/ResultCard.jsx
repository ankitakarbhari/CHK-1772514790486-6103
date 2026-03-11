import React from 'react';
import { motion } from 'framer-motion';
import { 
  FiCheckCircle,
  FiAlertTriangle,
  FiXCircle,
  FiClock,
  FiBarChart2,
  FiDownload,
  FiShare2,
  FiStar,
  FiCopy
} from 'react-icons/fi';
import Button from '@/components/common/Button';
import { useAlert } from '@/components/common/Alert';

/**
 * ResultCard Component - Display detection results in a card
 * 
 * @param {Object} props
 * @param {Object} props.result - Detection result object
 * @param {string} props.type - 'image', 'video', 'audio', 'text', 'url'
 * @param {Function} props.onDownload - Download report handler
 * @param {Function} props.onShare - Share result handler
 */

const ResultCard = ({ result, type = 'image', onDownload, onShare }) => {
  const { success } = useAlert();

  if (!result) return null;

  const isFake = result.prediction === 'FAKE' || result.prediction === 'AI';
  const confidence = result.confidence ? (result.confidence * 100).toFixed(1) : '0';

  // Get result icon and color
  const getResultStyle = () => {
    if (isFake) {
      return {
        icon: FiAlertTriangle,
        color: 'text-red-400',
        bg: 'bg-red-500/20',
        border: 'border-red-500/20',
        label: result.prediction || 'FAKE'
      };
    } else if (result.prediction === 'SUSPICIOUS') {
      return {
        icon: FiAlertTriangle,
        color: 'text-yellow-400',
        bg: 'bg-yellow-500/20',
        border: 'border-yellow-500/20',
        label: 'SUSPICIOUS'
      };
    } else {
      return {
        icon: FiCheckCircle,
        color: 'text-green-400',
        bg: 'bg-green-500/20',
        border: 'border-green-500/20',
        label: result.prediction || 'REAL'
      };
    }
  };

  const style = getResultStyle();
  const Icon = style.icon;

  const handleCopyId = () => {
    navigator.clipboard.writeText(result.id || 'N/A');
    success('Copied!', 'Result ID copied to clipboard');
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="glass-card overflow-hidden"
    >
      {/* Header */}
      <div className={`p-6 border-b ${style.border}`}>
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-4">
            <div className={`w-14 h-14 rounded-xl ${style.bg} flex items-center justify-center`}>
              <Icon className={`w-7 h-7 ${style.color}`} />
            </div>
            <div>
              <h3 className="text-white text-lg font-semibold">Detection Result</h3>
              <p className="text-sm text-gray-400">
                {type.charAt(0).toUpperCase() + type.slice(1)} Analysis
              </p>
            </div>
          </div>
          <div className="flex gap-2">
            <Button
              variant="ghost"
              size="sm"
              onClick={() => onDownload?.(result)}
              icon={<FiDownload className="w-4 h-4" />}
            />
            <Button
              variant="ghost"
              size="sm"
              onClick={() => onShare?.(result)}
              icon={<FiShare2 className="w-4 h-4" />}
            />
          </div>
        </div>
      </div>

      {/* Content */}
      <div className="p-6 space-y-6">
        {/* Main Result */}
        <div className="text-center">
          <div className={`inline-block px-6 py-3 rounded-full ${style.bg} border ${style.border} mb-3`}>
            <span className={`text-xl font-bold ${style.color}`}>{style.label}</span>
          </div>
          <div className="flex items-center justify-center gap-2">
            <FiBarChart2 className="w-4 h-4 text-gray-400" />
            <span className="text-gray-400">Confidence:</span>
            <span className="text-white font-semibold">{confidence}%</span>
          </div>
        </div>

        {/* Quick Stats */}
        <div className="grid grid-cols-2 gap-4">
          <div className="glass p-4 rounded-xl">
            <p className="text-xs text-gray-400 mb-1">Filename</p>
            <p className="text-white text-sm font-medium truncate">
              {result.filename || 'N/A'}
            </p>
          </div>
          <div className="glass p-4 rounded-xl">
            <p className="text-xs text-gray-400 mb-1">Result ID</p>
            <div className="flex items-center gap-2">
              <p className="text-white text-sm font-mono truncate">
                {result.id || 'N/A'}
              </p>
              <button
                onClick={handleCopyId}
                className="p-1 hover:bg-white/10 rounded-lg transition-colors"
              >
                <FiCopy className="w-3 h-3 text-gray-400" />
              </button>
            </div>
          </div>
          <div className="glass p-4 rounded-xl">
            <p className="text-xs text-gray-400 mb-1">Timestamp</p>
            <p className="text-white text-sm">
              {result.timestamp ? new Date(result.timestamp).toLocaleString() : 'N/A'}
            </p>
          </div>
          <div className="glass p-4 rounded-xl">
            <p className="text-xs text-gray-400 mb-1">Processing Time</p>
            <p className="text-white text-sm">
              {result.processing_time ? `${(result.processing_time * 1000).toFixed(0)}ms` : 'N/A'}
            </p>
          </div>
        </div>

        {/* Type-specific details */}
        {type === 'image' && result.faces_detected !== undefined && (
          <div className="glass p-4 rounded-xl">
            <p className="text-sm text-gray-400 mb-2">Face Detection</p>
            <div className="flex items-center gap-4">
              <div className="flex-1">
                <p className="text-xs text-gray-400">Faces Found</p>
                <p className="text-white text-lg font-semibold">{result.faces_detected}</p>
              </div>
              {result.face_details && (
                <div className="flex-1">
                  <p className="text-xs text-gray-400">Confidence (Avg)</p>
                  <p className="text-white text-lg font-semibold">
                    {(result.face_details.reduce((acc, f) => acc + f.confidence, 0) / 
                      result.face_details.length * 100).toFixed(1)}%
                  </p>
                </div>
              )}
            </div>
          </div>
        )}

        {type === 'video' && result.frames_analyzed && (
          <div className="glass p-4 rounded-xl">
            <p className="text-sm text-gray-400 mb-3">Video Analysis</p>
            <div className="grid grid-cols-2 gap-4">
              <div>
                <p className="text-xs text-gray-400">Frames Analyzed</p>
                <p className="text-white font-semibold">{result.frames_analyzed}</p>
              </div>
              <div>
                <p className="text-xs text-gray-400">Fake Frames</p>
                <p className="text-red-400 font-semibold">{result.fake_frames}</p>
              </div>
              <div>
                <p className="text-xs text-gray-400">Real Frames</p>
                <p className="text-green-400 font-semibold">{result.real_frames}</p>
              </div>
              <div>
                <p className="text-xs text-gray-400">Fake %</p>
                <p className="text-white font-semibold">{result.fake_percentage?.toFixed(1)}%</p>
              </div>
            </div>
          </div>
        )}

        {type === 'text' && result.metrics && (
          <div className="glass p-4 rounded-xl">
            <p className="text-sm text-gray-400 mb-3">Text Metrics</p>
            <div className="grid grid-cols-2 gap-4">
              <div>
                <p className="text-xs text-gray-400">Perplexity</p>
                <p className="text-white font-semibold">{result.metrics.perplexity?.toFixed(2)}</p>
              </div>
              <div>
                <p className="text-xs text-gray-400">Burstiness</p>
                <p className="text-white font-semibold">{result.metrics.burstiness?.toFixed(2)}</p>
              </div>
              <div>
                <p className="text-xs text-gray-400">Word Count</p>
                <p className="text-white font-semibold">{result.metrics.word_count || 0}</p>
              </div>
              <div>
                <p className="text-xs text-gray-400">Readability</p>
                <p className="text-white font-semibold">{result.metrics.readability_score?.toFixed(2)}</p>
              </div>
            </div>
          </div>
        )}

        {type === 'url' && result.threat_assessment && (
          <div className="glass p-4 rounded-xl">
            <p className="text-sm text-gray-400 mb-3">Threat Assessment</p>
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-xs text-gray-400">Risk Score</span>
                <span className={`text-sm font-semibold ${
                  result.threat_assessment.risk_level === 'CRITICAL' ? 'text-red-400' :
                  result.threat_assessment.risk_level === 'HIGH' ? 'text-orange-400' :
                  result.threat_assessment.risk_level === 'MEDIUM' ? 'text-yellow-400' :
                  'text-green-400'
                }`}>
                  {result.threat_assessment.risk_score}/100
                </span>
              </div>
              {result.threat_assessment.threat_types?.length > 0 && (
                <div className="flex flex-wrap gap-2">
                  {result.threat_assessment.threat_types.map((threat, idx) => (
                    <span key={idx} className="px-2 py-1 bg-red-500/20 text-red-400 rounded-full text-xs">
                      {threat}
                    </span>
                  ))}
                </div>
              )}
            </div>
          </div>
        )}

        {/* Warnings */}
        {result.warnings?.length > 0 && (
          <div className="glass p-4 rounded-xl border border-yellow-500/20">
            <p className="text-yellow-400 text-sm font-medium mb-2">⚠️ Warnings</p>
            <ul className="space-y-2">
              {result.warnings.map((warning, idx) => (
                <li key={idx} className="text-xs text-gray-300 flex items-start gap-2">
                  <span>•</span>
                  <span>{warning}</span>
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>
    </motion.div>
  );
};

export default ResultCard;