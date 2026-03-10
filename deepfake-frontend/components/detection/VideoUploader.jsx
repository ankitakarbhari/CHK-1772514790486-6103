import React, { useState, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useDropzone } from 'react-dropzone';
import ReactPlayer from 'react-player';
import { 
  FiUpload, 
  FiX, 
  FiVideo,
  FiSearch,
  FiDownload,
  FiRefreshCw,
  FiShield,
  FiAlertTriangle,
  FiCheckCircle,
  FiClock,
  FiBarChart2
} from 'react-icons/fi';
import Button from '@/components/common/Button';
import Loader from '@/components/common/Loader';
import { useAlert } from '@/components/common/Alert';
import { detectVideo } from '@/utils/api';
import { formatFileSize } from '@/utils/helpers';

/**
 * VideoUploader Component - Upload and analyze videos for deepfakes
 */

const VideoUploader = () => {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [activeTab, setActiveTab] = useState('upload');
  const [progress, setProgress] = useState(0);
  const [options, setOptions] = useState({
    sampleRate: 1,
    maxFrames: 300,
    analyzeAudio: true,
  });

  const { success, error, info } = useAlert();

  // Handle file drop
  const onDrop = useCallback((acceptedFiles) => {
    const file = acceptedFiles[0];
    if (!file) return;

    // Validate file size (100MB max)
    if (file.size > 100 * 1024 * 1024) {
      error('File too large', 'Maximum file size is 100MB');
      return;
    }

    setFile(file);
    setResult(null);
    setActiveTab('preview');

    // Create preview URL
    const url = URL.createObjectURL(file);
    setPreview(url);

    info('Video ready', 'Click "Analyze Video" to start detection');
  }, [error, info]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'video/*': ['.mp4', '.avi', '.mov', '.mkv', '.webm']
    },
    maxFiles: 1,
  });

  // Handle analysis
  const handleAnalyze = async () => {
    if (!file) {
      error('No file selected', 'Please upload a video first');
      return;
    }

    setLoading(true);
    setActiveTab('results');
    setProgress(0);

    // Simulate progress
    const interval = setInterval(() => {
      setProgress(prev => Math.min(prev + 5, 90));
    }, 1000);

    try {
      const response = await detectVideo(file, options);
      clearInterval(interval);
      setProgress(100);
      setResult(response.result);
      
      success('Analysis complete', `Video analysis finished`);
    } catch (err) {
      clearInterval(interval);
      error('Analysis failed', err.response?.data?.detail || 'Please try again');
    } finally {
      setLoading(false);
    }
  };

  // Reset upload
  const handleReset = () => {
    if (preview) {
      URL.revokeObjectURL(preview);
    }
    setFile(null);
    setPreview(null);
    setResult(null);
    setActiveTab('upload');
    setProgress(0);
  };

  // Download report
  const handleDownload = () => {
    const report = {
      filename: file?.name,
      timestamp: new Date().toISOString(),
      result: result,
      options: options,
    };

    const blob = new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `video-report-${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-white">Video Deepfake Detection</h2>
          <p className="text-gray-400 mt-1">
            Upload a video to detect face swaps, lip-sync deepfakes, and manipulations
          </p>
        </div>
        <div className="flex items-center gap-2 text-sm text-gray-400">
          <FiShield className="w-4 h-4" />
          <span>96.8% Accuracy</span>
        </div>
      </div>

      {/* Main content */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left column - Upload/Options */}
        <div className="lg:col-span-1 space-y-6">
          {/* Upload area */}
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            className="glass-card p-6"
          >
            <h3 className="text-white font-semibold mb-4">Upload Video</h3>
            
            <div
              {...getRootProps()}
              className={`
                upload-area p-8 mb-4
                ${isDragActive ? 'dragover' : ''}
                ${file ? 'border-primary-500 bg-primary-500/5' : ''}
              `}
            >
              <input {...getInputProps()} />
              
              {file ? (
                <div className="text-center">
                  <FiVideo className="w-12 h-12 text-green-400 mx-auto mb-3" />
                  <p className="text-white font-medium">{file.name}</p>
                  <p className="text-sm text-gray-400 mt-1">
                    {formatFileSize(file.size)}
                  </p>
                </div>
              ) : (
                <>
                  <FiUpload className="w-12 h-12 text-gray-400 mx-auto mb-3" />
                  <p className="text-white font-medium mb-1">
                    {isDragActive ? 'Drop here' : 'Drag & drop'}
                  </p>
                  <p className="text-sm text-gray-400 mb-3">
                    or click to browse
                  </p>
                  <p className="text-xs text-gray-500">
                    Supports: MP4, AVI, MOV, MKV (Max 100MB)
                  </p>
                </>
              )}
            </div>

            {/* Options */}
            <div className="space-y-4">
              <div>
                <label className="block text-gray-400 text-sm mb-2">Sample Rate (fps)</label>
                <select
                  value={options.sampleRate}
                  onChange={(e) => setOptions({...options, sampleRate: Number(e.target.value)})}
                  className="input-field"
                >
                  <option value={0.5}>0.5 fps (Slow, accurate)</option>
                  <option value={1}>1 fps (Recommended)</option>
                  <option value={2}>2 fps (Fast)</option>
                  <option value={5}>5 fps (Very fast)</option>
                </select>
              </div>

              <div>
                <label className="block text-gray-400 text-sm mb-2">Max Frames</label>
                <select
                  value={options.maxFrames}
                  onChange={(e) => setOptions({...options, maxFrames: Number(e.target.value)})}
                  className="input-field"
                >
                  <option value={100}>100 frames</option>
                  <option value={300}>300 frames</option>
                  <option value={500}>500 frames</option>
                  <option value={1000}>1000 frames</option>
                </select>
              </div>

              <label className="flex items-center gap-3 p-3 glass rounded-xl cursor-pointer">
                <input
                  type="checkbox"
                  checked={options.analyzeAudio}
                  onChange={(e) => setOptions({...options, analyzeAudio: e.target.checked})}
                  className="w-4 h-4 rounded border-white/10 bg-white/5 text-primary-500"
                />
                <div>
                  <p className="text-white text-sm font-medium">Analyze Audio</p>
                  <p className="text-xs text-gray-400">Detect voice deepfakes</p>
                </div>
              </label>
            </div>

            {/* Action buttons */}
            <div className="flex gap-3 mt-6">
              <Button
                variant="primary"
                size="lg"
                fullWidth
                onClick={handleAnalyze}
                disabled={!file || loading}
                loading={loading}
                icon={<FiSearch className="w-5 h-5" />}
              >
                Analyze
              </Button>
              
              {file && (
                <Button
                  variant="ghost"
                  size="lg"
                  onClick={handleReset}
                  icon={<FiRefreshCw className="w-5 h-5" />}
                />
              )}
            </div>
          </motion.div>
        </div>

        {/* Right column - Preview/Results */}
        <div className="lg:col-span-2">
          <AnimatePresence mode="wait">
            {activeTab === 'upload' && !file && (
              <motion.div
                key="empty"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="glass-card p-12 text-center"
              >
                <FiVideo className="w-16 h-16 text-gray-600 mx-auto mb-4" />
                <h3 className="text-xl text-white font-medium mb-2">No Video Selected</h3>
                <p className="text-gray-400">
                  Upload a video to begin deepfake detection
                </p>
              </motion.div>
            )}

            {activeTab === 'preview' && preview && !result && (
              <motion.div
                key="preview"
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.95 }}
                className="glass-card p-6"
              >
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-white font-semibold">Video Preview</h3>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={handleReset}
                    icon={<FiX className="w-4 h-4" />}
                  />
                </div>
                
                <div className="rounded-xl overflow-hidden border border-white/10 bg-black">
                  <ReactPlayer
                    url={preview}
                    controls
                    width="100%"
                    height="auto"
                  />
                </div>

                {loading && (
                  <div className="mt-6">
                    <Loader text={`Analyzing video... ${progress}%`} />
                    <div className="progress-bar mt-4">
                      <div 
                        className="progress-fill"
                        style={{ width: `${progress}%` }}
                      />
                    </div>
                  </div>
                )}
              </motion.div>
            )}

            {activeTab === 'results' && result && (
              <motion.div
                key="results"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                className="space-y-6"
              >
                {/* Result card */}
                <div className="glass-card p-6">
                  <div className="flex items-center justify-between mb-6">
                    <h3 className="text-white font-semibold">Analysis Results</h3>
                    <div className="flex gap-2">
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={handleDownload}
                        icon={<FiDownload className="w-4 h-4" />}
                      >
                        Report
                      </Button>
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={handleReset}
                        icon={<FiRefreshCw className="w-4 h-4" />}
                      />
                    </div>
                  </div>

                  {/* Summary stats */}
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
                    <div className="text-center p-4 glass rounded-xl">
                      <p className="text-2xl text-white font-bold">{result.frames_analyzed}</p>
                      <p className="text-xs text-gray-400">Frames Analyzed</p>
                    </div>
                    <div className="text-center p-4 glass rounded-xl">
                      <p className="text-2xl text-red-400 font-bold">{result.fake_frames}</p>
                      <p className="text-xs text-gray-400">Fake Frames</p>
                    </div>
                    <div className="text-center p-4 glass rounded-xl">
                      <p className="text-2xl text-green-400 font-bold">{result.real_frames}</p>
                      <p className="text-xs text-gray-400">Real Frames</p>
                    </div>
                    <div className="text-center p-4 glass rounded-xl">
                      <p className="text-2xl text-white font-bold">{result.fake_percentage.toFixed(1)}%</p>
                      <p className="text-xs text-gray-400">Fake %</p>
                    </div>
                  </div>

                  {/* Verdict */}
                  <div className={`
                    p-6 rounded-xl text-center mb-6
                    ${result.verdict === 'FAKE' ? 'bg-red-500/10 border border-red-500/20' :
                      result.verdict === 'REAL' ? 'bg-green-500/10 border border-green-500/20' :
                      'bg-yellow-500/10 border border-yellow-500/20'}
                  `}>
                    <h4 className={`text-2xl font-bold mb-2 ${
                      result.verdict === 'FAKE' ? 'text-red-400' :
                      result.verdict === 'REAL' ? 'text-green-400' :
                      'text-yellow-400'
                    }`}>
                      {result.verdict === 'FAKE' ? 'Deepfake Detected' :
                       result.verdict === 'REAL' ? 'Authentic Video' :
                       'Suspicious Content'}
                    </h4>
                    <p className="text-gray-400">
                      Confidence: {(result.confidence_avg * 100).toFixed(1)}%
                    </p>
                  </div>

                  {/* Timeline preview */}
                  <div>
                    <p className="text-gray-400 text-sm mb-2">Detection Timeline</p>
                    <div className="flex h-8 gap-0.5">
                      {result.timeline?.slice(0, 50).map((frame, idx) => (
                        <div
                          key={idx}
                          className={`flex-1 ${
                            frame.is_fake ? 'bg-red-500' : 'bg-green-500'
                          }`}
                          style={{ opacity: frame.confidence }}
                          title={`Frame ${idx}: ${frame.is_fake ? 'FAKE' : 'REAL'} (${(frame.confidence*100).toFixed(0)}%)`}
                        />
                      ))}
                    </div>
                  </div>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </div>
    </div>
  );
};

export default VideoUploader;