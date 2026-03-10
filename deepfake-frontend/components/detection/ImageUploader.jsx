import React, { useState, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useDropzone } from 'react-dropzone';
import { 
  FiUpload, 
  FiX, 
  FiImage, 
  FiSearch,
  FiDownload,
  FiRefreshCw,
  FiShield,
  FiAlertTriangle,
  FiCheckCircle,
  FiCamera,
  FiFile
} from 'react-icons/fi';
import Button from '@/components/common/Button';
import Loader from '@/components/common/Loader';
import { useAlert } from '@/components/common/Alert';
import { detectImage } from '@/utils/api';
import { formatFileSize } from '@/utils/helpers';

/**
 * ImageUploader Component - Upload and analyze images for deepfakes
 */

const ImageUploader = () => {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [heatmap, setHeatmap] = useState(null);
  const [activeTab, setActiveTab] = useState('upload');
  const [options, setOptions] = useState({
    generateHeatmap: true,
    storeOnBlockchain: false,
  });

  const { success, error, info } = useAlert();

  // Handle file drop
  const onDrop = useCallback((acceptedFiles) => {
    const file = acceptedFiles[0];
    if (!file) return;

    // Validate file size (10MB max)
    if (file.size > 10 * 1024 * 1024) {
      error('File too large', 'Maximum file size is 10MB');
      return;
    }

    setFile(file);
    setResult(null);
    setHeatmap(null);
    setActiveTab('preview');

    // Create preview
    const reader = new FileReader();
    reader.onload = () => {
      setPreview(reader.result);
    };
    reader.readAsDataURL(file);

    info('File ready', 'Click "Analyze Image" to start detection');
  }, [error, info]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp']
    },
    maxFiles: 1,
  });

  // Handle analysis
  const handleAnalyze = async () => {
    if (!file) {
      error('No file selected', 'Please upload an image first');
      return;
    }

    setLoading(true);
    setActiveTab('results');

    try {
      const response = await detectImage(file, options);
      setResult(response.result);
      
      if (response.heatmap) {
        setHeatmap(response.heatmap);
      }
      
      success('Analysis complete', `Image is ${response.result.prediction === 'FAKE' ? 'FAKE' : 'REAL'}`);
    } catch (err) {
      error('Analysis failed', err.response?.data?.detail || 'Please try again');
    } finally {
      setLoading(false);
    }
  };

  // Reset upload
  const handleReset = () => {
    setFile(null);
    setPreview(null);
    setResult(null);
    setHeatmap(null);
    setActiveTab('upload');
  };

  // Download report
  const handleDownload = () => {
    // Create report data
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
    a.download = `deepfake-report-${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-white">Image Deepfake Detection</h2>
          <p className="text-gray-400 mt-1">
            Upload an image to check if it's AI-generated or manipulated
          </p>
        </div>
        <div className="flex items-center gap-2 text-sm text-gray-400">
          <FiShield className="w-4 h-4" />
          <span>98.3% Accuracy</span>
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
            <h3 className="text-white font-semibold mb-4">Upload Image</h3>
            
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
                  <FiCheckCircle className="w-12 h-12 text-green-400 mx-auto mb-3" />
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
                    Supports: JPG, PNG, GIF, WEBP (Max 10MB)
                  </p>
                </>
              )}
            </div>

            {/* Options */}
            <div className="space-y-3">
              <label className="flex items-center gap-3 p-3 glass rounded-xl cursor-pointer">
                <input
                  type="checkbox"
                  checked={options.generateHeatmap}
                  onChange={(e) => setOptions({...options, generateHeatmap: e.target.checked})}
                  className="w-4 h-4 rounded border-white/10 bg-white/5 text-primary-500"
                />
                <div>
                  <p className="text-white text-sm font-medium">Generate Heatmap</p>
                  <p className="text-xs text-gray-400">Visualize manipulated regions</p>
                </div>
              </label>

              <label className="flex items-center gap-3 p-3 glass rounded-xl cursor-pointer">
                <input
                  type="checkbox"
                  checked={options.storeOnBlockchain}
                  onChange={(e) => setOptions({...options, storeOnBlockchain: e.target.checked})}
                  className="w-4 h-4 rounded border-white/10 bg-white/5 text-primary-500"
                />
                <div>
                  <p className="text-white text-sm font-medium">Store on Blockchain</p>
                  <p className="text-xs text-gray-400">Immutable verification record</p>
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

          {/* Info card */}
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.1 }}
            className="glass-card p-6"
          >
            <h3 className="text-white font-semibold mb-4">Detection Features</h3>
            <ul className="space-y-3 text-sm">
              <li className="flex items-center gap-3 text-gray-300">
                <span className="w-1.5 h-1.5 bg-primary-400 rounded-full" />
                Face manipulation detection
              </li>
              <li className="flex items-center gap-3 text-gray-300">
                <span className="w-1.5 h-1.5 bg-primary-400 rounded-full" />
                GAN-generated image identification
              </li>
              <li className="flex items-center gap-3 text-gray-300">
                <span className="w-1.5 h-1.5 bg-primary-400 rounded-full" />
                Metadata analysis
              </li>
              <li className="flex items-center gap-3 text-gray-300">
                <span className="w-1.5 h-1.5 bg-primary-400 rounded-full" />
                Compression artifact detection
              </li>
            </ul>
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
                <FiImage className="w-16 h-16 text-gray-600 mx-auto mb-4" />
                <h3 className="text-xl text-white font-medium mb-2">No Image Selected</h3>
                <p className="text-gray-400">
                  Upload an image to begin deepfake detection
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
                  <h3 className="text-white font-semibold">Image Preview</h3>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={handleReset}
                    icon={<FiX className="w-4 h-4" />}
                  />
                </div>
                
                <div className="rounded-xl overflow-hidden border border-white/10">
                  <img src={preview} alt="Preview" className="w-full h-auto" />
                </div>

                {loading && (
                  <div className="mt-6">
                    <Loader text="Analyzing image..." />
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
                    <h3 className="text-white font-semibold">Detection Results</h3>
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

                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    {/* Result badge */}
                    <div className="text-center p-6 glass rounded-xl">
                      <div className={`
                        w-20 h-20 mx-auto rounded-full flex items-center justify-center mb-4
                        ${result.prediction === 'FAKE' 
                          ? 'bg-red-500/20 text-red-400' 
                          : 'bg-green-500/20 text-green-400'}
                      `}>
                        {result.prediction === 'FAKE' ? (
                          <FiAlertTriangle className="w-10 h-10" />
                        ) : (
                          <FiCheckCircle className="w-10 h-10" />
                        )}
                      </div>
                      <h4 className={`text-2xl font-bold mb-2 ${
                        result.prediction === 'FAKE' ? 'text-red-400' : 'text-green-400'
                      }`}>
                        {result.prediction}
                      </h4>
                      <p className="text-gray-400">
                        Confidence: {(result.confidence * 100).toFixed(1)}%
                      </p>
                    </div>

                    {/* Stats */}
                    <div className="space-y-4">
                      <div>
                        <p className="text-gray-400 text-sm mb-1">Real Probability</p>
                        <div className="flex items-center gap-3">
                          <div className="progress-bar flex-1">
                            <div 
                              className="progress-fill bg-green-500"
                              style={{ width: `${result.real_probability * 100}%` }}
                            />
                          </div>
                          <span className="text-white font-medium">
                            {(result.real_probability * 100).toFixed(1)}%
                          </span>
                        </div>
                      </div>

                      <div>
                        <p className="text-gray-400 text-sm mb-1">Fake Probability</p>
                        <div className="flex items-center gap-3">
                          <div className="progress-bar flex-1">
                            <div 
                              className="progress-fill bg-red-500"
                              style={{ width: `${result.fake_probability * 100}%` }}
                            />
                          </div>
                          <span className="text-white font-medium">
                            {(result.fake_probability * 100).toFixed(1)}%
                          </span>
                        </div>
                      </div>

                      {result.faces_detected > 0 && (
                        <div className="pt-4">
                          <p className="text-gray-400 text-sm">Faces Detected</p>
                          <p className="text-2xl text-white font-bold">
                            {result.faces_detected}
                          </p>
                        </div>
                      )}
                    </div>
                  </div>
                </div>

                {/* Heatmap */}
                {heatmap && (
                  <div className="glass-card p-6">
                    <h3 className="text-white font-semibold mb-4">Heatmap Analysis</h3>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                      <div>
                        <p className="text-gray-400 text-sm mb-2">Original</p>
                        <img 
                          src={preview} 
                          alt="Original" 
                          className="rounded-xl border border-white/10 w-full"
                        />
                      </div>
                      <div>
                        <p className="text-gray-400 text-sm mb-2">Manipulation Heatmap</p>
                        <img 
                          src={`data:image/png;base64,${heatmap.overlay}`} 
                          alt="Heatmap" 
                          className="rounded-xl border border-white/10 w-full"
                        />
                      </div>
                    </div>

                    {heatmap.manipulated_regions?.length > 0 && (
                      <div className="mt-4 p-4 glass rounded-xl">
                        <p className="text-white font-medium mb-2">Manipulated Regions</p>
                        <div className="space-y-2">
                          {heatmap.manipulated_regions.map((region, idx) => (
                            <div key={idx} className="flex items-center justify-between text-sm">
                              <span className="text-gray-400">Region {idx + 1}</span>
                              <span className="text-white">
                                Confidence: {(region.confidence * 100).toFixed(1)}%
                              </span>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                )}
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </div>
    </div>
  );
};

export default ImageUploader;