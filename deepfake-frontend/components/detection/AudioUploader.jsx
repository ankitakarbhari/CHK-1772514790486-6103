import React, { useState, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useDropzone } from 'react-dropzone';
import ReactPlayer from 'react-player';
import { 
  FiUpload, 
  FiX, 
  FiMusic,
  FiSearch,
  FiDownload,
  FiRefreshCw,
  FiShield,
  FiAlertTriangle,
  FiCheckCircle,
  FiWaveform,
  FiVolume2
} from 'react-icons/fi';
import Button from '@/components/common/Button';
import Loader from '@/components/common/Loader';
import { useAlert } from '@/components/common/Alert';
import { detectAudio } from '@/utils/api';
import { formatFileSize } from '@/utils/helpers';

/**
 * AudioUploader Component - Upload and analyze audio for deepfakes
 */

const AudioUploader = () => {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [activeTab, setActiveTab] = useState('upload');
  const [audioContext, setAudioContext] = useState(null);
  const [audioData, setAudioData] = useState(null);
  const [waveform, setWaveform] = useState([]);

  const { success, error, info } = useAlert();

  // Handle file drop
  const onDrop = useCallback(async (acceptedFiles) => {
    const file = acceptedFiles[0];
    if (!file) return;

    // Validate file size (50MB max)
    if (file.size > 50 * 1024 * 1024) {
      error('File too large', 'Maximum file size is 50MB');
      return;
    }

    setFile(file);
    setResult(null);
    setActiveTab('preview');

    // Create preview URL
    const url = URL.createObjectURL(file);
    setPreview(url);

    // Generate waveform
    await generateWaveform(file);

    info('Audio ready', 'Click "Analyze Audio" to start detection');
  }, [error, info]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'audio/*': ['.mp3', '.wav', '.flac', '.aac', '.ogg', '.m4a']
    },
    maxFiles: 1,
  });

  // Generate waveform visualization
  const generateWaveform = async (file) => {
    try {
      const arrayBuffer = await file.arrayBuffer();
      const audioContext = new (window.AudioContext || window.webkitAudioContext)();
      const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
      
      // Get audio data
      const channelData = audioBuffer.getChannelData(0);
      const samples = 100;
      const blockSize = Math.floor(channelData.length / samples);
      const waveform = [];
      
      for (let i = 0; i < samples; i++) {
        let blockStart = blockSize * i;
        let sum = 0;
        for (let j = 0; j < blockSize; j++) {
          sum += Math.abs(channelData[blockStart + j]);
        }
        waveform.push(sum / blockSize);
      }
      
      setWaveform(waveform);
      setAudioContext(audioContext);
    } catch (err) {
      console.error('Waveform generation failed:', err);
    }
  };

  // Handle analysis
  const handleAnalyze = async () => {
    if (!file) {
      error('No file selected', 'Please upload an audio file first');
      return;
    }

    setLoading(true);
    setActiveTab('results');

    try {
      const response = await detectAudio(file);
      setResult(response.result);
      
      success('Analysis complete', `Audio is ${response.result.prediction === 'FAKE' ? 'FAKE' : 'REAL'}`);
    } catch (err) {
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
    if (audioContext) {
      audioContext.close();
    }
    setFile(null);
    setPreview(null);
    setResult(null);
    setAudioContext(null);
    setWaveform([]);
    setActiveTab('upload');
  };

  // Download report
  const handleDownload = () => {
    const report = {
      filename: file?.name,
      timestamp: new Date().toISOString(),
      result: result,
    };

    const blob = new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `audio-report-${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-white">Audio Deepfake Detection</h2>
          <p className="text-gray-400 mt-1">
            Upload audio to detect AI-generated voices and synthetic speech
          </p>
        </div>
        <div className="flex items-center gap-2 text-sm text-gray-400">
          <FiShield className="w-4 h-4" />
          <span>94.5% Accuracy</span>
        </div>
      </div>

      {/* Main content */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left column - Upload */}
        <div className="lg:col-span-1 space-y-6">
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            className="glass-card p-6"
          >
            <h3 className="text-white font-semibold mb-4">Upload Audio</h3>
            
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
                  <FiMusic className="w-12 h-12 text-green-400 mx-auto mb-3" />
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
                    Supports: MP3, WAV, FLAC, AAC (Max 50MB)
                  </p>
                </>
              )}
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
                <FiMusic className="w-16 h-16 text-gray-600 mx-auto mb-4" />
                <h3 className="text-xl text-white font-medium mb-2">No Audio Selected</h3>
                <p className="text-gray-400">
                  Upload an audio file to begin deepfake detection
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
                  <h3 className="text-white font-semibold">Audio Preview</h3>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={handleReset}
                    icon={<FiX className="w-4 h-4" />}
                  />
                </div>
                
                {/* Audio player */}
                <div className="glass p-4 rounded-xl mb-4">
                  <audio controls className="w-full">
                    <source src={preview} type={file?.type} />
                    Your browser does not support the audio element.
                  </audio>
                </div>

                {/* Waveform visualization */}
                {waveform.length > 0 && (
                  <div className="glass p-4 rounded-xl">
                    <p className="text-gray-400 text-sm mb-3">Waveform</p>
                    <div className="flex items-end h-24 gap-0.5">
                      {waveform.map((value, idx) => (
                        <div
                          key={idx}
                          className="flex-1 bg-primary-500/50 hover:bg-primary-400 transition-colors"
                          style={{ height: `${value * 100}%` }}
                        />
                      ))}
                    </div>
                  </div>
                )}

                {loading && (
                  <div className="mt-6">
                    <Loader text="Analyzing audio..." />
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

                  {/* Result badge */}
                  <div className="text-center mb-6">
                    <div className={`
                      w-24 h-24 mx-auto rounded-full flex items-center justify-center mb-4
                      ${result.prediction === 'FAKE' 
                        ? 'bg-red-500/20 text-red-400' 
                        : 'bg-green-500/20 text-green-400'}
                    `}>
                      {result.prediction === 'FAKE' ? (
                        <FiAlertTriangle className="w-12 h-12" />
                      ) : (
                        <FiCheckCircle className="w-12 h-12" />
                      )}
                    </div>
                    <h4 className={`text-3xl font-bold mb-2 ${
                      result.prediction === 'FAKE' ? 'text-red-400' : 'text-green-400'
                    }`}>
                      {result.prediction}
                    </h4>
                    <p className="text-gray-400">
                      Confidence: {(result.confidence * 100).toFixed(1)}%
                    </p>
                  </div>

                  {/* Probability bars */}
                  <div className="space-y-4">
                    <div>
                      <p className="text-gray-400 text-sm mb-1">AI Probability</p>
                      <div className="flex items-center gap-3">
                        <div className="progress-bar flex-1">
                          <div 
                            className="progress-fill bg-red-500"
                            style={{ width: `${result.ai_probability * 100}%` }}
                          />
                        </div>
                        <span className="text-white font-medium">
                          {(result.ai_probability * 100).toFixed(1)}%
                        </span>
                      </div>
                    </div>

                    <div>
                      <p className="text-gray-400 text-sm mb-1">Human Probability</p>
                      <div className="flex items-center gap-3">
                        <div className="progress-bar flex-1">
                          <div 
                            className="progress-fill bg-green-500"
                            style={{ width: `${result.human_probability * 100}%` }}
                          />
                        </div>
                        <span className="text-white font-medium">
                          {(result.human_probability * 100).toFixed(1)}%
                        </span>
                      </div>
                    </div>
                  </div>

                  {/* Warnings */}
                  {result.warning_flags?.length > 0 && (
                    <div className="mt-6 p-4 glass rounded-xl">
                      <p className="text-yellow-400 font-medium mb-2">Warnings</p>
                      <ul className="space-y-2">
                        {result.warning_flags.map((warning, idx) => (
                          <li key={idx} className="text-sm text-gray-300 flex items-start gap-2">
                            <FiAlertTriangle className="w-4 h-4 text-yellow-400 mt-0.5 flex-shrink-0" />
                            <span>{warning}</span>
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </div>
    </div>
  );
};

export default AudioUploader;