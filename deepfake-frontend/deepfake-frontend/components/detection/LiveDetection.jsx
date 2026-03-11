import React, { useState, useRef, useEffect, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import Webcam from 'react-webcam';
import { 
  FiCamera,
  FiVideo,
  FiMic,
  FiMonitor,
  FiPlay,
  FiStopCircle,
  FiAlertTriangle,
  FiCheckCircle,
  FiActivity,
  FiUsers,
  FiSettings,
  FiMaximize2,
  FiMinimize2,
  FiDownload,
  FiRefreshCw
} from 'react-icons/fi';
import Button from '@/components/common/Button';
import Loader from '@/components/common/Loader';
import { useAlert } from '@/components/common/Alert';
import useWebSocket from '@/hooks/useWebSocket';

/**
 * LiveDetection Component - Real-time deepfake detection from camera/microphone
 */

const LiveDetection = () => {
  const [isActive, setIsActive] = useState(false);
  const [mode, setMode] = useState('camera'); // 'camera', 'screen', 'audio'
  const [devices, setDevices] = useState([]);
  const [selectedDevice, setSelectedDevice] = useState(null);
  const [detections, setDetections] = useState([]);
  const [stats, setStats] = useState({
    framesProcessed: 0,
    deepfakesDetected: 0,
    facesDetected: 0,
    confidenceAvg: 0,
  });
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  const [settings, setSettings] = useState({
    processEveryNFrames: 5,
    showHeatmap: true,
    alertThreshold: 0.7,
    enableAudio: true,
  });

  const webcamRef = useRef(null);
  const mediaRecorderRef = useRef(null);
  const canvasRef = useRef(null);
  const containerRef = useRef(null);

  const { success, error, warning } = useAlert();
  const { isConnected, lastMessage, sendMessage } = useWebSocket(
    process.env.NEXT_PUBLIC_WS_URL + '/ws/detect/live'
  );

  // Get available cameras
  useEffect(() => {
    const getDevices = async () => {
      const devices = await navigator.mediaDevices.enumerateDevices();
      const videoDevices = devices.filter(device => device.kind === 'videoinput');
      setDevices(videoDevices);
      if (videoDevices.length > 0) {
        setSelectedDevice(videoDevices[0].deviceId);
      }
    };
    getDevices();
  }, []);

  // Handle WebSocket messages
  useEffect(() => {
    if (lastMessage) {
      if (lastMessage.type === 'detection') {
        handleDetection(lastMessage);
      } else if (lastMessage.type === 'error') {
        error('Detection Error', lastMessage.message);
      }
    }
  }, [lastMessage]);

  const handleDetection = (data) => {
    // Update detections list
    setDetections(prev => [data, ...prev].slice(0, 10));
    
    // Update stats
    setStats(prev => ({
      framesProcessed: prev.framesProcessed + 1,
      deepfakesDetected: prev.deepfakesDetected + (data.result.prediction === 'FAKE' ? 1 : 0),
      facesDetected: data.faces_detected,
      confidenceAvg: (prev.confidenceAvg * prev.framesProcessed + data.result.confidence) / (prev.framesProcessed + 1),
    }));

    // Draw overlay on canvas
    drawOverlay(data);
  };

  const drawOverlay = (data) => {
    const canvas = canvasRef.current;
    if (!canvas || !webcamRef.current) return;

    const ctx = canvas.getContext('2d');
    const video = webcamRef.current.video;

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (data.face_box) {
      const { x, y, width, height } = data.face_box;
      
      // Draw bounding box
      ctx.strokeStyle = data.result.prediction === 'FAKE' ? '#ef4444' : '#10b981';
      ctx.lineWidth = 3;
      ctx.strokeRect(x, y, width, height);

      // Draw label
      ctx.fillStyle = data.result.prediction === 'FAKE' ? '#ef4444' : '#10b981';
      ctx.font = 'bold 16px Inter';
      ctx.fillStyle = 'rgba(0,0,0,0.7)';
      ctx.fillRect(x, y - 30, 150, 30);
      
      ctx.fillStyle = '#ffffff';
      ctx.fillText(
        `${data.result.prediction} (${(data.result.confidence * 100).toFixed(1)}%)`,
        x + 5,
        y - 8
      );
    }
  };

  const startDetection = useCallback(() => {
    if (!isConnected) {
      error('Not Connected', 'WebSocket not connected');
      return;
    }

    setIsActive(true);
    sendMessage({ type: 'start', mode, deviceId: selectedDevice });
    success('Detection Started', 'Live analysis is now active');
  }, [isConnected, mode, selectedDevice, sendMessage]);

  const stopDetection = useCallback(() => {
    setIsActive(false);
    sendMessage({ type: 'stop' });
    
    // Clear canvas
    const canvas = canvasRef.current;
    if (canvas) {
      const ctx = canvas.getContext('2d');
      ctx.clearRect(0, 0, canvas.width, canvas.height);
    }
  }, [sendMessage]);

  const toggleFullscreen = () => {
    if (!isFullscreen) {
      containerRef.current?.requestFullscreen();
    } else {
      document.exitFullscreen();
    }
    setIsFullscreen(!isFullscreen);
  };

  const videoConstraints = {
    deviceId: selectedDevice,
    width: 1280,
    height: 720,
    facingMode: 'user',
  };

  return (
    <div ref={containerRef} className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-white">Live Deepfake Detection</h2>
          <p className="text-gray-400 mt-1">
            Real-time analysis from camera, screen share, or microphone
          </p>
        </div>
        <div className="flex items-center gap-3">
          <div className={`flex items-center gap-2 px-3 py-1.5 rounded-full ${
            isConnected ? 'bg-green-500/20 text-green-400' : 'bg-red-500/20 text-red-400'
          }`}>
            <span className={`w-2 h-2 rounded-full ${isConnected ? 'bg-green-400 animate-pulse' : 'bg-red-400'}`} />
            <span className="text-xs font-medium">{isConnected ? 'Connected' : 'Disconnected'}</span>
          </div>
          <Button
            variant={isActive ? 'danger' : 'primary'}
            size="md"
            onClick={isActive ? stopDetection : startDetection}
            icon={isActive ? <FiStopCircle className="w-5 h-5" /> : <FiPlay className="w-5 h-5" />}
            disabled={!isConnected}
          >
            {isActive ? 'Stop' : 'Start Detection'}
          </Button>
        </div>
      </div>

      {/* Mode Selection */}
      <div className="flex gap-3">
        <button
          onClick={() => setMode('camera')}
          className={`flex items-center gap-2 px-4 py-2 rounded-xl transition-all ${
            mode === 'camera'
              ? 'bg-primary-500 text-white'
              : 'glass text-gray-400 hover:text-white hover:bg-white/10'
          }`}
        >
          <FiCamera className="w-5 h-5" />
          <span>Camera</span>
        </button>
        <button
          onClick={() => setMode('screen')}
          className={`flex items-center gap-2 px-4 py-2 rounded-xl transition-all ${
            mode === 'screen'
              ? 'bg-primary-500 text-white'
              : 'glass text-gray-400 hover:text-white hover:bg-white/10'
          }`}
        >
          <FiMonitor className="w-5 h-5" />
          <span>Screen Share</span>
        </button>
        <button
          onClick={() => setMode('audio')}
          className={`flex items-center gap-2 px-4 py-2 rounded-xl transition-all ${
            mode === 'audio'
              ? 'bg-primary-500 text-white'
              : 'glass text-gray-400 hover:text-white hover:bg-white/10'
          }`}
        >
          <FiMic className="w-5 h-5" />
          <span>Audio Only</span>
        </button>
      </div>

      {/* Main Content */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Video/Audio Feed */}
        <div className="lg:col-span-2 space-y-4">
          <div className="relative glass-card overflow-hidden">
            {/* Video Feed */}
            {mode !== 'audio' ? (
              <div className="relative aspect-video bg-black">
                <Webcam
                  ref={webcamRef}
                  audio={settings.enableAudio}
                  videoConstraints={videoConstraints}
                  className="w-full h-full object-cover"
                  mirrored={true}
                />
                
                {/* Overlay Canvas */}
                <canvas
                  ref={canvasRef}
                  className="absolute top-0 left-0 w-full h-full pointer-events-none"
                />

                {/* Status Overlay */}
                {!isActive && (
                  <div className="absolute inset-0 flex items-center justify-center bg-dark-500/50 backdrop-blur-sm">
                    <div className="text-center">
                      <FiCamera className="w-12 h-12 text-gray-400 mx-auto mb-3" />
                      <p className="text-white font-medium">Detection Paused</p>
                      <p className="text-sm text-gray-400 mt-1">Click Start to begin analysis</p>
                    </div>
                  </div>
                )}
              </div>
            ) : (
              <div className="aspect-video flex items-center justify-center bg-dark-300">
                <div className="text-center">
                  <FiMic className="w-16 h-16 text-gray-600 mx-auto mb-4" />
                  <p className="text-white font-medium">Audio Mode Active</p>
                  <p className="text-sm text-gray-400 mt-1">Listening for voice deepfakes</p>
                </div>
              </div>
            )}

            {/* Controls Overlay */}
            <div className="absolute bottom-4 left-4 right-4 flex items-center justify-between">
              <div className="flex gap-2">
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={toggleFullscreen}
                  icon={isFullscreen ? <FiMinimize2 className="w-4 h-4" /> : <FiMaximize2 className="w-4 h-4" />}
                />
                {mode === 'camera' && devices.length > 0 && (
                  <select
                    value={selectedDevice || ''}
                    onChange={(e) => setSelectedDevice(e.target.value)}
                    className="glass px-3 py-1.5 rounded-lg text-sm text-white"
                    disabled={isActive}
                  >
                    {devices.map(device => (
                      <option key={device.deviceId} value={device.deviceId}>
                        {device.label || `Camera ${device.deviceId.slice(0, 5)}`}
                      </option>
                    ))}
                  </select>
                )}
              </div>

              {/* Recording Indicator */}
              {isActive && (
                <div className="flex items-center gap-2 px-3 py-1.5 glass rounded-full">
                  <span className="w-2 h-2 bg-red-500 rounded-full animate-pulse" />
                  <span className="text-xs text-white">LIVE</span>
                </div>
              )}
            </div>
          </div>

          {/* Device Settings */}
          {showSettings && (
            <motion.div
              initial={{ opacity: 0, y: -20 }}
              animate={{ opacity: 1, y: 0 }}
              className="glass-card p-4"
            >
              <h3 className="text-white font-semibold mb-4">Detection Settings</h3>
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-gray-400 text-sm mb-2">Process Every N Frames</label>
                  <select
                    value={settings.processEveryNFrames}
                    onChange={(e) => setSettings({...settings, processEveryNFrames: Number(e.target.value)})}
                    className="input-field"
                  >
                    <option value={1}>Every frame (Slow)</option>
                    <option value={3}>Every 3 frames</option>
                    <option value={5}>Every 5 frames</option>
                    <option value={10}>Every 10 frames (Fast)</option>
                  </select>
                </div>
                <div>
                  <label className="block text-gray-400 text-sm mb-2">Alert Threshold</label>
                  <input
                    type="range"
                    min="0.5"
                    max="0.95"
                    step="0.05"
                    value={settings.alertThreshold}
                    onChange={(e) => setSettings({...settings, alertThreshold: parseFloat(e.target.value)})}
                    className="w-full"
                  />
                  <p className="text-xs text-gray-400 mt-1">{(settings.alertThreshold * 100).toFixed(0)}% confidence</p>
                </div>
                <label className="flex items-center gap-3 p-3 glass rounded-xl cursor-pointer">
                  <input
                    type="checkbox"
                    checked={settings.showHeatmap}
                    onChange={(e) => setSettings({...settings, showHeatmap: e.target.checked})}
                    className="w-4 h-4 rounded border-white/10 bg-white/5 text-primary-500"
                  />
                  <span className="text-white text-sm">Show Heatmap</span>
                </label>
                <label className="flex items-center gap-3 p-3 glass rounded-xl cursor-pointer">
                  <input
                    type="checkbox"
                    checked={settings.enableAudio}
                    onChange={(e) => setSettings({...settings, enableAudio: e.target.checked})}
                    className="w-4 h-4 rounded border-white/10 bg-white/5 text-primary-500"
                  />
                  <span className="text-white text-sm">Enable Audio</span>
                </label>
              </div>
            </motion.div>
          )}
        </div>

        {/* Stats and Detections Panel */}
        <div className="lg:col-span-1 space-y-6">
          {/* Stats Cards */}
          <div className="glass-card p-6">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-white font-semibold">Live Statistics</h3>
              <button
                onClick={() => setShowSettings(!showSettings)}
                className="p-2 rounded-lg hover:bg-white/10 text-gray-400 hover:text-white"
              >
                <FiSettings className="w-5 h-5" />
              </button>
            </div>

            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <span className="text-gray-400">Frames Processed</span>
                <span className="text-white font-medium">{stats.framesProcessed}</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-gray-400">Deepfakes Detected</span>
                <span className="text-red-400 font-medium">{stats.deepfakesDetected}</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-gray-400">Faces Detected</span>
                <span className="text-white font-medium">{stats.facesDetected}</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-gray-400">Avg Confidence</span>
                <span className="text-white font-medium">{(stats.confidenceAvg * 100).toFixed(1)}%</span>
              </div>
            </div>

            {/* Progress Bar */}
            <div className="mt-4 pt-4 border-t border-white/10">
              <div className="flex items-center justify-between mb-2">
                <span className="text-xs text-gray-400">Deepfake Rate</span>
                <span className="text-xs text-white">
                  {stats.framesProcessed > 0 
                    ? ((stats.deepfakesDetected / stats.framesProcessed) * 100).toFixed(1)
                    : 0}%
                </span>
              </div>
              <div className="progress-bar">
                <div 
                  className="progress-fill bg-red-500"
                  style={{
                    width: stats.framesProcessed > 0 
                      ? `${(stats.deepfakesDetected / stats.framesProcessed) * 100}%`
                      : '0%'
                  }}
                />
              </div>
            </div>
          </div>

          {/* Recent Detections */}
          <div className="glass-card p-6">
            <h3 className="text-white font-semibold mb-4">Recent Detections</h3>
            
            <div className="space-y-3 max-h-96 overflow-y-auto">
              <AnimatePresence>
                {detections.map((det, idx) => (
                  <motion.div
                    key={idx}
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    exit={{ opacity: 0, x: 20 }}
                    className={`p-3 glass rounded-xl ${
                      det.result.prediction === 'FAKE' 
                        ? 'border-l-4 border-red-500' 
                        : 'border-l-4 border-green-500'
                    }`}
                  >
                    <div className="flex items-center justify-between mb-2">
                      <span className={`text-xs font-medium ${
                        det.result.prediction === 'FAKE' ? 'text-red-400' : 'text-green-400'
                      }`}>
                        {det.result.prediction}
                      </span>
                      <span className="text-xs text-gray-400">
                        {(det.result.confidence * 100).toFixed(1)}%
                      </span>
                    </div>
                    <div className="flex items-center justify-between text-xs text-gray-400">
                      <span>Frame {det.frame}</span>
                      <span>{det.faces_detected} face(s)</span>
                    </div>
                  </motion.div>
                ))}
              </AnimatePresence>

              {detections.length === 0 && (
                <div className="text-center py-8">
                  <FiActivity className="w-8 h-8 text-gray-600 mx-auto mb-2" />
                  <p className="text-sm text-gray-400">No detections yet</p>
                </div>
              )}
            </div>
          </div>

          {/* Active Participants (for video calls) */}
          {mode === 'screen' && (
            <div className="glass-card p-6">
              <h3 className="text-white font-semibold mb-4">Active Participants</h3>
              <div className="space-y-3">
                <div className="flex items-center gap-3 p-2 glass rounded-lg">
                  <div className="w-8 h-8 rounded-full bg-primary-500/20 flex items-center justify-center">
                    <FiUsers className="w-4 h-4 text-primary-400" />
                  </div>
                  <div className="flex-1">
                    <p className="text-white text-sm">Participant 1</p>
                    <p className="text-xs text-green-400">Speaking</p>
                  </div>
                </div>
                <div className="flex items-center gap-3 p-2 glass rounded-lg">
                  <div className="w-8 h-8 rounded-full bg-gray-500/20 flex items-center justify-center">
                    <FiUsers className="w-4 h-4 text-gray-400" />
                  </div>
                  <div className="flex-1">
                    <p className="text-white text-sm">Participant 2</p>
                    <p className="text-xs text-gray-400">Muted</p>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default LiveDetection;