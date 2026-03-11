// API Endpoints
export const API_ENDPOINTS = {
  HEALTH: '/api/health',
  STATUS: '/api/status',
  STATS: '/api/stats',
  DETECT_IMAGE: '/api/detect/image',
  DETECT_IMAGE_BATCH: '/api/detect/image/batch',
  DETECT_VIDEO: '/api/detect/video',
  DETECT_AUDIO: '/api/detect/audio',
  DETECT_TEXT: '/api/detect/text',
  DETECT_TEXT_BATCH: '/api/detect/text/batch',
  ANALYZE_URL: '/api/analyze/url',
  BLOCKCHAIN_VERIFY: '/api/blockchain/verify',
  BLOCKCHAIN_STORE: '/api/blockchain/store',
  RECENT_DETECTIONS: '/api/detections/recent',
  REPORTS: '/api/reports',
  CERTIFICATES: '/api/certificates',
};

// WebSocket Events
export const WS_EVENTS = {
  CONNECT: 'connect',
  DISCONNECT: 'disconnect',
  DETECTION: 'detection',
  ERROR: 'error',
  PING: 'ping',
  PONG: 'pong',
};

// Detection Types
export const DETECTION_TYPES = {
  IMAGE: 'image',
  VIDEO: 'video',
  AUDIO: 'audio',
  TEXT: 'text',
  URL: 'url',
  LIVE: 'live',
};

// Detection Results
export const DETECTION_RESULTS = {
  REAL: 'REAL',
  FAKE: 'FAKE',
  SUSPICIOUS: 'SUSPICIOUS',
  ERROR: 'ERROR',
  UNCERTAIN: 'UNCERTAIN',
};

// File Upload Limits
export const UPLOAD_LIMITS = {
  IMAGE_MAX_SIZE: 10 * 1024 * 1024, // 10MB
  VIDEO_MAX_SIZE: 100 * 1024 * 1024, // 100MB
  AUDIO_MAX_SIZE: 50 * 1024 * 1024, // 50MB
  MAX_FILES_PER_BATCH: 10,
};

// Supported File Types
export const SUPPORTED_FORMATS = {
  IMAGE: ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'],
  VIDEO: ['.mp4', '.avi', '.mov', '.mkv', '.webm'],
  AUDIO: ['.mp3', '.wav', '.flac', '.aac', '.ogg'],
};

// Chart Colors
export const CHART_COLORS = {
  primary: '#3b82f6',
  secondary: '#8b5cf6',
  success: '#10b981',
  warning: '#f59e0b',
  error: '#ef4444',
  info: '#60a5fa',
  purple: '#a78bfa',
  pink: '#ec4899',
};

// Status Messages
export const STATUS_MESSAGES = {
  LOADING: 'Loading...',
  PROCESSING: 'Processing...',
  UPLOADING: 'Uploading...',
  ANALYZING: 'Analyzing...',
  SUCCESS: 'Operation completed successfully',
  ERROR: 'An error occurred',
  WARNING: 'Please check your input',
  INFO: 'Processing your request',
};

// Routes
export const ROUTES = {
  HOME: '/',
  DASHBOARD: '/dashboard',
  IMAGE: '/image',
  VIDEO: '/video',
  AUDIO: '/audio',
  TEXT: '/text',
  URL: '/url',
  LIVE: '/live',
  PROFILE: '/profile',
  SETTINGS: '/settings',
  HELP: '/help',
};