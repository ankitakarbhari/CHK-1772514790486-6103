import axios from 'axios';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor
api.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('token');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => Promise.reject(error)
);

// Response interceptor
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.code === 'ECONNABORTED') {
      console.error('Request timeout');
    }
    return Promise.reject(error);
  }
);

// ============================================
// Image Detection APIs
// ============================================

export const detectImage = async (file, options = {}) => {
  const formData = new FormData();
  formData.append('file', file);
  
  if (options.generateHeatmap) {
    formData.append('generate_heatmap', 'true');
  }
  
  if (options.storeOnBlockchain) {
    formData.append('store_on_blockchain', 'true');
  }

  const response = await api.post('/api/detect/image', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });
  return response.data;
};

export const detectImageBatch = async (files) => {
  const formData = new FormData();
  files.forEach(file => formData.append('files', file));

  const response = await api.post('/api/detect/image/batch', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });
  return response.data;
};

// ============================================
// Video Detection APIs
// ============================================

export const detectVideo = async (file, options = {}) => {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('sample_rate', options.sampleRate || 1);
  formData.append('max_frames', options.maxFrames || 300);
  formData.append('analyze_audio', options.analyzeAudio || true);

  const response = await api.post('/api/detect/video', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
    timeout: 60000,
  });
  return response.data;
};

// ============================================
// Audio Detection APIs
// ============================================

export const detectAudio = async (file) => {
  const formData = new FormData();
  formData.append('file', file);

  const response = await api.post('/api/detect/audio', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });
  return response.data;
};

// ============================================
// Text Detection APIs
// ============================================

export const detectText = async (text, options = {}) => {
  const formData = new FormData();
  formData.append('text', text);
  formData.append('return_details', options.returnDetails || true);

  const response = await api.post('/api/detect/text', formData);
  return response.data;
};

export const detectTextBatch = async (texts) => {
  const formData = new FormData();
  formData.append('texts', JSON.stringify(texts));

  const response = await api.post('/api/detect/text/batch', formData);
  return response.data;
};

// ============================================
// URL Analysis APIs
// ============================================

export const analyzeUrl = async (url, options = {}) => {
  const formData = new FormData();
  formData.append('url', url);
  formData.append('extract_content', options.extractContent || true);
  formData.append('check_ssl', options.checkSsl || true);
  formData.append('whois_lookup', options.whoisLookup || true);
  formData.append('analyze_images', options.analyzeImages || false);

  const response = await api.post('/api/analyze/url', formData);
  return response.data;
};

// ============================================
// Blockchain APIs
// ============================================

export const blockchainVerify = async (file) => {
  const formData = new FormData();
  formData.append('file', file);

  const response = await api.post('/api/blockchain/verify', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });
  return response.data;
};

export const blockchainStore = async (file, isAuthentic, confidence) => {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('is_authentic', isAuthentic);
  formData.append('confidence', confidence);

  const response = await api.post('/api/blockchain/store', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });
  return response.data;
};

// ============================================
// System APIs
// ============================================

export const getHealth = async () => {
  const response = await api.get('/api/health');
  return response.data;
};

export const getStatus = async () => {
  const response = await api.get('/api/status');
  return response.data;
};

export const getStats = async () => {
  const response = await api.get('/api/stats');
  return response.data;
};

export const getRecentDetections = async (limit = 10) => {
  const response = await api.get(`/api/detections/recent?limit=${limit}`);
  return response.data;
};

// ============================================
// File Management
// ============================================

export const getReport = async (fileId) => {
  const response = await api.get(`/api/reports/${fileId}`, {
    responseType: 'blob',
  });
  return response.data;
};

export const getCertificate = async (fileId) => {
  const response = await api.get(`/api/certificates/${fileId}`, {
    responseType: 'blob',
  });
  return response.data;
};

export default api;