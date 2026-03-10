// deepfake-frontend/utils/api.js
import axios from 'axios';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
});

// Image detection
export const detectImage = async (file, generateHeatmap = false) => {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('generate_heatmap', generateHeatmap);
  
  const response = await api.post('/api/detect/image', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
  return response.data;
};

// Text detection
export const detectText = async (text) => {
  const formData = new FormData();
  formData.append('text', text);
  
  const response = await api.post('/api/detect/text', formData);
  return response.data;
};

// Get dashboard stats
export const getStats = async () => {
  const response = await api.get('/api/stats');
  return response.data;
};

// Get recent detections
export const getRecentDetections = async () => {
  const response = await api.get('/api/detections/recent');
  return response.data;
};