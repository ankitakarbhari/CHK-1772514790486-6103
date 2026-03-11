import { useState, useCallback } from 'react';
import * as api from '@/utils/api';
import { parseApiError } from '@/utils/helpers';
import { DETECTION_RESULTS } from '@/utils/constants';

const useDetection = () => {
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [progress, setProgress] = useState(0);

  // Image detection
  const detectImage = useCallback(async (file, options = {}) => {
    setLoading(true);
    setError(null);
    setProgress(0);
    
    try {
      // Simulate progress
      const interval = setInterval(() => {
        setProgress(prev => Math.min(prev + 10, 90));
      }, 200);

      const response = await api.detectImage(file, options);
      
      clearInterval(interval);
      setProgress(100);
      setResult(response);
      
      return response;
    } catch (err) {
      setError(parseApiError(err));
      throw err;
    } finally {
      setLoading(false);
    }
  }, []);

  // Video detection
  const detectVideo = useCallback(async (file, options = {}) => {
    setLoading(true);
    setError(null);
    setProgress(0);
    
    try {
      const interval = setInterval(() => {
        setProgress(prev => Math.min(prev + 5, 90));
      }, 500);

      const response = await api.detectVideo(file, options);
      
      clearInterval(interval);
      setProgress(100);
      setResult(response);
      
      return response;
    } catch (err) {
      setError(parseApiError(err));
      throw err;
    } finally {
      setLoading(false);
    }
  }, []);

  // Audio detection
  const detectAudio = useCallback(async (file) => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await api.detectAudio(file);
      setResult(response);
      return response;
    } catch (err) {
      setError(parseApiError(err));
      throw err;
    } finally {
      setLoading(false);
    }
  }, []);

  // Text detection
  const detectText = useCallback(async (text, options = {}) => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await api.detectText(text, options);
      setResult(response);
      return response;
    } catch (err) {
      setError(parseApiError(err));
      throw err;
    } finally {
      setLoading(false);
    }
  }, []);

  // URL analysis
  const analyzeUrl = useCallback(async (url, options = {}) => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await api.analyzeUrl(url, options);
      setResult(response);
      return response;
    } catch (err) {
      setError(parseApiError(err));
      throw err;
    } finally {
      setLoading(false);
    }
  }, []);

  // Reset state
  const reset = useCallback(() => {
    setResult(null);
    setError(null);
    setProgress(0);
  }, []);

  // Helper to check if result is fake
  const isFake = result?.result?.prediction === DETECTION_RESULTS.FAKE;
  
  // Get confidence
  const confidence = result?.result?.confidence || 0;

  return {
    loading,
    result,
    error,
    progress,
    isFake,
    confidence,
    detectImage,
    detectVideo,
    detectAudio,
    detectText,
    analyzeUrl,
    reset,
  };
};

export default useDetection;