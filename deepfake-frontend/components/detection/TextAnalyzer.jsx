import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { 
  FiFileText,
  FiSearch,
  FiDownload,
  FiRefreshCw,
  FiShield,
  FiAlertTriangle,
  FiCheckCircle,
  FiBarChart2,
  FiCopy,
  FiTrash2
} from 'react-icons/fi';
import Button from '@/components/common/Button';
import Loader from '@/components/common/Loader';
import { useAlert } from '@/components/common/Alert';
import { detectText } from '@/utils/api';

/**
 * TextAnalyzer Component - Analyze text for AI-generated content
 */

const TextAnalyzer = () => {
  const [text, setText] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [stats, setStats] = useState({
    words: 0,
    chars: 0,
    sentences: 0,
  });

  const { success, error, info } = useAlert();

  // Update statistics
  const updateStats = (input) => {
    const words = input.trim() ? input.trim().split(/\s+/).length : 0;
    const chars = input.length;
    const sentences = input.split(/[.!?]+/).filter(Boolean).length;

    setStats({ words, chars, sentences });
  };

  // Handle text change
  const handleTextChange = (e) => {
    const newText = e.target.value;
    setText(newText);
    updateStats(newText);
    setResult(null);
  };

  // Handle analysis
  const handleAnalyze = async () => {
    if (!text.trim()) {
      error('No text entered', 'Please enter some text to analyze');
      return;
    }

    if (text.length < 20) {
      error('Text too short', 'Please enter at least 20 characters');
      return;
    }

    setLoading(true);

    try {
      const response = await detectText(text, { returnDetails: true });
      setResult(response.result);
      
      success('Analysis complete', 
        `Text is ${response.result.prediction === 'AI' ? 'AI-GENERATED' : 'HUMAN-WRITTEN'}`
      );
    } catch (err) {
      error('Analysis failed', err.response?.data?.detail || 'Please try again');
    } finally {
      setLoading(false);
    }
  };

  // Reset text
  const handleReset = () => {
    setText('');
    setResult(null);
    setStats({ words: 0, chars: 0, sentences: 0 });
  };

  // Copy to clipboard
  const handleCopy = () => {
    navigator.clipboard.writeText(text);
    info('Copied!', 'Text copied to clipboard');
  };

  // Download report
  const handleDownload = () => {
    const report = {
      text: text,
      timestamp: new Date().toISOString(),
      stats: stats,
      result: result,
    };

    const blob = new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `text-report-${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-white">AI Text Detection</h2>
          <p className="text-gray-400 mt-1">
            Analyze text to detect if it was written by AI (ChatGPT, Claude, etc.)
          </p>
        </div>
        <div className="flex items-center gap-2 text-sm text-gray-400">
          <FiShield className="w-4 h-4" />
          <span>95.2% Accuracy</span>
        </div>
      </div>

      {/* Main content */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left column - Input */}
        <div className="lg:col-span-2 space-y-6">
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            className="glass-card p-6"
          >
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-white font-semibold">Enter Text</h3>
              <div className="flex gap-2">
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={handleCopy}
                  disabled={!text}
                  icon={<FiCopy className="w-4 h-4" />}
                />
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={handleReset}
                  disabled={!text}
                  icon={<FiTrash2 className="w-4 h-4" />}
                />
              </div>
            </div>

            <textarea
              value={text}
              onChange={handleTextChange}
              placeholder="Paste or type text here to analyze..."
              className="input-field h-64 resize-none font-mono text-sm"
            />

            {/* Stats bar */}
            <div className="flex items-center justify-between mt-4 text-sm">
              <div className="flex gap-4">
                <span className="text-gray-400">
                  Words: <span className="text-white font-medium">{stats.words}</span>
                </span>
                <span className="text-gray-400">
                  Characters: <span className="text-white font-medium">{stats.chars}</span>
                </span>
                <span className="text-gray-400">
                  Sentences: <span className="text-white font-medium">{stats.sentences}</span>
                </span>
              </div>
              <Button
                variant="primary"
                size="sm"
                onClick={handleAnalyze}
                disabled={!text || loading}
                loading={loading}
                icon={<FiSearch className="w-4 h-4" />}
              >
                Analyze
              </Button>
            </div>
          </motion.div>

          {/* Results */}
          {result && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="glass-card p-6"
            >
              <div className="flex items-center justify-between mb-6">
                <h3 className="text-white font-semibold">Analysis Results</h3>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={handleDownload}
                  icon={<FiDownload className="w-4 h-4" />}
                >
                  Report
                </Button>
              </div>

              {/* Result badge */}
              <div className="text-center mb-6">
                <div className={`
                  w-24 h-24 mx-auto rounded-full flex items-center justify-center mb-4
                  ${result.prediction === 'AI' 
                    ? 'bg-red-500/20 text-red-400' 
                    : 'bg-green-500/20 text-green-400'}
                `}>
                  {result.prediction === 'AI' ? (
                    <FiAlertTriangle className="w-12 h-12" />
                  ) : (
                    <FiCheckCircle className="w-12 h-12" />
                  )}
                </div>
                <h4 className={`text-3xl font-bold mb-2 ${
                  result.prediction === 'AI' ? 'text-red-400' : 'text-green-400'
                }`}>
                  {result.prediction === 'AI' ? 'AI-GENERATED' : 'HUMAN-WRITTEN'}
                </h4>
                <p className="text-gray-400">
                  Confidence: {(result.confidence * 100).toFixed(1)}%
                </p>
              </div>

              {/* Probabilities */}
              <div className="space-y-4 mb-6">
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

              {/* Metrics */}
              {result.metrics && (
                <div className="glass p-4 rounded-xl">
                  <h4 className="text-white font-medium mb-3">Text Metrics</h4>
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <p className="text-gray-400 text-xs">Perplexity</p>
                      <p className="text-white font-medium">{result.metrics.perplexity.toFixed(2)}</p>
                    </div>
                    <div>
                      <p className="text-gray-400 text-xs">Burstiness</p>
                      <p className="text-white font-medium">{result.metrics.burstiness.toFixed(2)}</p>
                    </div>
                    <div>
                      <p className="text-gray-400 text-xs">Vocabulary Richness</p>
                      <p className="text-white font-medium">{result.metrics.vocabulary_richness.toFixed(2)}</p>
                    </div>
                    <div>
                      <p className="text-gray-400 text-xs">Readability</p>
                      <p className="text-white font-medium">{result.metrics.readability_score.toFixed(2)}</p>
                    </div>
                  </div>
                </div>
              )}

              {/* Suspicious sentences */}
              {result.suspicious_sentences?.length > 0 && (
                <div className="mt-6">
                  <h4 className="text-white font-medium mb-3">Suspicious Segments</h4>
                  <div className="space-y-3">
                    {result.suspicious_sentences.map((sent, idx) => (
                      <div key={idx} className="p-3 glass rounded-xl">
                        <p className="text-gray-300 text-sm mb-2">"{sent.text}"</p>
                        <div className="flex items-center justify-between text-xs">
                          <span className="text-yellow-400">Score: {(sent.score * 100).toFixed(1)}%</span>
                          <span className="text-gray-400">
                            {sent.reasons?.join(', ')}
                          </span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </motion.div>
          )}
        </div>

        {/* Right column - Info */}
        <div className="lg:col-span-1 space-y-6">
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            className="glass-card p-6"
          >
            <h3 className="text-white font-semibold mb-4">Detection Methods</h3>
            <ul className="space-y-4">
              <li className="flex items-start gap-3">
                <div className="w-6 h-6 rounded-lg bg-primary-500/20 flex items-center justify-center flex-shrink-0 mt-0.5">
                  <span className="text-primary-400 text-sm">1</span>
                </div>
                <div>
                  <p className="text-white text-sm font-medium">Perplexity Analysis</p>
                  <p className="text-xs text-gray-400">Measures text randomness</p>
                </div>
              </li>
              <li className="flex items-start gap-3">
                <div className="w-6 h-6 rounded-lg bg-primary-500/20 flex items-center justify-center flex-shrink-0 mt-0.5">
                  <span className="text-primary-400 text-sm">2</span>
                </div>
                <div>
                  <p className="text-white text-sm font-medium">Burstiness Detection</p>
                  <p className="text-xs text-gray-400">Analyzes sentence variation</p>
                </div>
              </li>
              <li className="flex items-start gap-3">
                <div className="w-6 h-6 rounded-lg bg-primary-500/20 flex items-center justify-center flex-shrink-0 mt-0.5">
                  <span className="text-primary-400 text-sm">3</span>
                </div>
                <div>
                  <p className="text-white text-sm font-medium">BERT Classifier</p>
                  <p className="text-xs text-gray-400">Fine-tuned AI detector</p>
                </div>
              </li>
              <li className="flex items-start gap-3">
                <div className="w-6 h-6 rounded-lg bg-primary-500/20 flex items-center justify-center flex-shrink-0 mt-0.5">
                  <span className="text-primary-400 text-sm">4</span>
                </div>
                <div>
                  <p className="text-white text-sm font-medium">Statistical Analysis</p>
                  <p className="text-xs text-gray-400">20+ linguistic features</p>
                </div>
              </li>
            </ul>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.1 }}
            className="glass-card p-6"
          >
            <h3 className="text-white font-semibold mb-4">Tips</h3>
            <ul className="space-y-2 text-sm text-gray-300">
              <li>• Enter at least 50 words for best results</li>
              <li>• AI text tends to be more predictable</li>
              <li>• Human writing has more variation</li>
              <li>• Check multiple paragraphs for accuracy</li>
            </ul>
          </motion.div>
        </div>
      </div>
    </div>
  );
};

export default TextAnalyzer;