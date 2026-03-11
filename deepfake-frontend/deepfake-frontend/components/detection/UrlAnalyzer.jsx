import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { 
  FiLink,
  FiSearch,
  FiDownload,
  FiRefreshCw,
  FiShield,
  FiAlertTriangle,
  FiCheckCircle,
  FiGlobe,
  FiClock,
  FiServer,
  FiLock,
  FiExternalLink
} from 'react-icons/fi';
import Button from '@/components/common/Button';
import Loader from '@/components/common/Loader';
import { useAlert } from '@/components/common/Alert';
import { analyzeUrl } from '@/utils/api';

/**
 * UrlAnalyzer Component - Analyze URLs for phishing, malware, and deepfakes
 */

const UrlAnalyzer = () => {
  const [url, setUrl] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [activeTab, setActiveTab] = useState('input');
  const [options, setOptions] = useState({
    extractContent: true,
    checkSsl: true,
    whoisLookup: true,
    analyzeImages: false,
  });

  const { success, error, info } = useAlert();

  // URL validation
  const isValidUrl = (string) => {
    try {
      new URL(string);
      return true;
    } catch (_) {
      return false;
    }
  };

  // Handle URL change
  const handleUrlChange = (e) => {
    setUrl(e.target.value);
    setResult(null);
    setActiveTab('input');
  };

  // Handle analysis
  const handleAnalyze = async () => {
    if (!url.trim()) {
      error('No URL entered', 'Please enter a URL to analyze');
      return;
    }

    // Add protocol if missing
    let urlToAnalyze = url;
    if (!urlToAnalyze.startsWith('http://') && !urlToAnalyze.startsWith('https://')) {
      urlToAnalyze = 'https://' + urlToAnalyze;
    }

    if (!isValidUrl(urlToAnalyze)) {
      error('Invalid URL', 'Please enter a valid URL');
      return;
    }

    setLoading(true);
    setActiveTab('results');

    try {
      const response = await analyzeUrl(urlToAnalyze, options);
      setResult(response.result);
      
      const riskLevel = response.result.threat_assessment.risk_level;
      success('Analysis complete', 
        `Risk level: ${riskLevel} - ${response.result.threat_assessment.risk_score}/100`
      );
    } catch (err) {
      error('Analysis failed', err.response?.data?.detail || 'Please try again');
    } finally {
      setLoading(false);
    }
  };

  // Reset
  const handleReset = () => {
    setUrl('');
    setResult(null);
    setActiveTab('input');
  };

  // Download report
  const handleDownload = () => {
    const report = {
      url: url,
      timestamp: new Date().toISOString(),
      result: result,
      options: options,
    };

    const blob = new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `url-report-${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  // Open URL in new tab
  const openUrl = () => {
    let urlToOpen = url;
    if (!urlToOpen.startsWith('http://') && !urlToOpen.startsWith('https://')) {
      urlToOpen = 'https://' + urlToOpen;
    }
    window.open(urlToOpen, '_blank');
  };

  // Get risk color
  const getRiskColor = (level) => {
    switch (level) {
      case 'CRITICAL': return 'text-red-400 bg-red-500/20';
      case 'HIGH': return 'text-orange-400 bg-orange-500/20';
      case 'MEDIUM': return 'text-yellow-400 bg-yellow-500/20';
      case 'LOW': return 'text-green-400 bg-green-500/20';
      default: return 'text-blue-400 bg-blue-500/20';
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-white">URL Security Analyzer</h2>
          <p className="text-gray-400 mt-1">
            Check URLs for phishing, malware, scams, and deepfake content
          </p>
        </div>
        <div className="flex items-center gap-2 text-sm text-gray-400">
          <FiShield className="w-4 h-4" />
          <span>97.8% Accuracy</span>
        </div>
      </div>

      {/* Main content */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left column - Input/Options */}
        <div className="lg:col-span-1 space-y-6">
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            className="glass-card p-6"
          >
            <h3 className="text-white font-semibold mb-4">Enter URL</h3>
            
            <div className="space-y-4">
              <div className="relative">
                <FiLink className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-5 h-5" />
                <input
                  type="url"
                  value={url}
                  onChange={handleUrlChange}
                  placeholder="https://example.com"
                  className="input-field pl-10"
                />
              </div>

              {/* Quick examples */}
              <div className="flex flex-wrap gap-2">
                <button
                  onClick={() => setUrl('google.com')}
                  className="px-2 py-1 text-xs glass rounded-lg hover:bg-white/10 text-gray-300"
                >
                  google.com
                </button>
                <button
                  onClick={() => setUrl('github.com')}
                  className="px-2 py-1 text-xs glass rounded-lg hover:bg-white/10 text-gray-300"
                >
                  github.com
                </button>
                <button
                  onClick={() => setUrl('example.com')}
                  className="px-2 py-1 text-xs glass rounded-lg hover:bg-white/10 text-gray-300"
                >
                  example.com
                </button>
              </div>
            </div>

            {/* Options */}
            <div className="space-y-4 mt-6">
              <h4 className="text-white text-sm font-medium">Analysis Options</h4>
              
              <label className="flex items-center gap-3 p-3 glass rounded-xl cursor-pointer">
                <input
                  type="checkbox"
                  checked={options.extractContent}
                  onChange={(e) => setOptions({...options, extractContent: e.target.checked})}
                  className="w-4 h-4 rounded border-white/10 bg-white/5 text-primary-500"
                />
                <div>
                  <p className="text-white text-sm font-medium">Extract Page Content</p>
                  <p className="text-xs text-gray-400">Analyze text and metadata</p>
                </div>
              </label>

              <label className="flex items-center gap-3 p-3 glass rounded-xl cursor-pointer">
                <input
                  type="checkbox"
                  checked={options.checkSsl}
                  onChange={(e) => setOptions({...options, checkSsl: e.target.checked})}
                  className="w-4 h-4 rounded border-white/10 bg-white/5 text-primary-500"
                />
                <div>
                  <p className="text-white text-sm font-medium">Check SSL Certificate</p>
                  <p className="text-xs text-gray-400">Validate security</p>
                </div>
              </label>

              <label className="flex items-center gap-3 p-3 glass rounded-xl cursor-pointer">
                <input
                  type="checkbox"
                  checked={options.whoisLookup}
                  onChange={(e) => setOptions({...options, whoisLookup: e.target.checked})}
                  className="w-4 h-4 rounded border-white/10 bg-white/5 text-primary-500"
                />
                <div>
                  <p className="text-white text-sm font-medium">WHOIS Lookup</p>
                  <p className="text-xs text-gray-400">Check domain registration</p>
                </div>
              </label>

              <label className="flex items-center gap-3 p-3 glass rounded-xl cursor-pointer">
                <input
                  type="checkbox"
                  checked={options.analyzeImages}
                  onChange={(e) => setOptions({...options, analyzeImages: e.target.checked})}
                  className="w-4 h-4 rounded border-white/10 bg-white/5 text-primary-500"
                />
                <div>
                  <p className="text-white text-sm font-medium">Analyze Images</p>
                  <p className="text-xs text-gray-400">Detect deepfakes in images</p>
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
                disabled={!url || loading}
                loading={loading}
                icon={<FiSearch className="w-5 h-5" />}
              >
                Analyze
              </Button>
              
              {url && (
                <Button
                  variant="ghost"
                  size="lg"
                  onClick={handleReset}
                  icon={<FiRefreshCw className="w-5 h-5" />}
                />
              )}
            </div>
          </motion.div>

          {/* Quick stats */}
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.1 }}
            className="glass-card p-6"
          >
            <h3 className="text-white font-semibold mb-4">Why Check URLs?</h3>
            <ul className="space-y-3 text-sm">
              <li className="flex items-center gap-3 text-gray-300">
                <span className="w-1.5 h-1.5 bg-red-400 rounded-full" />
                1.5M+ phishing sites detected daily
              </li>
              <li className="flex items-center gap-3 text-gray-300">
                <span className="w-1.5 h-1.5 bg-red-400 rounded-full" />
                75% of malicious URLs use HTTPS
              </li>
              <li className="flex items-center gap-3 text-gray-300">
                <span className="w-1.5 h-1.5 bg-red-400 rounded-full" />
                New phishing site every 20 seconds
              </li>
            </ul>
          </motion.div>
        </div>

        {/* Right column - Results */}
        <div className="lg:col-span-2">
          {activeTab === 'input' && !result && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="glass-card p-12 text-center"
            >
              <FiGlobe className="w-16 h-16 text-gray-600 mx-auto mb-4" />
              <h3 className="text-xl text-white font-medium mb-2">Enter a URL to Analyze</h3>
              <p className="text-gray-400">
                Get comprehensive security analysis including phishing detection, SSL validation, and content scanning
              </p>
            </motion.div>
          )}

          {loading && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="glass-card p-12 text-center"
            >
              <Loader text="Analyzing URL..." />
              <p className="text-gray-400 mt-4">Checking domain, SSL, and content...</p>
            </motion.div>
          )}

          {result && !loading && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="space-y-6"
            >
              {/* Risk Score Card */}
              <div className="glass-card p-6">
                <div className="flex items-center justify-between mb-6">
                  <h3 className="text-white font-semibold">Security Assessment</h3>
                  <div className="flex gap-2">
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={openUrl}
                      icon={<FiExternalLink className="w-4 h-4" />}
                    >
                      Visit
                    </Button>
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={handleDownload}
                      icon={<FiDownload className="w-4 h-4" />}
                    >
                      Report
                    </Button>
                  </div>
                </div>

                {/* Risk Score */}
                <div className="text-center mb-6">
                  <div className="inline-block p-8 glass rounded-full mb-4">
                    <span className={`text-4xl font-bold ${
                      result.threat_assessment.risk_level === 'CRITICAL' ? 'text-red-400' :
                      result.threat_assessment.risk_level === 'HIGH' ? 'text-orange-400' :
                      result.threat_assessment.risk_level === 'MEDIUM' ? 'text-yellow-400' :
                      'text-green-400'
                    }`}>
                      {result.threat_assessment.risk_score}
                    </span>
                  </div>
                  <h4 className={`text-2xl font-bold mb-2 ${
                    result.threat_assessment.risk_level === 'CRITICAL' ? 'text-red-400' :
                    result.threat_assessment.risk_level === 'HIGH' ? 'text-orange-400' :
                    result.threat_assessment.risk_level === 'MEDIUM' ? 'text-yellow-400' :
                    'text-green-400'
                  }`}>
                    {result.threat_assessment.risk_level} RISK
                  </h4>
                  <p className="text-gray-400">
                    {result.url_info.full_domain}
                  </p>
                </div>

                {/* Threat Types */}
                {result.threat_assessment.threat_types?.length > 0 && (
                  <div className="flex flex-wrap gap-2 justify-center mb-6">
                    {result.threat_assessment.threat_types.map((threat, idx) => (
                      <span
                        key={idx}
                        className="px-3 py-1 bg-red-500/20 text-red-400 rounded-full text-xs"
                      >
                        {threat}
                      </span>
                    ))}
                  </div>
                )}

                {/* Quick Stats */}
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  <div className="text-center p-3 glass rounded-xl">
                    <FiGlobe className="w-5 h-5 text-primary-400 mx-auto mb-2" />
                    <p className="text-xs text-gray-400">Domain Age</p>
                    <p className="text-white font-medium">
                      {result.whois_info?.days_old ? `${result.whois_info.days_old} days` : 'Unknown'}
                    </p>
                  </div>
                  <div className="text-center p-3 glass rounded-xl">
                    <FiLock className="w-5 h-5 text-primary-400 mx-auto mb-2" />
                    <p className="text-xs text-gray-400">SSL</p>
                    <p className="text-white font-medium">
                      {result.ssl_info?.is_valid ? 'Valid' : 'Invalid'}
                    </p>
                  </div>
                  <div className="text-center p-3 glass rounded-xl">
                    <FiServer className="w-5 h-5 text-primary-400 mx-auto mb-2" />
                    <p className="text-xs text-gray-400">Status</p>
                    <p className="text-white font-medium">{result.status_code}</p>
                  </div>
                  <div className="text-center p-3 glass rounded-xl">
                    <FiClock className="w-5 h-5 text-primary-400 mx-auto mb-2" />
                    <p className="text-xs text-gray-400">Response</p>
                    <p className="text-white font-medium">{(result.response_time * 1000).toFixed(0)}ms</p>
                  </div>
                </div>
              </div>

              {/* Warnings */}
              {result.threat_assessment.warnings?.length > 0 && (
                <div className="glass-card p-6">
                  <h3 className="text-white font-semibold mb-4">⚠️ Warnings</h3>
                  <ul className="space-y-3">
                    {result.threat_assessment.warnings.map((warning, idx) => (
                      <li key={idx} className="flex items-start gap-3 text-sm">
                        <FiAlertTriangle className="w-5 h-5 text-yellow-400 flex-shrink-0 mt-0.5" />
                        <span className="text-gray-300">{warning}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              )}

              {/* URL Details */}
              <div className="glass-card p-6">
                <h3 className="text-white font-semibold mb-4">URL Details</h3>
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <p className="text-gray-400 text-xs mb-1">Domain</p>
                    <p className="text-white text-sm">{result.url_info.full_domain}</p>
                  </div>
                  <div>
                    <p className="text-gray-400 text-xs mb-1">TLD</p>
                    <p className="text-white text-sm">.{result.url_info.tld}</p>
                  </div>
                  <div>
                    <p className="text-gray-400 text-xs mb-1">Scheme</p>
                    <p className="text-white text-sm">{result.url_info.scheme}</p>
                  </div>
                  <div>
                    <p className="text-gray-400 text-xs mb-1">Path Length</p>
                    <p className="text-white text-sm">{result.url_info.path.length}</p>
                  </div>
                  <div>
                    <p className="text-gray-400 text-xs mb-1">Parameters</p>
                    <p className="text-white text-sm">{result.url_info.params ? Object.keys(result.url_info.params).length : 0}</p>
                  </div>
                  <div>
                    <p className="text-gray-400 text-xs mb-1">Subdomains</p>
                    <p className="text-white text-sm">{result.url_info.subdomain || 'None'}</p>
                  </div>
                </div>
              </div>

              {/* WHOIS Info */}
              {result.whois_info && (
                <div className="glass-card p-6">
                  <h3 className="text-white font-semibold mb-4">WHOIS Information</h3>
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <p className="text-gray-400 text-xs mb-1">Registrar</p>
                      <p className="text-white text-sm">{result.whois_info.registrar || 'Unknown'}</p>
                    </div>
                    <div>
                      <p className="text-gray-400 text-xs mb-1">Created</p>
                      <p className="text-white text-sm">
                        {result.whois_info.creation_date ? new Date(result.whois_info.creation_date).toLocaleDateString() : 'Unknown'}
                      </p>
                    </div>
                    <div>
                      <p className="text-gray-400 text-xs mb-1">Expires</p>
                      <p className="text-white text-sm">
                        {result.whois_info.expiration_date ? new Date(result.whois_info.expiration_date).toLocaleDateString() : 'Unknown'}
                      </p>
                    </div>
                    <div>
                      <p className="text-gray-400 text-xs mb-1">Name Servers</p>
                      <p className="text-white text-sm">{result.whois_info.name_servers?.length || 0}</p>
                    </div>
                  </div>
                </div>
              )}

              {/* SSL Info */}
              {result.ssl_info && (
                <div className="glass-card p-6">
                  <h3 className="text-white font-semibold mb-4">SSL Certificate</h3>
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <p className="text-gray-400 text-xs mb-1">Issuer</p>
                      <p className="text-white text-sm">{result.ssl_info.issuer?.CN || 'Unknown'}</p>
                    </div>
                    <div>
                      <p className="text-gray-400 text-xs mb-1">Valid From</p>
                      <p className="text-white text-sm">
                        {new Date(result.ssl_info.not_before).toLocaleDateString()}
                      </p>
                    </div>
                    <div>
                      <p className="text-gray-400 text-xs mb-1">Valid Until</p>
                      <p className="text-white text-sm">
                        {new Date(result.ssl_info.not_after).toLocaleDateString()}
                      </p>
                    </div>
                    <div>
                      <p className="text-gray-400 text-xs mb-1">Days Left</p>
                      <p className="text-white text-sm">{result.ssl_info.days_until_expiry}</p>
                    </div>
                  </div>
                </div>
              )}
            </motion.div>
          )}
        </div>
      </div>
    </div>
  );
};

export default UrlAnalyzer;