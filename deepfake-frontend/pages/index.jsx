import React from 'react';
import Link from 'next/link';
import { motion } from 'framer-motion';
import { 
  FiShield, 
  FiCpu, 
  FiGlobe, 
  FiUsers,
  FiArrowRight,
  FiGithub,
  FiTwitter,
  FiLinkedin
} from 'react-icons/fi';

export default function Home() {
  const features = [
    {
      icon: <FiCpu className="w-8 h-8" />,
      title: 'AI-Generated Images',
      description: 'Detect GAN-generated and AI-manipulated images with 98.3% accuracy',
      color: 'blue'
    },
    {
      icon: <FiGlobe className="w-8 h-8" />,
      title: 'Video Deepfakes',
      description: 'Identify face-swapped videos and manipulated content in real-time',
      color: 'purple'
    },
    {
      icon: <FiUsers className="w-8 h-8" />,
      title: 'Voice Synthesis',
      description: 'Spot AI-generated voices and audio manipulations',
      color: 'green'
    },
    {
      icon: <FiShield className="w-8 h-8" />,
      title: 'Text Analysis',
      description: 'Detect ChatGPT and other AI-generated text content',
      color: 'red'
    }
  ];

  const stats = [
    { value: '98.3%', label: 'Accuracy' },
    { value: '<1s', label: 'Response Time' },
    { value: '10+', label: 'Media Types' },
    { value: '1M+', label: 'Scans' }
  ];

  return (
    <div className="min-h-screen">
      {/* Hero Section */}
      <section className="relative min-h-screen flex items-center justify-center overflow-hidden">
        {/* Background */}
        <div className="absolute inset-0 bg-gradient-to-br from-primary-600/20 via-purple-600/20 to-pink-600/20" />
        <div className="absolute inset-0 bg-[url('/grid.svg')] opacity-20" />
        <div className="particles" />
        
        {/* Animated orbs */}
        <div className="absolute top-20 left-20 w-72 h-72 bg-primary-500/30 rounded-full blur-3xl animate-pulse" />
        <div className="absolute bottom-20 right-20 w-96 h-96 bg-purple-500/30 rounded-full blur-3xl animate-pulse delay-1000" />

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
          className="relative z-10 text-center max-w-5xl mx-auto px-4"
        >
          {/* Logo/Icon */}
          <motion.div
            initial={{ scale: 0 }}
            animate={{ scale: 1 }}
            transition={{ delay: 0.2, type: 'spring', stiffness: 200 }}
            className="w-28 h-28 mx-auto mb-8 bg-gradient-to-r from-primary-600 to-primary-400 
                       rounded-3xl flex items-center justify-center shadow-2xl shadow-primary-500/30"
          >
            <FiShield className="w-14 h-14 text-white" />
          </motion.div>

          {/* Title */}
          <h1 className="text-6xl md:text-8xl font-bold text-white mb-6">
            Deep<span className="text-primary-400">Shield</span>
          </h1>
          
          <p className="text-xl md:text-2xl text-gray-300 mb-8 max-w-3xl mx-auto leading-relaxed">
            Advanced AI-powered deepfake detection system that protects you from 
            manipulated media, AI-generated content, and digital deception.
          </p>

          {/* CTA Buttons */}
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <Link href="/dashboard">
              <motion.button
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                className="group relative px-8 py-4 bg-gradient-to-r from-primary-600 to-primary-500 
                           text-white font-semibold rounded-xl shadow-lg shadow-primary-500/25 
                           hover:shadow-xl transition-all duration-300 text-lg inline-flex items-center gap-2"
              >
                Get Started
                <FiArrowRight className="w-5 h-5 group-hover:translate-x-1 transition-transform" />
              </motion.button>
            </Link>
            <Link href="/image">
              <motion.button
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                className="px-8 py-4 glass text-white font-semibold rounded-xl 
                           hover:bg-white/20 transition-all duration-300 text-lg border border-white/10"
              >
                Try Demo
              </motion.button>
            </Link>
          </div>

          {/* Stats */}
          <motion.div 
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.4 }}
            className="grid grid-cols-2 md:grid-cols-4 gap-8 mt-16"
          >
            {stats.map((stat, index) => (
              <div key={index} className="text-center">
                <div className="text-3xl md:text-4xl font-bold text-white mb-2">{stat.value}</div>
                <div className="text-sm text-gray-400">{stat.label}</div>
              </div>
            ))}
          </motion.div>
        </motion.div>

        {/* Scroll indicator */}
        <motion.div 
          animate={{ y: [0, 10, 0] }}
          transition={{ duration: 1.5, repeat: Infinity }}
          className="absolute bottom-10 left-1/2 transform -translate-x-1/2"
        >
          <div className="w-6 h-10 border-2 border-white/30 rounded-full flex justify-center">
            <div className="w-1 h-3 bg-white/50 rounded-full mt-2" />
          </div>
        </motion.div>
      </section>

      {/* Features Section */}
      <section className="py-20 bg-dark-400/50">
        <div className="container mx-auto px-4">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="text-center mb-16"
          >
            <h2 className="text-4xl md:text-5xl font-bold text-white mb-4">
              Advanced Detection <span className="text-primary-400">Capabilities</span>
            </h2>
            <p className="text-xl text-gray-400 max-w-3xl mx-auto">
              Our AI models are trained on millions of samples to detect various types of deepfakes
            </p>
          </motion.div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8">
            {features.map((feature, index) => {
              const colors = {
                blue: 'from-blue-600 to-blue-400',
                purple: 'from-purple-600 to-purple-400',
                green: 'from-green-600 to-green-400',
                red: 'from-red-600 to-red-400'
              };

              return (
                <motion.div
                  key={index}
                  initial={{ opacity: 0, y: 20 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  viewport={{ once: true }}
                  transition={{ delay: index * 0.1 }}
                  whileHover={{ y: -10 }}
                  className="glass-card p-8 text-center group"
                >
                  <div className={`w-20 h-20 mx-auto mb-6 bg-gradient-to-r ${colors[feature.color]} 
                                  rounded-2xl flex items-center justify-center text-white
                                  group-hover:scale-110 transition-transform duration-300`}>
                    {feature.icon}
                  </div>
                  <h3 className="text-xl font-semibold text-white mb-3">{feature.title}</h3>
                  <p className="text-gray-400 leading-relaxed">{feature.description}</p>
                </motion.div>
              );
            })}
          </div>
        </div>
      </section>

      {/* How It Works Section */}
      <section className="py-20">
        <div className="container mx-auto px-4">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="text-center mb-16"
          >
            <h2 className="text-4xl md:text-5xl font-bold text-white mb-4">
              How It <span className="text-primary-400">Works</span>
            </h2>
            <p className="text-xl text-gray-400 max-w-3xl mx-auto">
              Simple three-step process to verify any media
            </p>
          </motion.div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            {[
              {
                step: '01',
                title: 'Upload Media',
                description: 'Upload your image, video, audio, or text for analysis',
                icon: '📤'
              },
              {
                step: '02',
                title: 'AI Analysis',
                description: 'Our ensemble models analyze for signs of manipulation',
                icon: '🤖'
              },
              {
                step: '03',
                title: 'Get Results',
                description: 'Receive detailed report with confidence scores',
                icon: '📊'
              }
            ].map((item, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, x: index === 1 ? 0 : index === 0 ? -50 : 50 }}
                whileInView={{ opacity: 1, x: 0 }}
                viewport={{ once: true }}
                transition={{ delay: index * 0.2 }}
                className="relative text-center"
              >
                <div className="text-8xl mb-4 opacity-20">{item.step}</div>
                <div className="text-5xl mb-4">{item.icon}</div>
                <h3 className="text-2xl font-semibold text-white mb-3">{item.title}</h3>
                <p className="text-gray-400">{item.description}</p>
                
                {index < 2 && (
                  <div className="hidden md:block absolute top-1/3 -right-4 text-2xl text-gray-600">
                    →
                  </div>
                )}
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-20 bg-gradient-to-r from-primary-600/20 via-purple-600/20 to-pink-600/20">
        <div className="container mx-auto px-4 text-center">
          <motion.h2
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="text-4xl md:text-5xl font-bold text-white mb-6"
          >
            Ready to secure your <span className="text-primary-400">digital world</span>?
          </motion.h2>
          
          <motion.p
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ delay: 0.1 }}
            className="text-xl text-gray-300 mb-8 max-w-2xl mx-auto"
          >
            Join thousands of users who trust DeepShield AI for accurate deepfake detection
          </motion.p>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ delay: 0.2 }}
          >
            <Link href="/dashboard">
              <button className="group px-10 py-5 bg-gradient-to-r from-primary-600 to-primary-500 
                                 text-white font-semibold rounded-xl shadow-lg shadow-primary-500/25 
                                 hover:shadow-xl transition-all duration-300 text-lg inline-flex items-center gap-3">
                Start Detecting Now
                <FiArrowRight className="w-5 h-5 group-hover:translate-x-1 transition-transform" />
              </button>
            </Link>
          </motion.div>
        </div>
      </section>

      {/* Footer */}
      <footer className="bg-dark-500 py-12 border-t border-white/10">
        <div className="container mx-auto px-4">
          <div className="grid grid-cols-1 md:grid-cols-4 gap-8">
            <div>
              <div className="flex items-center gap-2 mb-4">
                <div className="w-10 h-10 bg-gradient-to-r from-primary-600 to-primary-400 rounded-xl flex items-center justify-center">
                  <FiShield className="w-5 h-5 text-white" />
                </div>
                <span className="text-xl font-bold text-white">DeepShield</span>
              </div>
              <p className="text-gray-400 text-sm">
                Advanced AI-powered deepfake detection for a secure digital future.
              </p>
            </div>
            
            <div>
              <h4 className="text-white font-semibold mb-4">Product</h4>
              <ul className="space-y-2 text-gray-400">
                <li><Link href="/features" className="hover:text-primary-400 transition">Features</Link></li>
                <li><Link href="/pricing" className="hover:text-primary-400 transition">Pricing</Link></li>
                <li><Link href="/api" className="hover:text-primary-400 transition">API</Link></li>
              </ul>
            </div>
            
            <div>
              <h4 className="text-white font-semibold mb-4">Resources</h4>
              <ul className="space-y-2 text-gray-400">
                <li><Link href="/docs" className="hover:text-primary-400 transition">Documentation</Link></li>
                <li><Link href="/blog" className="hover:text-primary-400 transition">Blog</Link></li>
                <li><Link href="/support" className="hover:text-primary-400 transition">Support</Link></li>
              </ul>
            </div>
            
            <div>
              <h4 className="text-white font-semibold mb-4">Connect</h4>
              <div className="flex gap-4">
                <a href="#" className="w-10 h-10 glass rounded-xl flex items-center justify-center text-gray-400 hover:text-white hover:bg-white/10 transition">
                  <FiGithub className="w-5 h-5" />
                </a>
                <a href="#" className="w-10 h-10 glass rounded-xl flex items-center justify-center text-gray-400 hover:text-white hover:bg-white/10 transition">
                  <FiTwitter className="w-5 h-5" />
                </a>
                <a href="#" className="w-10 h-10 glass rounded-xl flex items-center justify-center text-gray-400 hover:text-white hover:bg-white/10 transition">
                  <FiLinkedin className="w-5 h-5" />
                </a>
              </div>
            </div>
          </div>
          
          <div className="mt-12 pt-8 border-t border-white/10 text-center text-gray-400 text-sm">
            <p>&copy; {new Date().getFullYear()} DeepShield AI. All rights reserved.</p>
          </div>
        </div>
      </footer>
    </div>
  );
}