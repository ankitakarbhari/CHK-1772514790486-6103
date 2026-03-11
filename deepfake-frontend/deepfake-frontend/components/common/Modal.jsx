import React, { useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import Button from './Button';

/**
 * Modal Component - Reusable modal dialog with animations
 * 
 * @param {Object} props
 * @param {boolean} props.isOpen - Controls modal visibility
 * @param {Function} props.onClose - Function to close modal
 * @param {string} props.title - Modal title
 * @param {React.ReactNode} props.children - Modal content
 * @param {React.ReactNode} props.footer - Custom footer content
 * @param {string} props.size - 'sm', 'md', 'lg', 'xl', 'full'
 * @param {boolean} props.closeOnClickOutside - Close when clicking outside
 * @param {boolean} props.closeOnEsc - Close when pressing Escape
 * @param {boolean} props.showCloseButton - Show close button in header
 * @param {string} props.className - Additional CSS classes
 * @param {Function} props.onAfterOpen - Callback after modal opens
 * @param {Function} props.onAfterClose - Callback after modal closes
 */

const Modal = ({
  isOpen,
  onClose,
  title,
  children,
  footer,
  size = 'md',
  closeOnClickOutside = true,
  closeOnEsc = true,
  showCloseButton = true,
  className = '',
  onAfterOpen,
  onAfterClose,
}) => {
  
  // Handle ESC key press
  useEffect(() => {
    const handleEsc = (e) => {
      if (closeOnEsc && e.key === 'Escape' && isOpen) {
        onClose();
      }
    };

    window.addEventListener('keydown', handleEsc);
    return () => window.removeEventListener('keydown', handleEsc);
  }, [closeOnEsc, isOpen, onClose]);

  // Prevent body scroll when modal is open
  useEffect(() => {
    if (isOpen) {
      document.body.style.overflow = 'hidden';
      onAfterOpen?.();
    } else {
      document.body.style.overflow = 'unset';
      onAfterClose?.();
    }

    return () => {
      document.body.style.overflow = 'unset';
    };
  }, [isOpen, onAfterOpen, onAfterClose]);

  // Size mappings
  const sizeStyles = {
    sm: 'max-w-md',
    md: 'max-w-lg',
    lg: 'max-w-2xl',
    xl: 'max-w-4xl',
    full: 'max-w-[90vw] w-full',
  };

  // Animation variants
  const backdropVariants = {
    hidden: { opacity: 0 },
    visible: { opacity: 1 },
  };

  const modalVariants = {
    hidden: { opacity: 0, scale: 0.95, y: 20 },
    visible: { 
      opacity: 1, 
      scale: 1, 
      y: 0,
      transition: { type: 'spring', damping: 25, stiffness: 300 }
    },
    exit: { 
      opacity: 0, 
      scale: 0.95, 
      y: 20,
      transition: { duration: 0.2 }
    },
  };

  if (!isOpen) return null;

  return (
    <AnimatePresence>
      {isOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
          {/* Backdrop */}
          <motion.div
            variants={backdropVariants}
            initial="hidden"
            animate="visible"
            exit="hidden"
            onClick={closeOnClickOutside ? onClose : undefined}
            className="absolute inset-0 bg-dark-500/80 backdrop-blur-sm"
          />

          {/* Modal */}
          <motion.div
            variants={modalVariants}
            initial="hidden"
            animate="visible"
            exit="exit"
            className={`
              relative glass rounded-2xl w-full 
              ${sizeStyles[size]} max-h-[90vh] overflow-hidden
              flex flex-col
              ${className}
            `}
          >
            {/* Header */}
            {(title || showCloseButton) && (
              <div className="flex items-center justify-between p-6 border-b border-white/10">
                {title && (
                  <h3 className="text-xl font-semibold text-white">{title}</h3>
                )}
                {showCloseButton && (
                  <button
                    onClick={onClose}
                    className="p-2 rounded-lg hover:bg-white/10 text-gray-400 hover:text-white transition-colors"
                    aria-label="Close modal"
                  >
                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                    </svg>
                  </button>
                )}
              </div>
            )}

            {/* Content */}
            <div className="flex-1 overflow-y-auto p-6">
              {children}
            </div>

            {/* Footer */}
            {footer && (
              <div className="p-6 border-t border-white/10">
                {footer}
              </div>
            )}
          </motion.div>
        </div>
      )}
    </AnimatePresence>
  );
};

// ============================================
// Confirmation Modal (Pre-built for confirmations)
// ============================================

export const ConfirmModal = ({
  isOpen,
  onClose,
  onConfirm,
  title = 'Confirm Action',
  message = 'Are you sure you want to proceed?',
  confirmText = 'Confirm',
  cancelText = 'Cancel',
  variant = 'danger',
  loading = false,
}) => {
  return (
    <Modal
      isOpen={isOpen}
      onClose={onClose}
      title={title}
      size="sm"
      footer={
        <div className="flex justify-end gap-3">
          <Button variant="ghost" onClick={onClose} disabled={loading}>
            {cancelText}
          </Button>
          <Button 
            variant={variant} 
            onClick={onConfirm}
            loading={loading}
          >
            {confirmText}
          </Button>
        </div>
      }
    >
      <div className="text-center">
        <div className="mb-4">
          {variant === 'danger' && (
            <div className="w-16 h-16 mx-auto bg-red-500/20 rounded-full flex items-center justify-center">
              <svg className="w-8 h-8 text-red-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
              </svg>
            </div>
          )}
          {variant === 'success' && (
            <div className="w-16 h-16 mx-auto bg-green-500/20 rounded-full flex items-center justify-center">
              <svg className="w-8 h-8 text-green-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
            </div>
          )}
        </div>
        <p className="text-gray-300">{message}</p>
      </div>
    </Modal>
  );
};

// ============================================
// Form Modal (Pre-built for forms)
// ============================================

export const FormModal = ({
  isOpen,
  onClose,
  onSubmit,
  title,
  children,
  submitText = 'Submit',
  cancelText = 'Cancel',
  size = 'md',
  loading = false,
}) => {
  const handleSubmit = (e) => {
    e.preventDefault();
    onSubmit(e);
  };

  return (
    <Modal
      isOpen={isOpen}
      onClose={onClose}
      title={title}
      size={size}
      footer={
        <div className="flex justify-end gap-3">
          <Button variant="ghost" onClick={onClose} disabled={loading}>
            {cancelText}
          </Button>
          <Button 
            type="submit" 
            form="modal-form"
            variant="primary" 
            loading={loading}
          >
            {submitText}
          </Button>
        </div>
      }
    >
      <form id="modal-form" onSubmit={handleSubmit}>
        {children}
      </form>
    </Modal>
  );
};

// ============================================
// Image Preview Modal
// ============================================

export const ImagePreviewModal = ({
  isOpen,
  onClose,
  src,
  alt = 'Image preview',
}) => {
  return (
    <Modal
      isOpen={isOpen}
      onClose={onClose}
      size="xl"
      showCloseButton={true}
      className="bg-transparent"
    >
      <div className="flex items-center justify-center">
        <img 
          src={src} 
          alt={alt} 
          className="max-w-full max-h-[80vh] rounded-lg"
        />
      </div>
    </Modal>
  );
};

// ============================================
// Video Preview Modal
// ============================================

export const VideoPreviewModal = ({
  isOpen,
  onClose,
  src,
  type = 'video/mp4',
}) => {
  return (
    <Modal
      isOpen={isOpen}
      onClose={onClose}
      size="xl"
      showCloseButton={true}
    >
      <video 
        controls 
        autoPlay 
        className="w-full rounded-lg"
      >
        <source src={src} type={type} />
        Your browser does not support the video tag.
      </video>
    </Modal>
  );
};

// ============================================
// Results Modal (For showing detection results)
// ============================================

export const ResultModal = ({
  isOpen,
  onClose,
  result,
}) => {
  if (!result) return null;

  const isFake = result.prediction === 'FAKE';
  const confidence = (result.confidence * 100).toFixed(2);

  return (
    <Modal
      isOpen={isOpen}
      onClose={onClose}
      title="Detection Result"
      size="md"
    >
      <div className="space-y-6">
        {/* Result Icon */}
        <div className="flex justify-center">
          <div className={`
            w-20 h-20 rounded-full flex items-center justify-center
            ${isFake ? 'bg-red-500/20' : 'bg-green-500/20'}
          `}>
            {isFake ? (
              <svg className="w-10 h-10 text-red-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
            ) : (
              <svg className="w-10 h-10 text-green-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
            )}
          </div>
        </div>

        {/* Result Text */}
        <div className="text-center">
          <h3 className={`text-2xl font-bold mb-2 ${isFake ? 'text-red-400' : 'text-green-400'}`}>
            {isFake ? 'Deepfake Detected!' : 'Authentic Media'}
          </h3>
          <p className="text-gray-400">
            Confidence: <span className="text-white font-semibold">{confidence}%</span>
          </p>
        </div>

        {/* Details */}
        {result.details && (
          <div className="glass p-4 rounded-xl">
            <h4 className="text-white font-semibold mb-2">Details</h4>
            <p className="text-gray-400 text-sm">{result.details}</p>
          </div>
        )}

        {/* Face Detection Info */}
        {result.faces_detected > 0 && (
          <div className="glass p-4 rounded-xl">
            <h4 className="text-white font-semibold mb-2">Faces Detected</h4>
            <p className="text-gray-400">Found {result.faces_detected} face(s) in the image</p>
          </div>
        )}

        {/* Actions */}
        <div className="flex justify-center gap-3 pt-4">
          <Button variant="secondary" onClick={onClose}>
            Close
          </Button>
          <Button variant="primary" onClick={() => window.location.reload()}>
            New Detection
          </Button>
        </div>
      </div>
    </Modal>
  );
};

// ============================================
// Loading Modal
// ============================================

export const LoadingModal = ({
  isOpen,
  message = 'Processing...',
}) => {
  return (
    <Modal
      isOpen={isOpen}
      onClose={() => {}} // Prevent closing
      size="sm"
      closeOnClickOutside={false}
      closeOnEsc={false}
      showCloseButton={false}
    >
      <div className="text-center py-8">
        <div className="spinner mx-auto mb-4"></div>
        <p className="text-gray-300">{message}</p>
        <p className="text-sm text-gray-500 mt-2">This may take a few moments</p>
      </div>
    </Modal>
  );
};

// ============================================
// Settings Modal
// ============================================

export const SettingsModal = ({
  isOpen,
  onClose,
  settings,
  onSave,
}) => {
  const [localSettings, setLocalSettings] = React.useState(settings);

  return (
    <FormModal
      isOpen={isOpen}
      onClose={onClose}
      onSubmit={() => onSave(localSettings)}
      title="Settings"
      size="lg"
    >
      <div className="space-y-6">
        {/* API Settings */}
        <div>
          <h4 className="text-white font-semibold mb-4">API Configuration</h4>
          <div className="space-y-4">
            <div>
              <label className="block text-sm text-gray-400 mb-2">API URL</label>
              <input
                type="text"
                value={localSettings?.apiUrl || ''}
                onChange={(e) => setLocalSettings({...localSettings, apiUrl: e.target.value})}
                className="input-field"
                placeholder="http://localhost:8000"
              />
            </div>
          </div>
        </div>

        {/* Detection Settings */}
        <div>
          <h4 className="text-white font-semibold mb-4">Detection Settings</h4>
          <div className="space-y-4">
            <label className="flex items-center gap-3">
              <input
                type="checkbox"
                checked={localSettings?.generateHeatmap || false}
                onChange={(e) => setLocalSettings({...localSettings, generateHeatmap: e.target.checked})}
                className="w-4 h-4 rounded border-white/10 bg-white/5 text-primary-500"
              />
              <span className="text-gray-300">Generate heatmaps</span>
            </label>
            
            <label className="flex items-center gap-3">
              <input
                type="checkbox"
                checked={localSettings?.storeOnBlockchain || false}
                onChange={(e) => setLocalSettings({...localSettings, storeOnBlockchain: e.target.checked})}
                className="w-4 h-4 rounded border-white/10 bg-white/5 text-primary-500"
              />
              <span className="text-gray-300">Store results on blockchain</span>
            </label>
          </div>
        </div>

        {/* Notification Settings */}
        <div>
          <h4 className="text-white font-semibold mb-4">Notifications</h4>
          <div className="space-y-4">
            <label className="flex items-center gap-3">
              <input
                type="checkbox"
                checked={localSettings?.emailNotifications || false}
                onChange={(e) => setLocalSettings({...localSettings, emailNotifications: e.target.checked})}
                className="w-4 h-4 rounded border-white/10 bg-white/5 text-primary-500"
              />
              <span className="text-gray-300">Email notifications</span>
            </label>
          </div>
        </div>
      </div>
    </FormModal>
  );
};

export default Modal;