import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  FiCheckCircle, 
  FiAlertCircle, 
  FiInfo, 
  FiXCircle,
  FiX 
} from 'react-icons/fi';

/**
 * Alert Component - Displays contextual feedback messages
 * 
 * @param {Object} props
 * @param {string} props.type - 'success', 'error', 'warning', 'info'
 * @param {string} props.message - Alert message
 * @param {string} props.description - Optional detailed description
 * @param {boolean} props.dismissible - Whether alert can be dismissed
 * @param {number} props.autoClose - Auto close timeout in ms (0 to disable)
 * @param {function} props.onClose - Callback when alert is closed
 * @param {string} props.className - Additional CSS classes
 * @param {Object} props.icon - Custom icon component
 * @param {boolean} props.showIcon - Whether to show icon
 * @param {string} props.size - 'sm', 'md', 'lg'
 * @param {string} props.variant - 'solid', 'outline', 'soft'
 */

const Alert = ({
  type = 'info',
  message,
  description,
  dismissible = true,
  autoClose = 0,
  onClose,
  className = '',
  icon: CustomIcon,
  showIcon = true,
  size = 'md',
  variant = 'soft',
  ...props
}) => {
  const [isVisible, setIsVisible] = useState(true);

  // Auto close functionality
  useEffect(() => {
    if (autoClose > 0 && isVisible) {
      const timer = setTimeout(() => {
        handleClose();
      }, autoClose);

      return () => clearTimeout(timer);
    }
  }, [autoClose, isVisible]);

  const handleClose = () => {
    setIsVisible(false);
    if (onClose) {
      onClose();
    }
  };

  // Define styles based on type and variant
  const getStyles = () => {
    const baseStyles = {
      success: {
        solid: 'bg-green-600 text-white border-green-700',
        outline: 'bg-transparent border-2 border-green-500 text-green-400',
        soft: 'bg-green-500/10 border border-green-500/30 text-green-400',
      },
      error: {
        solid: 'bg-red-600 text-white border-red-700',
        outline: 'bg-transparent border-2 border-red-500 text-red-400',
        soft: 'bg-red-500/10 border border-red-500/30 text-red-400',
      },
      warning: {
        solid: 'bg-yellow-600 text-white border-yellow-700',
        outline: 'bg-transparent border-2 border-yellow-500 text-yellow-400',
        soft: 'bg-yellow-500/10 border border-yellow-500/30 text-yellow-400',
      },
      info: {
        solid: 'bg-blue-600 text-white border-blue-700',
        outline: 'bg-transparent border-2 border-blue-500 text-blue-400',
        soft: 'bg-blue-500/10 border border-blue-500/30 text-blue-400',
      },
    };

    return baseStyles[type]?.[variant] || baseStyles.info.soft;
  };

  // Size classes
  const sizeClasses = {
    sm: 'p-3 text-sm',
    md: 'p-4 text-base',
    lg: 'p-5 text-lg',
  };

  // Icons based on type
  const getIcon = () => {
    if (CustomIcon) return CustomIcon;

    const iconProps = {
      success: FiCheckCircle,
      error: FiXCircle,
      warning: FiAlertCircle,
      info: FiInfo,
    };

    const IconComponent = iconProps[type] || FiInfo;
    
    const iconSizes = {
      sm: 'w-4 h-4',
      md: 'w-5 h-5',
      lg: 'w-6 h-6',
    };

    return <IconComponent className={`${iconSizes[size]} flex-shrink-0`} />;
  };

  if (!isVisible) return null;

  return (
    <AnimatePresence>
      {isVisible && (
        <motion.div
          initial={{ opacity: 0, y: -20, scale: 0.95 }}
          animate={{ opacity: 1, y: 0, scale: 1 }}
          exit={{ opacity: 0, x: 100 }}
          transition={{ duration: 0.2 }}
          className={`
            relative rounded-xl backdrop-blur-sm
            ${getStyles()}
            ${sizeClasses[size]}
            ${className}
          `}
          role="alert"
          {...props}
        >
          <div className="flex items-start gap-3">
            {/* Icon */}
            {showIcon && (
              <div className="flex-shrink-0 mt-0.5">
                {getIcon()}
              </div>
            )}

            {/* Content */}
            <div className="flex-1">
              <div className="font-semibold">{message}</div>
              {description && (
                <div className={`
                  mt-1 
                  ${variant === 'solid' ? 'text-white/90' : 'text-gray-300'}
                  ${size === 'sm' ? 'text-xs' : size === 'md' ? 'text-sm' : 'text-base'}
                `}>
                  {description}
                </div>
              )}
            </div>

            {/* Dismiss button */}
            {dismissible && (
              <button
                onClick={handleClose}
                className={`
                  flex-shrink-0 p-1 rounded-lg transition-all duration-200
                  hover:bg-white/10 focus:outline-none focus:ring-2 
                  focus:ring-white/50
                `}
                aria-label="Close alert"
              >
                <FiX className={`${size === 'sm' ? 'w-3 h-3' : size === 'md' ? 'w-4 h-4' : 'w-5 h-5'}`} />
              </button>
            )}
          </div>

          {/* Progress bar for auto-close */}
          {autoClose > 0 && (
            <motion.div
              initial={{ width: '100%' }}
              animate={{ width: '0%' }}
              transition={{ duration: autoClose / 1000, ease: 'linear' }}
              className={`
                absolute bottom-0 left-0 h-1 
                ${type === 'success' ? 'bg-green-400' :
                  type === 'error' ? 'bg-red-400' :
                  type === 'warning' ? 'bg-yellow-400' :
                  'bg-blue-400'}
                rounded-b-xl
              `}
            />
          )}
        </motion.div>
      )}
    </AnimatePresence>
  );
};

// Context Provider for global alerts
export const AlertContext = React.createContext();

export const AlertProvider = ({ children }) => {
  const [alerts, setAlerts] = useState([]);

  const showAlert = (alert) => {
    const id = Date.now();
    setAlerts((prev) => [...prev, { ...alert, id }]);
    
    // Auto remove if not dismissible
    if (!alert.dismissible && alert.autoClose === 0) {
      setTimeout(() => {
        removeAlert(id);
      }, 5000);
    }
  };

  const removeAlert = (id) => {
    setAlerts((prev) => prev.filter((alert) => alert.id !== id));
  };

  const success = (message, options = {}) => {
    showAlert({ type: 'success', message, ...options });
  };

  const error = (message, options = {}) => {
    showAlert({ type: 'error', message, ...options });
  };

  const warning = (message, options = {}) => {
    showAlert({ type: 'warning', message, ...options });
  };

  const info = (message, options = {}) => {
    showAlert({ type: 'info', message, ...options });
  };

  const clearAll = () => {
    setAlerts([]);
  };

  return (
    <AlertContext.Provider
      value={{
        alerts,
        showAlert,
        removeAlert,
        success,
        error,
        warning,
        info,
        clearAll,
      }}
    >
      {children}
      
      {/* Alert Container */}
      <div className="fixed top-20 right-4 z-50 space-y-3 w-96 max-w-full">
        <AnimatePresence mode="popLayout">
          {alerts.map((alert) => (
            <motion.div
              key={alert.id}
              initial={{ opacity: 0, x: 100 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: 100 }}
              transition={{ duration: 0.2 }}
            >
              <Alert
                {...alert}
                onClose={() => removeAlert(alert.id)}
              />
            </motion.div>
          ))}
        </AnimatePresence>
      </div>
    </AlertContext.Provider>
  );
};

// Custom hook for using alerts
export const useAlert = () => {
  const context = React.useContext(AlertContext);
  if (!context) {
    throw new Error('useAlert must be used within an AlertProvider');
  }
  return context;
};

// Individual alert variants for direct use
export const SuccessAlert = (props) => <Alert type="success" {...props} />;
export const ErrorAlert = (props) => <Alert type="error" {...props} />;
export const WarningAlert = (props) => <Alert type="warning" {...props} />;
export const InfoAlert = (props) => <Alert type="info" {...props} />;

export default Alert;