import React, { createContext, useContext, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

const AlertContext = createContext();

export const AlertProvider = ({ children }) => {
  const [alerts, setAlerts] = useState([]);

  const showAlert = ({ type, message, description, autoClose = 5000 }) => {
    const id = Date.now();
    setAlerts(prev => [...prev, { id, type, message, description }]);
    setTimeout(() => {
      setAlerts(prev => prev.filter(a => a.id !== id));
    }, autoClose);
  };

  const alertMethods = {
    success: (msg, desc) => showAlert({ type: 'success', message: msg, description: desc }),
    error: (msg, desc) => showAlert({ type: 'error', message: msg, description: desc }),
    warning: (msg, desc) => showAlert({ type: 'warning', message: msg, description: desc }),
    info: (msg, desc) => showAlert({ type: 'info', message: msg, description: desc }),
  };

  return (
    <AlertContext.Provider value={alertMethods}>
      {children}
      <div className="fixed top-20 right-4 z-50 space-y-3 w-96">
        <AnimatePresence>
          {alerts.map(alert => (
            <motion.div
              key={alert.id}
              initial={{ opacity: 0, x: 100 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: 100 }}
              className={`p-4 rounded-xl border ${
                alert.type === 'success' ? 'bg-green-500/10 border-green-500/30 text-green-400' :
                alert.type === 'error' ? 'bg-red-500/10 border-red-500/30 text-red-400' :
                alert.type === 'warning' ? 'bg-yellow-500/10 border-yellow-500/30 text-yellow-400' :
                'bg-blue-500/10 border-blue-500/30 text-blue-400'
              }`}
            >
              <p className="font-medium">{alert.message}</p>
              {alert.description && <p className="text-sm mt-1 opacity-90">{alert.description}</p>}
            </motion.div>
          ))}
        </AnimatePresence>
      </div>
    </AlertContext.Provider>
  );
};

export const useAlert = () => {
  const context = useContext(AlertContext);
  if (!context) throw new Error('useAlert must be used within AlertProvider');
  return context;
};