import React from 'react';
import { motion } from 'framer-motion';
import { ButtonLoader } from './Loader';

/**
 * Button Component - Versatile button with multiple variants and states
 * 
 * @param {Object} props
 * @param {React.ReactNode} props.children - Button content
 * @param {'primary'|'secondary'|'outline'|'ghost'|'danger'|'success'|'warning'|'info'} props.variant - Button style variant
 * @param {'sm'|'md'|'lg'} props.size - Button size
 * @param {boolean} props.loading - Show loading state
 * @param {boolean} props.disabled - Disable button
 * @param {Function} props.onClick - Click handler
 * @param {string} props.type - Button type (button, submit, reset)
 * @param {React.ReactNode} props.icon - Icon component
 * @param {'left'|'right'} props.iconPosition - Icon position
 * @param {boolean} props.fullWidth - Make button full width
 * @param {string} props.className - Additional CSS classes
 * @param {boolean} props.rounded - Use rounded-full instead of rounded-xl
 * @param {string} props.href - If provided, renders as link
 * @param {Object} props.props - Additional props
 */

const Button = ({
  children,
  variant = 'primary',
  size = 'md',
  loading = false,
  disabled = false,
  onClick,
  type = 'button',
  icon,
  iconPosition = 'left',
  fullWidth = false,
  className = '',
  rounded = false,
  href,
  ...props
}) => {
  
  // Base styles
  const baseStyles = 'font-semibold inline-flex items-center justify-center gap-2 transition-all duration-300 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-dark-500';
  
  // Border radius
  const radiusStyles = rounded ? 'rounded-full' : 'rounded-xl';
  
  // Size styles
  const sizeStyles = {
    sm: 'px-4 py-2 text-sm gap-1.5',
    md: 'px-6 py-3 text-base gap-2',
    lg: 'px-8 py-4 text-lg gap-3',
  };

  // Variant styles
  const variantStyles = {
    // Primary - Gradient blue
    primary: 'bg-gradient-to-r from-primary-600 to-primary-500 hover:from-primary-500 hover:to-primary-400 text-white shadow-lg shadow-primary-500/25 hover:shadow-xl focus:ring-primary-500',
    
    // Secondary - Glass effect
    secondary: 'glass hover:bg-white/20 text-white border border-white/20 focus:ring-white/50',
    
    // Outline - Blue border
    outline: 'border-2 border-primary-500 text-primary-400 hover:bg-primary-500/10 hover:border-primary-400 focus:ring-primary-500',
    
    // Ghost - Transparent
    ghost: 'text-gray-400 hover:text-white hover:bg-white/10 focus:ring-white/50',
    
    // Danger - Red
    danger: 'bg-red-600 hover:bg-red-700 text-white shadow-lg shadow-red-600/25 hover:shadow-xl focus:ring-red-500',
    
    // Success - Green
    success: 'bg-green-600 hover:bg-green-700 text-white shadow-lg shadow-green-600/25 hover:shadow-xl focus:ring-green-500',
    
    // Warning - Yellow
    warning: 'bg-yellow-600 hover:bg-yellow-700 text-white shadow-lg shadow-yellow-600/25 hover:shadow-xl focus:ring-yellow-500',
    
    // Info - Light blue
    info: 'bg-blue-600 hover:bg-blue-700 text-white shadow-lg shadow-blue-600/25 hover:shadow-xl focus:ring-blue-500',
  };

  // Disabled styles
  const disabledStyles = 'opacity-50 cursor-not-allowed pointer-events-none';
  
  // Loading styles
  const loadingStyles = 'cursor-wait';

  // Combine all styles
  const buttonStyles = `
    ${baseStyles}
    ${radiusStyles}
    ${sizeStyles[size]}
    ${variantStyles[variant]}
    ${disabled || loading ? disabledStyles : ''}
    ${loading ? loadingStyles : ''}
    ${fullWidth ? 'w-full' : ''}
    ${className}
  `;

  // Icon element
  const IconElement = icon && (
    <span className={`inline-flex ${loading ? 'opacity-0' : ''}`}>
      {icon}
    </span>
  );

  // Content with icon positioning
  const content = (
    <>
      {icon && iconPosition === 'left' && IconElement}
      <span className={loading ? 'opacity-0' : ''}>{children}</span>
      {icon && iconPosition === 'right' && IconElement}
      {loading && (
        <span className="absolute inset-0 flex items-center justify-center">
          <ButtonLoader size={size} color={variant === 'primary' ? 'white' : 'primary'} />
        </span>
      )}
    </>
  );

  // Animation variants
  const animationVariants = {
    hover: { scale: 1.02 },
    tap: { scale: 0.98 },
  };

  // If href is provided, render as link
  if (href) {
    return (
      <motion.a
        href={href}
        whileHover={!disabled && !loading ? "hover" : undefined}
        whileTap={!disabled && !loading ? "tap" : undefined}
        variants={animationVariants}
        className={buttonStyles}
        {...props}
      >
        {content}
      </motion.a>
    );
  }

  // Otherwise render as button
  return (
    <motion.button
      type={type}
      onClick={onClick}
      disabled={disabled || loading}
      whileHover={!disabled && !loading ? "hover" : undefined}
      whileTap={!disabled && !loading ? "tap" : undefined}
      variants={animationVariants}
      className={buttonStyles}
      {...props}
    >
      {content}
    </motion.button>
  );
};

// ============================================
// Icon Button Component (Square/Circle button with just icon)
// ============================================

export const IconButton = ({
  icon,
  variant = 'secondary',
  size = 'md',
  loading = false,
  disabled = false,
  onClick,
  type = 'button',
  rounded = true,
  className = '',
  label,
  ...props
}) => {
  
  const sizeStyles = {
    sm: 'p-2',
    md: 'p-3',
    lg: 'p-4',
  };

  const iconSizes = {
    sm: 'w-4 h-4',
    md: 'w-5 h-5',
    lg: 'w-6 h-6',
  };

  const variantStyles = {
    primary: 'bg-gradient-to-r from-primary-600 to-primary-500 hover:from-primary-500 hover:to-primary-400 text-white',
    secondary: 'glass hover:bg-white/20 text-white',
    outline: 'border-2 border-primary-500 text-primary-400 hover:bg-primary-500/10',
    ghost: 'text-gray-400 hover:text-white hover:bg-white/10',
    danger: 'bg-red-600 hover:bg-red-700 text-white',
    success: 'bg-green-600 hover:bg-green-700 text-white',
  };

  return (
    <motion.button
      type={type}
      onClick={onClick}
      disabled={disabled || loading}
      whileHover={{ scale: 1.05 }}
      whileTap={{ scale: 0.95 }}
      className={`
        inline-flex items-center justify-center
        ${sizeStyles[size]}
        ${rounded ? 'rounded-full' : 'rounded-xl'}
        ${variantStyles[variant]}
        ${disabled || loading ? 'opacity-50 cursor-not-allowed' : ''}
        relative
        ${className}
      `}
      aria-label={label}
      {...props}
    >
      {loading ? (
        <ButtonLoader size={size} color={variant === 'primary' ? 'white' : 'primary'} />
      ) : (
        <span className={iconSizes[size]}>{icon}</span>
      )}
    </motion.button>
  );
};

// ============================================
// Button Group Component
// ============================================

export const ButtonGroup = ({
  children,
  orientation = 'horizontal',
  className = '',
  ...props
}) => {
  return (
    <div
      className={`
        inline-flex
        ${orientation === 'horizontal' ? 'flex-row' : 'flex-col'}
        gap-px
        ${className}
      `}
      {...props}
    >
      {React.Children.map(children, (child, index) => {
        if (!child) return null;
        
        // Add border radius styles based on position
        const isFirst = index === 0;
        const isLast = index === React.Children.count(children) - 1;
        
        let radiusClasses = '';
        if (orientation === 'horizontal') {
          if (isFirst) radiusClasses = 'rounded-r-none';
          if (isLast) radiusClasses = 'rounded-l-none';
          if (!isFirst && !isLast) radiusClasses = 'rounded-none';
        } else {
          if (isFirst) radiusClasses = 'rounded-b-none';
          if (isLast) radiusClasses = 'rounded-t-none';
          if (!isFirst && !isLast) radiusClasses = 'rounded-none';
        }
        
        return React.cloneElement(child, {
          className: `${child.props.className || ''} ${radiusClasses}`,
        });
      })}
    </div>
  );
};

// ============================================
// Floating Action Button
// ============================================

export const FAB = ({
  icon,
  onClick,
  label,
  position = 'bottom-right',
  ...props
}) => {
  const positionStyles = {
    'bottom-right': 'bottom-6 right-6',
    'bottom-left': 'bottom-6 left-6',
    'top-right': 'top-6 right-6',
    'top-left': 'top-6 left-6',
  };

  return (
    <motion.button
      onClick={onClick}
      whileHover={{ scale: 1.1 }}
      whileTap={{ scale: 0.9 }}
      initial={{ scale: 0 }}
      animate={{ scale: 1 }}
      className={`
        fixed ${positionStyles[position]} z-50
        p-4 bg-gradient-to-r from-primary-600 to-primary-500
        text-white rounded-full shadow-xl shadow-primary-500/50
        hover:shadow-2xl transition-all duration-300
        flex items-center justify-center
      `}
      aria-label={label}
      {...props}
    >
      <span className="w-6 h-6">{icon}</span>
    </motion.button>
  );
};

// ============================================
// Social Media Buttons
// ============================================

export const SocialButton = ({
  provider,
  onClick,
  className = '',
  ...props
}) => {
  const providers = {
    google: {
      name: 'Google',
      icon: (
        <svg className="w-5 h-5" viewBox="0 0 24 24">
          <path
            fill="currentColor"
            d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"
          />
          <path
            fill="currentColor"
            d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"
          />
          <path
            fill="currentColor"
            d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z"
          />
          <path
            fill="currentColor"
            d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"
          />
        </svg>
      ),
      bg: 'bg-white hover:bg-gray-100 text-gray-800',
    },
    github: {
      name: 'GitHub',
      icon: (
        <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
          <path d="M12 2C6.477 2 2 6.484 2 12.017c0 4.425 2.865 8.18 6.839 9.504.5.092.682-.217.682-.483 0-.237-.008-.868-.013-1.703-2.782.605-3.369-1.343-3.369-1.343-.454-1.158-1.11-1.466-1.11-1.466-.908-.62.069-.608.069-.608 1.003.07 1.531 1.032 1.531 1.032.892 1.53 2.341 1.088 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.113-4.555-4.951 0-1.093.39-1.988 1.029-2.688-.103-.253-.446-1.272.098-2.65 0 0 .84-.27 2.75 1.026A9.564 9.564 0 0112 6.844c.85.004 1.705.115 2.504.337 1.909-1.296 2.747-1.027 2.747-1.027.546 1.379.202 2.398.1 2.651.64.7 1.028 1.595 1.028 2.688 0 3.848-2.339 4.695-4.566 4.943.359.309.678.92.678 1.855 0 1.338-.012 2.419-.012 2.747 0 .268.18.58.688.482A10.019 10.019 0 0022 12.017C22 6.484 17.522 2 12 2z" />
        </svg>
      ),
      bg: 'bg-gray-800 hover:bg-gray-700 text-white',
    },
    twitter: {
      name: 'Twitter',
      icon: (
        <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
          <path d="M8.29 20.251c7.547 0 11.675-6.253 11.675-11.675 0-.178 0-.355-.012-.53A8.348 8.348 0 0022 5.92a8.19 8.19 0 01-2.357.646 4.118 4.118 0 001.804-2.27 8.224 8.224 0 01-2.605.996 4.107 4.107 0 00-6.993 3.743 11.65 11.65 0 01-8.457-4.287 4.106 4.106 0 001.27 5.477A4.072 4.072 0 012.8 9.713v.052a4.105 4.105 0 003.292 4.022 4.095 4.095 0 01-1.853.07 4.108 4.108 0 003.834 2.85A8.233 8.233 0 012 18.407a11.616 11.616 0 006.29 1.84" />
        </svg>
      ),
      bg: 'bg-blue-400 hover:bg-blue-500 text-white',
    },
  };

  const providerData = providers[provider] || providers.google;

  return (
    <button
      onClick={onClick}
      className={`
        flex items-center justify-center gap-2 px-4 py-2
        rounded-xl font-medium transition-all duration-300
        ${providerData.bg}
        ${className}
      `}
      {...props}
    >
      {providerData.icon}
      <span>Continue with {providerData.name}</span>
    </button>
  );
};

// ============================================
// Preset combinations for common use cases
// ============================================

export const SubmitButton = (props) => (
  <Button type="submit" variant="primary" {...props} />
);

export const CancelButton = (props) => (
  <Button variant="ghost" {...props} />
);

export const DeleteButton = (props) => (
  <Button variant="danger" icon={
    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
    </svg>
  } {...props} />
);

export const UploadButton = (props) => (
  <Button variant="primary" icon={
    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12" />
    </svg>
  } {...props} />
);

export const DownloadButton = (props) => (
  <Button variant="secondary" icon={
    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
    </svg>
  } {...props} />
);

export const RefreshButton = (props) => (
  <IconButton
    icon={
      <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
      </svg>
    }
    variant="ghost"
    {...props}
  />
);

export const CloseButton = (props) => (
  <IconButton
    icon={
      <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
      </svg>
    }
    variant="ghost"
    size="sm"
    {...props}
  />
);

export default Button;