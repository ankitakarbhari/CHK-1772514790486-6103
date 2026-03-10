# training/train_url_classifier.py
"""
Complete Training Script for URL/Link Phishing Detection Model
Trains multiple ML models to detect malicious URLs, phishing sites, and scams
Supports feature extraction from URLs, domain analysis, and ensemble learning
"""

import os
import sys
import re
import json
import time
import math
import random
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from urllib.parse import urlparse, parse_qs
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

# ========== MACHINE LEARNING IMPORTS ==========
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.ensemble import (
    RandomForestClassifier, 
    GradientBoostingClassifier, 
    AdaBoostClassifier,
    ExtraTreesClassifier,
    VotingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report,
    precision_recall_curve, average_precision_score
)
from sklearn.pipeline import Pipeline
import joblib

# ========== URL PROCESSING ==========
import tldextract
import whois
import socket
import dns.resolver
from urllib.parse import urlparse

# ========== VISUALIZATION ==========
import matplotlib.pyplot as plt
import seaborn as sns

# ========== UTILS ==========
from sklearn.utils import class_weight

# Set random seed
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)

set_seed(42)


# ============================================
# CONFIGURATION
# ============================================

class Config:
    """Configuration for URL classifier training"""
    
    # Data configuration
    data_dir = Path('datasets/url_data')
    train_file = data_dir / 'train.csv'
    val_file = data_dir / 'val.csv'
    test_file = data_dir / 'test.csv'
    
    # Feature extraction
    use_whois = False  # WHOIS lookups are slow, disable for training
    use_dns = False    # DNS lookups are slow, disable for training
    
    # Model selection
    model_types = ['rf', 'gb', 'lr', 'svm', 'xgb', 'ensemble']  
    # 'rf': Random Forest, 'gb': Gradient Boosting, 'lr': Logistic Regression
    # 'svm': SVM, 'xgb': XGBoost, 'ensemble': Voting Ensemble
    
    # Training configuration
    test_size = 0.2
    validation_size = 0.2
    random_state = 42
    
    # Model hyperparameters
    rf_params = {
        'n_estimators': 100,
        'max_depth': 10,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'class_weight': 'balanced'
    }
    
    gb_params = {
        'n_estimators': 100,
        'max_depth': 5,
        'learning_rate': 0.1,
        'subsample': 0.8
    }
    
    lr_params = {
        'C': 1.0,
        'max_iter': 1000,
        'class_weight': 'balanced'
    }
    
    svm_params = {
        'C': 1.0,
        'kernel': 'rbf',
        'gamma': 'scale',
        'probability': True,
        'class_weight': 'balanced'
    }
    
    xgb_params = {
        'n_estimators': 100,
        'max_depth': 5,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8
    }
    
    # Grid search for hyperparameter tuning
    do_grid_search = True
    grid_search_params = {
        'rf': {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, 15, None],
            'min_samples_split': [2, 5, 10]
        },
        'gb': {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2]
        }
    }
    
    # Model save paths
    save_dir = Path('models/url_classifier')
    model_save_path = Path('models/url_classifier.pkl')
    scaler_save_path = Path('models/url_scaler.pkl')
    feature_names_path = Path('models/url_features.json')
    results_path = Path('models/url_classifier_results.json')
    
    # Evaluation
    cv_folds = 5
    
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        
        # Create directories
        self.save_dir.mkdir(exist_ok=True, parents=True)
        self.model_save_path.parent.mkdir(exist_ok=True, parents=True)
    
    def __str__(self):
        return json.dumps(self.__dict__, indent=2, default=str)
    
    def save(self, path):
        with open(path, 'w') as f:
            json.dump(self.__dict__, f, indent=2, default=str)


# ============================================
# URL FEATURE EXTRACTOR
# ============================================

class URLFeatureExtractor:
    """
    Extract 50+ features from URLs for ML classification
    """
    
    def __init__(self, use_whois=False, use_dns=False):
        self.use_whois = use_whois
        self.use_dns = use_dns
        
        # Suspicious TLDs
        self.suspicious_tlds = {
            'tk', 'ml', 'ga', 'cf', 'xyz', 'top', 'club', 'online', 
            'site', 'web', 'work', 'review', 'stream', 'download',
            'bid', 'trade', 'webcam', 'win', 'science', 'date', 'kim',
            'men', 'loan', 'click', 'link', 'download', 'review', 'racing',
            'faith', 'lol', 'bet', 'accountant', 'work', 'gdn'
        }
        
        # Suspicious keywords in URL
        self.suspicious_keywords = [
            'secure', 'login', 'signin', 'verify', 'account', 'update',
            'confirm', 'banking', 'paypal', 'apple', 'microsoft', 'google',
            'facebook', 'instagram', 'amazon', 'ebay', 'netflix', 'pay',
            'wallet', 'bitcoin', 'crypto', 'blockchain', 'support',
            'customer', 'service', 'help', 'claim', 'prize', 'winner',
            'free', 'bonus', 'promo', 'discount', 'offer', 'deal',
            'password', 'credit', 'card', 'ssn', 'social security',
            'verification', 'authenticate', 'security', 'alert', 'warning'
        ]
        
        # Brand names for typosquatting detection
        self.brands = [
            'google', 'facebook', 'amazon', 'apple', 'microsoft', 'netflix',
            'paypal', 'instagram', 'twitter', 'linkedin', 'whatsapp', 'spotify',
            'yahoo', 'bing', 'ebay', 'walmart', 'target', 'bestbuy', 'nike',
            'adidas', 'zara', 'hm', 'gap', 'ikea', 'wikipedia', 'reddit'
        ]
        
        # Common TLDs for legitimate sites
        self.common_tlds = {'com', 'org', 'net', 'edu', 'gov', 'io', 'co', 'uk', 'de', 'fr', 'jp', 'au', 'ca'}
        
        # Compile regex patterns
        self.ip_pattern = re.compile(r'^\d+\.\d+\.\d+\.\d+$')
        self.hex_pattern = re.compile(r'%[0-9a-fA-F]{2}')
        self.multi_dot_pattern = re.compile(r'\.{2,}')
        self.domain_pattern = re.compile(r'[a-zA-Z0-9-]+\.[a-zA-Z]{2,}')
    
    def extract_features(self, url):
        """
        Extract features from URL
        
        Returns:
            Dictionary of features
        """
        features = {}
        
        # Basic parsing
        parsed = urlparse(url)
        extracted = tldextract.extract(url)
        
        domain = extracted.domain
        suffix = extracted.suffix
        subdomain = extracted.subdomain
        full_domain = f"{domain}.{suffix}" if domain and suffix else ""
        
        # ====================================
        # URL Structure Features (1-15)
        # ====================================
        
        features['url_length'] = len(url)
        features['domain_length'] = len(full_domain)
        features['path_length'] = len(parsed.path)
        features['query_length'] = len(parsed.query)
        features['fragment_length'] = len(parsed.fragment)
        
        # Special character counts
        features['num_dots'] = url.count('.')
        features['num_hyphens'] = url.count('-')
        features['num_underscores'] = url.count('_')
        features['num_slashes'] = url.count('/')
        features['num_question_marks'] = url.count('?')
        features['num_equals'] = url.count('=')
        features['num_ampersands'] = url.count('&')
        features['num_at'] = url.count('@')
        features['num_percent'] = url.count('%')
        features['num_hashes'] = url.count('#')
        
        # ====================================
        # Content-Based Features (16-25)
        # ====================================
        
        # Digit and letter counts
        features['num_digits'] = sum(c.isdigit() for c in url)
        features['num_letters'] = sum(c.isalpha() for c in url)
        features['digit_ratio'] = features['num_digits'] / (len(url) + 1)
        
        # Suspicious indicators
        features['has_ip'] = 1 if self.ip_pattern.match(parsed.netloc.split(':')[0]) else 0
        features['has_port'] = 1 if parsed.port else 0
        features['has_hex_chars'] = 1 if self.hex_pattern.search(url) else 0
        features['has_multi_dots'] = 1 if self.multi_dot_pattern.search(url) else 0
        
        # URL encoding
        features['num_encoded_chars'] = len(re.findall(r'%[0-9a-fA-F]{2}', url))
        
        # ====================================
        # Domain-Based Features (26-35)
        # ====================================
        
        # TLD analysis
        features['tld_length'] = len(suffix)
        features['suspicious_tld'] = 1 if suffix in self.suspicious_tlds else 0
        features['common_tld'] = 1 if suffix in self.common_tlds else 0
        
        # Subdomain analysis
        features['subdomain_length'] = len(subdomain)
        features['subdomain_count'] = len(subdomain.split('.')) if subdomain else 0
        features['has_www'] = 1 if 'www' in subdomain else 0
        
        # Domain contains brand?
        domain_lower = domain.lower()
        brand_match = 0
        for brand in self.brands:
            if brand in domain_lower and domain_lower != brand:
                brand_match += 1
        features['brand_in_domain'] = min(brand_match, 1)
        features['brand_typosquatting'] = brand_match
        
        # ====================================
        # Path-Based Features (36-45)
        # ====================================
        
        features['path_depth'] = len([p for p in parsed.path.split('/') if p])
        features['has_extension'] = 1 if '.' in parsed.path.split('/')[-1] else 0
        features['path_has_digits'] = 1 if any(c.isdigit() for c in parsed.path) else 0
        
        # Suspicious paths
        suspicious_paths = ['login', 'signin', 'verify', 'account', 'update', 'secure']
        path_lower = parsed.path.lower()
        features['suspicious_path'] = sum(1 for sp in suspicious_paths if sp in path_lower)
        
        # ====================================
        # Query Parameter Features (46-50)
        # ====================================
        
        params = parse_qs(parsed.query)
        features['num_params'] = len(params)
        features['max_param_length'] = max([len(v[0]) for v in params.values()]) if params else 0
        
        # Suspicious parameters
        suspicious_params = ['redirect', 'url', 'link', 'return', 'next']
        param_names = ' '.join(params.keys()).lower()
        features['suspicious_params'] = sum(1 for sp in suspicious_params if sp in param_names)
        
        # ====================================
        # Keyword Matching (51-55)
        # ====================================
        
        url_lower = url.lower()
        features['suspicious_keywords'] = sum(
            1 for kw in self.suspicious_keywords if kw in url_lower
        )
        
        # Phishing indicators
        phishing_indicators = ['secure', 'login', 'verify', 'account', 'update']
        features['phishing_indicators'] = sum(1 for pi in phishing_indicators if pi in url_lower)
        
        # ====================================
        # Entropy Features (56-58)
        # ====================================
        
        features['entropy'] = self._calculate_entropy(url)
        features['domain_entropy'] = self._calculate_entropy(full_domain)
        features['path_entropy'] = self._calculate_entropy(parsed.path)
        
        # ====================================
        # WHOIS Features (if enabled)
        # ====================================
        
        if self.use_whois and full_domain:
            whois_features = self._extract_whois_features(full_domain)
            features.update(whois_features)
        else:
            features['domain_age_days'] = -1
            features['domain_age_months'] = -1
            features['domain_age_years'] = -1
            features['days_to_expiry'] = -1
            features['has_registrar'] = 0
        
        # ====================================
        # DNS Features (if enabled)
        # ====================================
        
        if self.use_dns and full_domain:
            dns_features = self._extract_dns_features(full_domain)
            features.update(dns_features)
        else:
            features['has_dns'] = 0
            features['num_ips'] = 0
            features['has_mx'] = 0
        
        return features
    
    def _calculate_entropy(self, text):
        """Calculate Shannon entropy of string"""
        if not text:
            return 0.0
        
        entropy = 0.0
        for i in range(256):
            char = chr(i)
            freq = text.count(char)
            if freq > 0:
                freq = float(freq) / len(text)
                entropy -= freq * math.log2(freq)
        
        return entropy
    
    def _extract_whois_features(self, domain):
        """Extract features from WHOIS data"""
        features = {
            'domain_age_days': -1,
            'domain_age_months': -1,
            'domain_age_years': -1,
            'days_to_expiry': -1,
            'has_registrar': 0
        }
        
        try:
            domain_info = whois.whois(domain)
            
            # Creation date
            creation_date = domain_info.creation_date
            if isinstance(creation_date, list):
                creation_date = creation_date[0]
            
            if creation_date:
                age_days = (datetime.now() - creation_date).days
                features['domain_age_days'] = age_days
                features['domain_age_months'] = age_days / 30
                features['domain_age_years'] = age_days / 365
            
            # Expiration date
            expiration_date = domain_info.expiration_date
            if isinstance(expiration_date, list):
                expiration_date = expiration_date[0]
            
            if expiration_date:
                days_to_expiry = (expiration_date - datetime.now()).days
                features['days_to_expiry'] = days_to_expiry
            
            # Registrar
            features['has_registrar'] = 1 if domain_info.registrar else 0
            
        except Exception:
            pass
        
        return features
    
    def _extract_dns_features(self, domain):
        """Extract features from DNS records"""
        features = {
            'has_dns': 0,
            'num_ips': 0,
            'has_mx': 0
        }
        
        try:
            # A records
            answers = dns.resolver.resolve(domain, 'A')
            features['num_ips'] = len(answers)
            features['has_dns'] = 1 if answers else 0
            
            # MX records
            try:
                mx_answers = dns.resolver.resolve(domain, 'MX')
                features['has_mx'] = 1 if mx_answers else 0
            except:
                pass
                
        except Exception:
            pass
        
        return features
    
    def get_feature_names(self):
        """Get list of feature names"""
        # Create dummy URL to extract all features
        dummy_features = self.extract_features('https://example.com')
        return list(dummy_features.keys())


# ============================================
# DATASET GENERATOR
# ============================================

class URLDatasetGenerator:
    """
    Generate synthetic URL dataset for training
    Can also load real datasets from files
    """
    
    def __init__(self, config):
        self.config = config
        self.feature_extractor = URLFeatureExtractor(
            use_whois=config.use_whois,
            use_dns=config.use_dns
        )
    
    def generate_synthetic_dataset(self, n_samples=10000):
        """
        Generate synthetic URL dataset
        
        Returns:
            DataFrame with features and labels
        """
        print(f"\n📊 Generating synthetic dataset with {n_samples} samples...")
        
        urls = []
        labels = []
        
        # Legitimate URL patterns
        legit_domains = [
            'google', 'facebook', 'amazon', 'microsoft', 'apple',
            'netflix', 'spotify', 'github', 'stackoverflow', 'wikipedia',
            'youtube', 'twitter', 'linkedin', 'instagram', 'reddit',
            'yahoo', 'bing', 'ebay', 'walmart', 'target'
        ]
        
        # Phishing URL patterns
        phishing_keywords = [
            'secure', 'login', 'verify', 'account', 'update', 'confirm',
            'signin', 'banking', 'paypal', 'apple-id', 'amazon-security'
        ]
        
        suspicious_tlds = ['tk', 'ml', 'ga', 'cf', 'xyz', 'top', 'club']
        
        pbar = tqdm(total=n_samples, desc="Generating URLs")
        
        for i in range(n_samples):
            if i < n_samples // 2:
                # Generate legitimate URL
                domain = random.choice(legit_domains)
                tld = random.choice(['com', 'org', 'net', 'io', 'co'])
                
                # Random path
                paths = ['', 'home', 'about', 'products', 'blog', 'contact', 'help']
                path = random.choice(paths)
                
                if path and random.random() > 0.3:
                    url = f"https://www.{domain}.{tld}/{path}"
                else:
                    url = f"https://www.{domain}.{tld}"
                
                labels.append(0)  # 0 = legitimate
                
            else:
                # Generate phishing URL
                if random.random() > 0.5:
                    # Typosquatting
                    domain = random.choice(legit_domains)
                    # Add common typo
                    if random.random() > 0.5:
                        domain = domain.replace('o', '0').replace('l', '1').replace('e', '3')
                    else:
                        domain = domain + '-' + random.choice(phishing_keywords)
                else:
                    # Random suspicious domain
                    domain = random.choice(phishing_keywords) + '-' + random.choice(legit_domains)
                
                tld = random.choice(suspicious_tlds + ['com', 'org', 'net'])
                
                # Use HTTP (not HTTPS)
                protocol = 'http' if random.random() > 0.3 else 'https'
                
                # Add login path
                url = f"{protocol}://{domain}.{tld}"
                
                if random.random() > 0.3:
                    url += "/login.php"
                
                # Add query parameters
                if random.random() > 0.5:
                    url += "?username=admin&password=12345&redirect=http://evil.com"
                
                labels.append(1)  # 1 = phishing
            
            urls.append(url)
            pbar.update(1)
        
        pbar.close()
        
        # Extract features
        print("\n🔍 Extracting features from URLs...")
        features_list = []
        
        for url in tqdm(urls, desc="Extracting features"):
            features = self.feature_extractor.extract_features(url)
            features_list.append(features)
        
        # Create DataFrame
        df = pd.DataFrame(features_list)
        df['url'] = urls
        df['label'] = labels
        
        print(f"\n✅ Dataset generated: {df.shape[0]} samples, {df.shape[1]-2} features")
        print(f"   Legitimate: {(df['label'] == 0).sum()}, Phishing: {(df['label'] == 1).sum()}")
        
        return df
    
    def load_from_csv(self, file_path):
        """Load dataset from CSV file"""
        print(f"\n📂 Loading dataset from {file_path}...")
        
        df = pd.read_csv(file_path)
        
        if 'url' not in df.columns or 'label' not in df.columns:
            raise ValueError("CSV must contain 'url' and 'label' columns")
        
        print(f"   Loaded {len(df)} samples")
        print(f"   Legitimate: {(df['label'] == 0).sum()}, Phishing: {(df['label'] == 1).sum()}")
        
        # Extract features if not already present
        if len(df.columns) <= 3:  # Only url and label
            print("   Extracting features...")
            features_list = []
            
            for url in tqdm(df['url'], desc="Extracting features"):
                features = self.feature_extractor.extract_features(url)
                features_list.append(features)
            
            features_df = pd.DataFrame(features_list)
            df = pd.concat([df, features_df], axis=1)
        
        return df
    
    def save_dataset(self, df, file_path):
        """Save dataset to CSV"""
        df.to_csv(file_path, index=False)
        print(f"✅ Dataset saved to {file_path}")


# ============================================
# MODEL TRAINER
# ============================================

class URLModelTrainer:
    """Train multiple ML models for URL classification"""
    
    def __init__(self, config):
        self.config = config
        self.models = {}
        self.results = {}
        self.feature_names = []
        
        print(f"\n{'='*60}")
        print("URL CLASSIFIER TRAINER INITIALIZED")
        print(f"{'='*60}")
        print(f"Configuration:\n{config}")
    
    def prepare_data(self, df):
        """Prepare data for training"""
        
        # Separate features and target
        feature_cols = [col for col in df.columns if col not in ['url', 'label']]
        self.feature_names = feature_cols
        
        X = df[feature_cols].values
        y = df['label'].values
        
        # Handle missing values
        X = np.nan_to_num(X, nan=-1)
        
        # Split data
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, 
            test_size=self.config.test_size, 
            random_state=self.config.random_state,
            stratify=y
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=self.config.validation_size,
            random_state=self.config.random_state,
            stratify=y_temp
        )
        
        print(f"\n📊 Data split:")
        print(f"   Train: {len(X_train)} samples")
        print(f"   Validation: {len(X_val)} samples")
        print(f"   Test: {len(X_test)} samples")
        print(f"   Features: {len(feature_cols)}")
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Save scaler
        joblib.dump(self.scaler, self.config.scaler_save_path)
        print(f"✅ Scaler saved to {self.config.scaler_save_path}")
        
        # Save feature names
        with open(self.config.feature_names_path, 'w') as f:
            json.dump(feature_cols, f, indent=2)
        
        return (X_train_scaled, X_val_scaled, X_test_scaled,
                y_train, y_val, y_test, feature_cols)
    
    def create_model(self, model_type):
        """Create model instance"""
        
        if model_type == 'rf':
            return RandomForestClassifier(
                **self.config.rf_params,
                random_state=self.config.random_state,
                n_jobs=-1
            )
        
        elif model_type == 'gb':
            return GradientBoostingClassifier(
                **self.config.gb_params,
                random_state=self.config.random_state
            )
        
        elif model_type == 'lr':
            return LogisticRegression(
                **self.config.lr_params,
                random_state=self.config.random_state
            )
        
        elif model_type == 'svm':
            return SVC(
                **self.config.svm_params,
                random_state=self.config.random_state
            )
        
        elif model_type == 'xgb':
            try:
                import xgboost as xgb
                return xgb.XGBClassifier(
                    **self.config.xgb_params,
                    random_state=self.config.random_state,
                    use_label_encoder=False,
                    eval_metric='logloss'
                )
            except ImportError:
                print("   XGBoost not installed, using Gradient Boosting instead")
                return GradientBoostingClassifier(
                    **self.config.gb_params,
                    random_state=self.config.random_state
                )
        
        elif model_type == 'et':
            return ExtraTreesClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=self.config.random_state,
                n_jobs=-1
            )
        
        elif model_type == 'ada':
            return AdaBoostClassifier(
                n_estimators=100,
                random_state=self.config.random_state
            )
        
        elif model_type == 'knn':
            return KNeighborsClassifier(
                n_neighbors=5,
                weights='distance'
            )
        
        elif model_type == 'mlp':
            return MLPClassifier(
                hidden_layer_sizes=(100, 50),
                max_iter=1000,
                random_state=self.config.random_state
            )
        
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def grid_search(self, model_type, X_train, y_train):
        """Perform grid search for hyperparameter tuning"""
        
        if model_type not in self.config.grid_search_params:
            return None
        
        print(f"\n🔍 Performing grid search for {model_type}...")
        
        base_model = self.create_model(model_type)
        param_grid = self.config.grid_search_params[model_type]
        
        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=3,
            scoring='f1',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        print(f"\n   Best parameters: {grid_search.best_params_}")
        print(f"   Best score: {grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_
    
    def train_model(self, model_type, X_train, y_train, X_val, y_val):
        """Train a single model"""
        
        print(f"\n{'='*60}")
        print(f"Training {model_type.upper()} model")
        print(f"{'='*60}")
        
        # Grid search if enabled
        if self.config.do_grid_search and model_type in self.config.grid_search_params:
            model = self.grid_search(model_type, X_train, y_train)
        else:
            model = self.create_model(model_type)
        
        # Train
        start_time = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        # Predict
        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)
        
        # Get probabilities for AUC
        if hasattr(model, 'predict_proba'):
            y_train_proba = model.predict_proba(X_train)[:, 1]
            y_val_proba = model.predict_proba(X_val)[:, 1]
        else:
            y_train_proba = y_train_pred
            y_val_proba = y_val_pred
        
        # Calculate metrics
        train_metrics = self._calculate_metrics(y_train, y_train_pred, y_train_proba)
        val_metrics = self._calculate_metrics(y_val, y_val_pred, y_val_proba)
        
        print(f"\n📊 Training results:")
        print(f"   Train - Acc: {train_metrics['accuracy']:.4f}, "
              f"Prec: {train_metrics['precision']:.4f}, "
              f"Rec: {train_metrics['recall']:.4f}, "
              f"F1: {train_metrics['f1']:.4f}")
        print(f"   Val   - Acc: {val_metrics['accuracy']:.4f}, "
              f"Prec: {val_metrics['precision']:.4f}, "
              f"Rec: {val_metrics['recall']:.4f}, "
              f"F1: {val_metrics['f1']:.4f}")
        print(f"   Training time: {train_time:.2f}s")
        
        # Cross-validation
        cv_scores = cross_val_score(
            model, X_train, y_train, 
            cv=self.config.cv_folds, 
            scoring='f1'
        )
        print(f"   Cross-val F1: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Store results
        self.models[model_type] = model
        self.results[model_type] = {
            'train': train_metrics,
            'val': val_metrics,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'train_time': train_time
        }
        
        return model, val_metrics
    
    def _calculate_metrics(self, y_true, y_pred, y_proba=None):
        """Calculate classification metrics"""
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0)
        }
        
        if y_proba is not None and len(np.unique(y_true)) > 1:
            try:
                metrics['auc'] = roc_auc_score(y_true, y_proba)
            except:
                metrics['auc'] = 0.0
        
        return metrics
    
    def train_ensemble(self, X_train, y_train, X_val, y_val):
        """Train voting ensemble of best models"""
        
        print(f"\n{'='*60}")
        print("Training Ensemble Model")
        print(f"{'='*60}")
        
        # Select top models
        top_models = []
        model_scores = []
        
        for model_type, results in self.results.items():
            if model_type != 'ensemble':
                top_models.append((model_type, self.models[model_type]))
                model_scores.append(results['val']['f1'])
        
        # Sort by validation F1
        sorted_models = [m for _, m in sorted(
            zip(model_scores, top_models), 
            key=lambda x: x[0], 
            reverse=True
        )]
        
        # Take top 3
        estimators = [(m[0], m[1]) for m in sorted_models[:3]]
        
        # Create ensemble
        ensemble = VotingClassifier(
            estimators=estimators,
            voting='soft'
        )
        
        # Train
        start_time = time.time()
        ensemble.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        # Predict
        y_train_pred = ensemble.predict(X_train)
        y_val_pred = ensemble.predict(X_val)
        y_train_proba = ensemble.predict_proba(X_train)[:, 1]
        y_val_proba = ensemble.predict_proba(X_val)[:, 1]
        
        # Calculate metrics
        train_metrics = self._calculate_metrics(y_train, y_train_pred, y_train_proba)
        val_metrics = self._calculate_metrics(y_val, y_val_pred, y_val_proba)
        
        print(f"\n📊 Ensemble results:")
        print(f"   Train - Acc: {train_metrics['accuracy']:.4f}, "
              f"F1: {train_metrics['f1']:.4f}")
        print(f"   Val   - Acc: {val_metrics['accuracy']:.4f}, "
              f"F1: {val_metrics['f1']:.4f}")
        print(f"   Training time: {train_time:.2f}s")
        
        # Store
        self.models['ensemble'] = ensemble
        self.results['ensemble'] = {
            'train': train_metrics,
            'val': val_metrics,
            'train_time': train_time,
            'estimators': [e[0] for e in estimators]
        }
        
        return ensemble, val_metrics
    
    def evaluate_on_test(self, X_test, y_test):
        """Evaluate all models on test set"""
        
        print(f"\n{'='*60}")
        print("EVALUATING ON TEST SET")
        print(f"{'='*60}")
        
        best_model = None
        best_f1 = 0
        
        for model_type, model in self.models.items():
            print(f"\n📊 {model_type.upper()} on test set:")
            
            # Predict
            y_pred = model.predict(X_test)
            
            if hasattr(model, 'predict_proba'):
                y_proba = model.predict_proba(X_test)[:, 1]
            else:
                y_proba = y_pred
            
            # Metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            
            try:
                auc = roc_auc_score(y_test, y_proba)
            except:
                auc = 0.0
            
            print(f"   Accuracy:  {accuracy:.4f}")
            print(f"   Precision: {precision:.4f}")
            print(f"   Recall:    {recall:.4f}")
            print(f"   F1 Score:  {f1:.4f}")
            print(f"   AUC-ROC:   {auc:.4f}")
            
            # Store test results
            self.results[model_type]['test'] = {
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1': float(f1),
                'auc': float(auc)
            }
            
            if f1 > best_f1:
                best_f1 = f1
                best_model = model_type
        
        return best_model
    
    def plot_feature_importance(self, model_type='rf'):
        """Plot feature importance for tree-based models"""
        
        if model_type not in self.models:
            print(f"Model {model_type} not found")
            return
        
        model = self.models[model_type]
        
        if not hasattr(model, 'feature_importances_'):
            print(f"Model {model_type} doesn't have feature_importances_")
            return
        
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:20]  # Top 20
        
        plt.figure(figsize=(12, 8))
        plt.title(f'Top 20 Feature Importances - {model_type.upper()}')
        plt.barh(range(len(indices)), importances[indices])
        plt.yticks(range(len(indices)), [self.feature_names[i] for i in indices])
        plt.xlabel('Importance')
        plt.tight_layout()
        
        # Save
        plot_path = self.config.save_dir / f'feature_importance_{model_type}.png'
        plt.savefig(plot_path)
        plt.close()
        
        print(f"✅ Feature importance plot saved to {plot_path}")
    
    def plot_confusion_matrix(self, model_type, X_test, y_test):
        """Plot confusion matrix"""
        
        if model_type not in self.models:
            return
        
        model = self.models[model_type]
        y_pred = model.predict(X_test)
        
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Legitimate', 'Phishing'],
                   yticklabels=['Legitimate', 'Phishing'])
        plt.title(f'Confusion Matrix - {model_type.upper()}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # Save
        plot_path = self.config.save_dir / f'confusion_matrix_{model_type}.png'
        plt.savefig(plot_path)
        plt.close()
        
        print(f"✅ Confusion matrix saved to {plot_path}")
    
    def plot_roc_curves(self, X_test, y_test):
        """Plot ROC curves for all models"""
        
        plt.figure(figsize=(10, 8))
        
        for model_type, model in self.models.items():
            if hasattr(model, 'predict_proba'):
                y_proba = model.predict_proba(X_test)[:, 1]
                fpr, tpr, _ = roc_curve(y_test, y_proba)
                auc = roc_auc_score(y_test, y_proba)
                
                plt.plot(fpr, tpr, label=f'{model_type.upper()} (AUC = {auc:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves - URL Classifier')
        plt.legend()
        plt.grid(True)
        
        # Save
        plot_path = self.config.save_dir / 'roc_curves.png'
        plt.savefig(plot_path)
        plt.close()
        
        print(f"✅ ROC curves saved to {plot_path}")
    
    def save_best_model(self, best_model_type):
        """Save the best model"""
        
        best_model = self.models[best_model_type]
        
        # Save model
        joblib.dump(best_model, self.config.model_save_path)
        print(f"\n✅ Best model ({best_model_type}) saved to {self.config.model_save_path}")
        
        # Save results
        with open(self.config.results_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"✅ Results saved to {self.config.results_path}")
        
        return best_model


# ============================================
# MAIN TRAINING FUNCTION
# ============================================

def main():
    parser = argparse.ArgumentParser(description='Train URL Phishing Classifier')
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--data_file', type=str, help='Path to data CSV file')
    parser.add_argument('--generate', type=int, help='Generate synthetic dataset with N samples')
    parser.add_argument('--model_types', type=str, nargs='+', 
                       default=['rf', 'gb', 'lr', 'svm', 'ensemble'],
                       help='Model types to train')
    parser.add_argument('--no_grid', action='store_true', help='Disable grid search')
    parser.add_argument('--output_dir', type=str, help='Output directory')
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        with open(args.config, 'r') as f:
            config_dict = json.load(f)
        config = Config(**config_dict)
    else:
        config = Config()
    
    # Override with command line arguments
    if args.model_types:
        config.model_types = args.model_types
    
    if args.no_grid:
        config.do_grid_search = False
    
    if args.output_dir:
        config.save_dir = Path(args.output_dir)
        config.model_save_path = config.save_dir / 'url_classifier.pkl'
        config.scaler_save_path = config.save_dir / 'url_scaler.pkl'
        config.feature_names_path = config.save_dir / 'url_features.json'
        config.results_path = config.save_dir / 'url_classifier_results.json'
    
    # Create dataset
    generator = URLDatasetGenerator(config)
    
    if args.generate:
        # Generate synthetic dataset
        df = generator.generate_synthetic_dataset(args.generate)
        
        # Save
        config.data_dir.mkdir(exist_ok=True)
        generator.save_dataset(df, config.data_dir / 'synthetic_dataset.csv')
        
    elif args.data_file:
        # Load from file
        df = generator.load_from_csv(args.data_file)
    else:
        # Generate default dataset
        df = generator.generate_synthetic_dataset(20000)
        config.data_dir.mkdir(exist_ok=True)
        generator.save_dataset(df, config.data_dir / 'default_dataset.csv')
    
    # Train models
    trainer = URLModelTrainer(config)
    
    # Prepare data
    X_train, X_val, X_test, y_train, y_val, y_test, feature_names = trainer.prepare_data(df)
    
    # Train individual models
    for model_type in config.model_types:
        if model_type == 'ensemble':
            continue
        trainer.train_model(model_type, X_train, y_train, X_val, y_val)
    
    # Train ensemble if requested
    if 'ensemble' in config.model_types:
        trainer.train_ensemble(X_train, y_train, X_val, y_val)
    
    # Evaluate on test set
    best_model = trainer.evaluate_on_test(X_test, y_test)
    
    # Generate plots
    if 'rf' in trainer.models:
        trainer.plot_feature_importance('rf')
    
    if best_model:
        trainer.plot_confusion_matrix(best_model, X_test, y_test)
    
    trainer.plot_roc_curves(X_test, y_test)
    
    # Save best model
    trainer.save_best_model(best_model)
    
    print(f"\n{'='*60}")
    print("TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"Best model: {best_model}")
    print(f"Results saved to: {config.save_dir}")


if __name__ == "__main__":
    main()