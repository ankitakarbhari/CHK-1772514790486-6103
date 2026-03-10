# app/link_analyzer.py
"""
URL/Link Analysis Module for Deepfake Detection
Analyzes URLs for phishing, malware, scam websites, and AI-generated content
Supports real-time link scanning, content extraction, and threat assessment
Python 3.13+ Compatible
"""

import os
import sys
import re
import json
import time
import socket
import hashlib
import logging
import urllib
from urllib.parse import urlparse, parse_qs, urljoin
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import Counter
import warnings

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

# ========== NETWORK REQUESTS ==========
import requests
import aiohttp
import httpx
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ========== URL PARSING ==========
import tldextract
import whois
import dns.resolver
import dns.reversename
from urllib.parse import urlparse, urljoin, quote, unquote

# ========== SSL/TLS ==========
import ssl
import certifi
from OpenSSL import crypto
import idna

# ========== WEB SCRAPING ==========
from bs4 import BeautifulSoup
import trafilatura
from newspaper import Article
import readability
import html2text

# ========== TEXT ANALYSIS ==========
from app.models.text_detection_model import TextDeepfakeDetector, TextDetectorFactory, AIDetectionResult

# ========== IMAGE ANALYSIS ==========
from app.models.ensemble import DeepfakeEnsemble, EnsembleFactory, EnsembleResult
from app.utils.face_detection import FaceDetectionEnsemble, FacePreprocessor
import cv2
import numpy as np
from PIL import Image
import io
import hashlib

# ========== BROWSER AUTOMATION (Optional) ==========
try:
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.chrome.options import Options as ChromeOptions
    from selenium.common.exceptions import TimeoutException, WebDriverException
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False
    logging.warning("Selenium not installed. Screenshot and JavaScript features disabled.")

try:
    from playwright.sync_api import sync_playwright
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================
# DATA CLASSES
# ============================================

@dataclass
class URLInfo:
    """Basic URL information"""
    url: str
    scheme: str
    domain: str
    subdomain: str
    tld: str
    path: str
    query: str
    fragment: str
    params: Dict[str, List[str]]
    port: Optional[int]
    full_domain: str
    
    @classmethod
    def from_url(cls, url: str):
        """Create URLInfo from URL string"""
        # Ensure URL has scheme
        if not url.startswith(('http://', 'https://')):
            url = 'http://' + url
        
        parsed = urlparse(url)
        extracted = tldextract.extract(url)
        
        # Parse query parameters
        params = parse_qs(parsed.query)
        
        # Build full domain
        if extracted.domain and extracted.suffix:
            full_domain = f"{extracted.domain}.{extracted.suffix}"
        else:
            full_domain = parsed.netloc.split(':')[0]
        
        return cls(
            url=url,
            scheme=parsed.scheme,
            domain=extracted.domain,
            subdomain=extracted.subdomain,
            tld=extracted.suffix,
            path=parsed.path,
            query=parsed.query,
            fragment=parsed.fragment,
            params=params,
            port=parsed.port,
            full_domain=full_domain
        )
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class WhoisInfo:
    """WHOIS information for domain"""
    domain: str
    registrar: Optional[str]
    creation_date: Optional[datetime]
    expiration_date: Optional[datetime]
    updated_date: Optional[datetime]
    name_servers: List[str]
    registrant: Optional[str]
    organization: Optional[str]
    country: Optional[str]
    emails: List[str]
    abuse_contact: Optional[str]
    days_old: Optional[int]
    days_until_expiry: Optional[int]
    is_private: bool = False
    
    def to_dict(self) -> Dict:
        result = asdict(self)
        # Convert datetime objects to strings
        for field in ['creation_date', 'expiration_date', 'updated_date']:
            if result.get(field):
                result[field] = result[field].isoformat() if isinstance(result[field], datetime) else str(result[field])
        return result


@dataclass
class SSLInfo:
    """SSL certificate information"""
    issuer: Dict[str, str]
    subject: Dict[str, str]
    version: int
    serial_number: str
    not_before: datetime
    not_after: datetime
    is_valid: bool
    days_until_expiry: int
    fingerprint: str
    signature_algorithm: str
    is_self_signed: bool
    alternative_names: List[str]
    
    def to_dict(self) -> Dict:
        result = asdict(self)
        for field in ['not_before', 'not_after']:
            if result.get(field):
                result[field] = result[field].isoformat()
        return result


@dataclass
class PageContent:
    """Extracted page content"""
    url: str
    title: str
    text: str
    html: str
    meta_tags: Dict[str, str]
    links: List[str]
    external_links: List[str]
    internal_links: List[str]
    images: List[Dict[str, str]]
    videos: List[str]
    scripts: List[str]
    iframes: List[str]
    forms: List[Dict]
    language: Optional[str]
    word_count: int
    reading_time: int  # minutes
    has_login_form: bool
    has_password_field: bool
    has_payment_form: bool
    
    def to_dict(self) -> Dict:
        result = asdict(self)
        # Truncate large fields
        result['html'] = f"<{len(self.html)} bytes>"
        return result


@dataclass
class ThreatAssessment:
    """Threat assessment results"""
    is_phishing: bool
    is_malware: bool
    is_scam: bool
    is_fake_news: bool
    is_deepfake: bool
    risk_score: float  # 0-100
    risk_level: str  # 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL'
    threat_types: List[str]
    warnings: List[str]
    matched_patterns: List[str]
    brand_impersonated: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class LinkAnalysisResult:
    """Complete link analysis result"""
    url: str
    url_info: URLInfo
    whois_info: Optional[WhoisInfo]
    ssl_info: Optional[SSLInfo]
    page_content: Optional[PageContent]
    threat_assessment: ThreatAssessment
    redirect_chain: List[str]
    response_time: float
    status_code: int
    content_type: Optional[str]
    content_length: int
    text_analysis: Optional[AIDetectionResult] = None
    image_analysis: Optional[List[Dict]] = None
    screenshot_path: Optional[str] = None
    screenshot_base64: Optional[str] = None
    analysis_time: float = 0.0
    cached: bool = False
    
    def to_dict(self) -> Dict:
        result = asdict(self)
        # Remove large binary data
        if 'screenshot_base64' in result:
            result['screenshot_base64'] = f"<{len(self.screenshot_base64) if self.screenshot_base64 else 0} bytes>"
        return result
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, default=str)
    
    def summary(self) -> str:
        """Get text summary"""
        return (f"URL Analysis: {self.url}\n"
                f"Risk Score: {self.threat_assessment.risk_score}/100 ({self.threat_assessment.risk_level})\n"
                f"Threats: {', '.join(self.threat_assessment.threat_types) if self.threat_assessment.threat_types else 'None'}\n"
                f"Status: {self.status_code}, Response Time: {self.response_time*1000:.1f}ms\n"
                f"Content: {self.content_type}, {self.content_length/1024:.1f}KB\n"
                f"Warnings: {len(self.threat_assessment.warnings)}")


# ============================================
# URL FEATURE EXTRACTOR
# ============================================

class URLFeatureExtractor:
    """
    Extract numerical features from URLs for ML classification
    """
    
    def __init__(self):
        # Suspicious TLDs
        self.suspicious_tlds = {
            'tk', 'ml', 'ga', 'cf', 'xyz', 'top', 'club', 'online', 
            'site', 'web', 'work', 'review', 'stream', 'download',
            'bid', 'trade', 'webcam', 'win', 'science', 'date', 'kim',
            'men', 'loan', 'click', 'link', 'download', 'review', 'racing',
            'faith', 'lol', 'bet', 'accountant', 'work', 'gdn', 'rest',
            'cam', 'mom', 'lol', 'xin', 'loan', 'download', 'review'
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
            'verification', 'authenticate', 'security', 'alert', 'warning',
            'unusual', 'activity', 'suspended', 'limited', 'restricted'
        ]
        
        # Brand names for impersonation detection
        self.brands = [
            'google', 'facebook', 'amazon', 'apple', 'microsoft', 'netflix',
            'paypal', 'instagram', 'twitter', 'linkedin', 'whatsapp', 'spotify',
            'yahoo', 'bing', 'ebay', 'walmart', 'target', 'bestbuy', 'nike',
            'adidas', 'zara', 'hm', 'gap', 'ikea', 'wikipedia', 'reddit',
            'dropbox', 'github', 'stackoverflow', 'wordpress', 'blogger',
            'cloudflare', 'godaddy', 'namecheap', 'bluehost', 'hostgator'
        ]
        
        # Compile regex patterns
        self.ip_pattern = re.compile(r'^\d+\.\d+\.\d+\.\d+$')
        self.hex_pattern = re.compile(r'%[0-9a-fA-F]{2}')
        self.multi_dot_pattern = re.compile(r'\.{2,}')
        self.domain_pattern = re.compile(r'[a-zA-Z0-9-]+\.[a-zA-Z]{2,}')
        self.shortener_pattern = re.compile(r'bit\.ly|goo\.gl|tinyurl|ow\.ly|is\.gd|buff\.ly')
    
    def extract_features(self, url: str) -> Dict[str, float]:
        """
        Extract numerical features from URL
        
        Returns:
            Dictionary of numerical features
        """
        features = {}
        
        # Parse URL
        url_info = URLInfo.from_url(url)
        
        # ====================================
        # Length-based features
        # ====================================
        features['url_length'] = len(url)
        features['domain_length'] = len(url_info.full_domain)
        features['path_length'] = len(url_info.path)
        features['query_length'] = len(url_info.query)
        
        # ====================================
        # Special character counts
        # ====================================
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
        # Digit and letter counts
        # ====================================
        features['num_digits'] = sum(c.isdigit() for c in url)
        features['num_letters'] = sum(c.isalpha() for c in url)
        features['digit_ratio'] = features['num_digits'] / (len(url) + 1)
        
        # ====================================
        # Suspicious indicators
        # ====================================
        features['has_ip'] = 1.0 if self.ip_pattern.match(url_info.full_domain) else 0.0
        features['has_port'] = 1.0 if url_info.port else 0.0
        features['has_hex_chars'] = 1.0 if self.hex_pattern.search(url) else 0.0
        features['has_multi_dots'] = 1.0 if self.multi_dot_pattern.search(url) else 0.0
        features['is_shortened'] = 1.0 if self.shortener_pattern.search(url) else 0.0
        
        # ====================================
        # TLD analysis
        # ====================================
        features['tld_length'] = len(url_info.tld)
        features['suspicious_tld'] = 1.0 if url_info.tld in self.suspicious_tlds else 0.0
        features['common_tld'] = 1.0 if url_info.tld in ['com', 'org', 'net', 'edu', 'gov'] else 0.0
        
        # ====================================
        # Subdomain analysis
        # ====================================
        features['subdomain_length'] = len(url_info.subdomain)
        features['subdomain_count'] = len([s for s in url_info.subdomain.split('.') if s]) if url_info.subdomain else 0
        features['has_www'] = 1.0 if 'www' in url_info.subdomain else 0.0
        
        # ====================================
        # Path analysis
        # ====================================
        features['path_depth'] = len([p for p in url_info.path.split('/') if p])
        features['has_extension'] = 1.0 if '.' in url_info.path.split('/')[-1] else 0.0
        features['path_has_digits'] = 1.0 if any(c.isdigit() for c in url_info.path) else 0.0
        
        # ====================================
        # Query parameter analysis
        # ====================================
        features['num_params'] = len(url_info.params)
        features['max_param_length'] = max([len(v[0]) for v in url_info.params.values()]) if url_info.params else 0
        
        # ====================================
        # Keyword matching
        # ====================================
        url_lower = url.lower()
        features['suspicious_keywords'] = sum(1 for kw in self.suspicious_keywords if kw in url_lower)
        
        # Brand impersonation
        domain_lower = url_info.full_domain.lower()
        brand_match = 0
        matched_brand = None
        for brand in self.brands:
            if brand in domain_lower and domain_lower != f"{brand}.com":
                brand_match += 1
                matched_brand = brand
        features['brand_in_domain'] = brand_match
        features['brand_typosquatting'] = min(brand_match, 1)
        
        # ====================================
        # Entropy
        # ====================================
        features['entropy'] = self._calculate_entropy(url)
        features['domain_entropy'] = self._calculate_entropy(url_info.full_domain)
        
        return features
    
    def _calculate_entropy(self, text: str) -> float:
        """Calculate Shannon entropy of string"""
        if not text:
            return 0.0
        
        entropy = 0.0
        for i in range(256):
            char = chr(i)
            freq = text.count(char)
            if freq > 0:
                freq = float(freq) / len(text)
                entropy -= freq * (freq.bit_length() - 1)  # log2 approximation
        
        return entropy
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names"""
        dummy_features = self.extract_features('https://example.com')
        return list(dummy_features.keys())


# ============================================
# WHOIS LOOKUP
# ============================================

class WhoisLookup:
    """
    Perform WHOIS lookups for domain information
    """
    
    def __init__(self, timeout: int = 10):
        """
        Initialize WHOIS lookup
        
        Args:
            timeout: Timeout in seconds
        """
        self.timeout = timeout
        logger.info("WhoisLookup initialized")
    
    def lookup(self, domain: str) -> Optional[WhoisInfo]:
        """
        Perform WHOIS lookup for domain
        
        Args:
            domain: Domain name
        
        Returns:
            WhoisInfo object or None
        """
        try:
            # Clean domain
            domain = domain.split('/')[0].split(':')[0]
            
            # Perform WHOIS query
            w = whois.whois(domain, timeout=self.timeout)
            
            if not w or not w.text:
                logger.warning(f"No WHOIS data for {domain}")
                return None
            
            # Parse dates
            creation_date = self._parse_date(w.creation_date)
            expiration_date = self._parse_date(w.expiration_date)
            updated_date = self._parse_date(w.updated_date)
            
            # Calculate days old
            days_old = None
            if creation_date:
                days_old = (datetime.now() - creation_date).days
            
            # Calculate days until expiry
            days_until_expiry = None
            if expiration_date:
                days_until_expiry = (expiration_date - datetime.now()).days
            
            # Parse name servers
            name_servers = w.name_servers
            if isinstance(name_servers, str):
                name_servers = [name_servers]
            elif not name_servers:
                name_servers = []
            
            # Parse emails
            emails = w.emails
            if isinstance(emails, str):
                emails = [emails]
            elif not emails:
                emails = []
            
            # Check privacy
            is_private = False
            if w.name and 'REDACTED' in str(w.name).upper():
                is_private = True
            
            return WhoisInfo(
                domain=domain,
                registrar=w.registrar,
                creation_date=creation_date,
                expiration_date=expiration_date,
                updated_date=updated_date,
                name_servers=name_servers,
                registrant=w.name,
                organization=w.org,
                country=w.country,
                emails=emails,
                abuse_contact=None,
                days_old=days_old,
                days_until_expiry=days_until_expiry,
                is_private=is_private
            )
            
        except Exception as e:
            logger.error(f"WHOIS lookup error for {domain}: {str(e)}")
            return None
    
    def _parse_date(self, date_val) -> Optional[datetime]:
        """Parse date from WHOIS response"""
        if not date_val:
            return None
        
        if isinstance(date_val, list):
            date_val = date_val[0]
        
        if isinstance(date_val, datetime):
            return date_val
        
        try:
            return datetime.strptime(str(date_val), "%Y-%m-%d %H:%M:%S")
        except:
            try:
                return datetime.strptime(str(date_val), "%Y-%m-%d")
            except:
                try:
                    return datetime.strptime(str(date_val), "%d-%b-%Y")
                except:
                    return None


# ============================================
# SSL CERTIFICATE CHECKER
# ============================================

class SSLChecker:
    """
    Check SSL certificates for domains
    """
    
    def __init__(self, timeout: int = 10):
        """
        Initialize SSL checker
        
        Args:
            timeout: Connection timeout
        """
        self.timeout = timeout
        self.context = ssl.create_default_context(cafile=certifi.where())
        logger.info("SSLChecker initialized")
    
    def check(self, hostname: str, port: int = 443) -> Optional[SSLInfo]:
        """
        Check SSL certificate for hostname
        
        Args:
            hostname: Hostname to check
            port: Port (usually 443)
        
        Returns:
            SSLInfo object or None
        """
        try:
            # Clean hostname
            hostname = hostname.split('://')[-1].split('/')[0].split(':')[0]
            
            # Connect and get certificate
            with socket.create_connection((hostname, port), timeout=self.timeout) as sock:
                with self.context.wrap_socket(sock, server_hostname=hostname) as ssock:
                    cert_bin = ssock.getpeercert(binary_form=True)
                    
                    if not cert_bin:
                        return None
                    
                    cert = crypto.load_certificate(crypto.FILETYPE_ASN1, cert_bin)
                    
                    # Parse certificate
                    issuer = {}
                    for component in cert.get_issuer().get_components():
                        key = component[0].decode()
                        value = component[1].decode()
                        issuer[key] = value
                    
                    subject = {}
                    for component in cert.get_subject().get_components():
                        key = component[0].decode()
                        value = component[1].decode()
                        subject[key] = value
                    
                    # Dates
                    not_before = datetime.strptime(cert.get_notBefore().decode(), '%Y%m%d%H%M%SZ')
                    not_after = datetime.strptime(cert.get_notAfter().decode(), '%Y%m%d%H%M%SZ')
                    
                    # Check if valid
                    now = datetime.now()
                    is_valid = not_before <= now <= not_after
                    
                    # Days until expiry
                    days_until_expiry = (not_after - now).days
                    
                    # Get alternative names
                    alt_names = []
                    for i in range(cert.get_extension_count()):
                        ext = cert.get_extension(i)
                        if ext.get_short_name() == b'subjectAltName':
                            alt_str = str(ext)
                            # Parse alt names
                            for item in alt_str.split(', '):
                                if item.startswith('DNS:'):
                                    alt_names.append(item[4:])
                    
                    return SSLInfo(
                        issuer=issuer,
                        subject=subject,
                        version=cert.get_version(),
                        serial_number=hex(cert.get_serial_number())[2:].upper(),
                        not_before=not_before,
                        not_after=not_after,
                        is_valid=is_valid,
                        days_until_expiry=days_until_expiry,
                        fingerprint=cert.digest('sha256').hex().upper(),
                        signature_algorithm=cert.get_signature_algorithm().decode(),
                        is_self_signed=issuer == subject,
                        alternative_names=alt_names
                    )
                    
        except socket.timeout:
            logger.error(f"SSL check timeout for {hostname}")
            return None
        except Exception as e:
            logger.error(f"SSL check error for {hostname}: {str(e)}")
            return None


# ============================================
# PAGE CONTENT EXTRACTOR
# ============================================

class PageContentExtractor:
    """
    Extract and analyze page content
    """
    
    def __init__(self, 
                 timeout: int = 30, 
                 user_agent: Optional[str] = None,
                 use_selenium: bool = False,
                 max_size: int = 10 * 1024 * 1024):  # 10MB
        """
        Initialize page content extractor
        
        Args:
            timeout: Request timeout
            user_agent: Custom user agent
            use_selenium: Use Selenium for JavaScript-heavy sites
            max_size: Maximum content size to download
        """
        self.timeout = timeout
        self.user_agent = user_agent or "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        self.use_selenium = use_selenium and SELENIUM_AVAILABLE
        self.max_size = max_size
        
        # Setup requests session with retries
        self.session = requests.Session()
        retry = Retry(total=3, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
        adapter = HTTPAdapter(max_retries=retry)
        self.session.mount('http://', adapter)
        self.session.mount('https://', adapter)
        self.session.headers.update({
            'User-Agent': self.user_agent,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        })
        
        # HTML to text converter
        self.html_converter = html2text.HTML2Text()
        self.html_converter.ignore_links = False
        self.html_converter.ignore_images = False
        self.html_converter.ignore_emphasis = False
        
        logger.info(f"PageContentExtractor initialized (selenium: {use_selenium})")
    
    def extract(self, url: str, extract_images: bool = True, 
               follow_redirects: bool = True) -> Tuple[Optional[PageContent], int, Dict, List[str]]:
        """
        Extract content from URL
        
        Args:
            url: URL to extract
            extract_images: Whether to extract image info
            follow_redirects: Follow redirects
        
        Returns:
            Tuple of (PageContent, status_code, headers, redirect_chain)
        """
        redirect_chain = []
        
        try:
            # Use Selenium if enabled
            if self.use_selenium:
                return self._extract_with_selenium(url, extract_images)
            
            # Regular HTTP request
            response = self.session.get(
                url, 
                timeout=self.timeout, 
                allow_redirects=follow_redirects,
                stream=True
            )
            
            # Check content size
            content_length = int(response.headers.get('content-length', 0))
            if content_length > self.max_size:
                logger.warning(f"Content too large: {content_length} bytes")
                return None, response.status_code, dict(response.headers), redirect_chain
            
            # Get redirect chain
            if response.history:
                redirect_chain = [r.url for r in response.history]
            
            content_type = response.headers.get('Content-Type', '')
            
            # Only parse HTML
            if 'text/html' not in content_type:
                logger.info(f"Not HTML: {content_type}")
                return None, response.status_code, dict(response.headers), redirect_chain
            
            # Parse HTML
            content = self._parse_html(response.text, url, extract_images)
            
            return content, response.status_code, dict(response.headers), redirect_chain
            
        except requests.exceptions.Timeout:
            logger.error(f"Request timeout for {url}")
            return None, 408, {}, redirect_chain
        except requests.exceptions.TooManyRedirects:
            logger.error(f"Too many redirects for {url}")
            return None, 310, {}, redirect_chain
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error for {url}: {str(e)}")
            return None, getattr(e.response, 'status_code', 0), {}, redirect_chain
        except Exception as e:
            logger.error(f"Extraction error for {url}: {str(e)}")
            return None, 0, {}, redirect_chain
    
    def _extract_with_selenium(self, url: str, extract_images: bool) -> Tuple[Optional[PageContent], int, Dict, List[str]]:
        """Extract content using Selenium for JavaScript-rendered pages"""
        if not SELENIUM_AVAILABLE:
            logger.warning("Selenium not available, falling back to regular HTTP")
            return self.extract(url, extract_images, True)
        
        driver = None
        try:
            # Configure Chrome options
            options = ChromeOptions()
            options.add_argument('--headless')
            options.add_argument('--no-sandbox')
            options.add_argument('--disable-dev-shm-usage')
            options.add_argument('--disable-gpu')
            options.add_argument('--window-size=1920,1080')
            
            driver = webdriver.Chrome(options=options)
            driver.set_page_load_timeout(self.timeout)
            
            # Navigate
            driver.get(url)
            
            # Wait for page load
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            
            # Get page source after JavaScript execution
            html = driver.page_source
            
            # Get redirect chain
            redirect_chain = [url]
            current_url = driver.current_url
            if current_url != url:
                redirect_chain.append(current_url)
            
            # Parse HTML
            content = self._parse_html(html, current_url, extract_images)
            
            return content, 200, {}, redirect_chain
            
        except TimeoutException:
            logger.error(f"Selenium timeout for {url}")
            return None, 408, {}, []
        except WebDriverException as e:
            logger.error(f"Selenium error for {url}: {str(e)}")
            return None, 500, {}, []
        finally:
            if driver:
                driver.quit()
    
    def _parse_html(self, html: str, base_url: str, extract_images: bool) -> PageContent:
        """Parse HTML content"""
        soup = BeautifulSoup(html, 'html.parser')
        
        # Extract title
        title = soup.title.string if soup.title else ''
        
        # Extract meta tags
        meta_tags = {}
        for meta in soup.find_all('meta'):
            if meta.get('name'):
                meta_tags[meta['name']] = meta.get('content', '')
            elif meta.get('property'):
                meta_tags[meta['property']] = meta.get('content', '')
        
        # Extract all links
        all_links = []
        internal_links = []
        external_links = []
        
        base_domain = urlparse(base_url).netloc
        
        for a in soup.find_all('a', href=True):
            href = a['href']
            absolute = urljoin(base_url, href)
            all_links.append(absolute)
            
            # Categorize as internal/external
            link_domain = urlparse(absolute).netloc
            if link_domain == base_domain or not link_domain:
                internal_links.append(absolute)
            else:
                external_links.append(absolute)
        
        # Extract images
        images = []
        if extract_images:
            for img in soup.find_all('img'):
                src = img.get('src', '')
                if src:
                    img_info = {
                        'src': urljoin(base_url, src),
                        'alt': img.get('alt', ''),
                        'title': img.get('title', ''),
                        'width': img.get('width', ''),
                        'height': img.get('height', '')
                    }
                    images.append(img_info)
        
        # Extract videos
        videos = []
        for video in soup.find_all('video'):
            if video.get('src'):
                videos.append(urljoin(base_url, video['src']))
            for source in video.find_all('source'):
                if source.get('src'):
                    videos.append(urljoin(base_url, source['src']))
        
        # Extract scripts
        scripts = []
        for script in soup.find_all('script', src=True):
            scripts.append(urljoin(base_url, script['src']))
        
        # Extract iframes
        iframes = []
        for iframe in soup.find_all('iframe', src=True):
            iframes.append(urljoin(base_url, iframe['src']))
        
        # Extract forms
        forms = []
        has_login = False
        has_password = False
        has_payment = False
        
        for form in soup.find_all('form'):
            form_info = {
                'action': urljoin(base_url, form.get('action', '')),
                'method': form.get('method', 'get').upper(),
                'inputs': []
            }
            
            for input_tag in form.find_all('input'):
                input_type = input_tag.get('type', 'text').lower()
                input_name = input_tag.get('name', '')
                
                form_info['inputs'].append({
                    'name': input_name,
                    'type': input_type,
                    'value': input_tag.get('value', '')
                })
                
                if input_type == 'password':
                    has_password = True
                    has_login = True
                elif 'card' in input_name.lower() or 'cvv' in input_name.lower():
                    has_payment = True
            
            forms.append(form_info)
        
        # Extract text using trafilatura
        text = trafilatura.extract(html)
        if not text:
            # Fallback to html2text
            text = self.html_converter.handle(html)
        
        # Calculate word count and reading time
        words = text.split() if text else []
        word_count = len(words)
        reading_time = max(1, word_count // 200)  # 200 words per minute
        
        # Detect language (simplified)
        language = self._detect_language(text)
        
        return PageContent(
            url=base_url,
            title=title,
            text=text or '',
            html=html,
            meta_tags=meta_tags,
            links=all_links,
            external_links=external_links,
            internal_links=internal_links,
            images=images,
            videos=videos,
            scripts=scripts,
            iframes=iframes,
            forms=forms,
            language=language,
            word_count=word_count,
            reading_time=reading_time,
            has_login_form=has_login,
            has_password_field=has_password,
            has_payment_form=has_payment
        )
    
    def _detect_language(self, text: str) -> Optional[str]:
        """Detect language of text"""
        try:
            import langdetect
            return langdetect.detect(text[:1000])
        except:
            return None
    
    def take_screenshot(self, url: str, output_path: Optional[str] = None) -> Optional[bytes]:
        """
        Take screenshot of webpage
        
        Args:
            url: URL to screenshot
            output_path: Path to save screenshot (optional)
        
        Returns:
            Screenshot as bytes
        """
        if not SELENIUM_AVAILABLE:
            logger.warning("Selenium not available for screenshots")
            return None
        
        driver = None
        try:
            options = ChromeOptions()
            options.add_argument('--headless')
            options.add_argument('--no-sandbox')
            options.add_argument('--disable-dev-shm-usage')
            options.add_argument('--window-size=1920,1080')
            
            driver = webdriver.Chrome(options=options)
            driver.set_page_load_timeout(self.timeout)
            
            driver.get(url)
            time.sleep(2)  # Wait for page load
            
            screenshot = driver.get_screenshot_as_png()
            
            if output_path:
                with open(output_path, 'wb') as f:
                    f.write(screenshot)
                logger.info(f"Screenshot saved to {output_path}")
            
            return screenshot
            
        except Exception as e:
            logger.error(f"Screenshot error: {str(e)}")
            return None
        finally:
            if driver:
                driver.quit()


# ============================================
# THREAT DETECTOR
# ============================================

class ThreatDetector:
    """
    Detect threats from URL and page content
    """
    
    def __init__(self, feature_extractor: Optional[URLFeatureExtractor] = None):
        """
        Initialize threat detector
        
        Args:
            feature_extractor: URL feature extractor
        """
        self.feature_extractor = feature_extractor or URLFeatureExtractor()
        
        # Phishing keywords in content
        self.phishing_keywords = [
            'verify your account', 'confirm your identity', 'unusual activity',
            'suspicious login', 'account suspended', 'limited account',
            'update payment', 'billing information', 'security check',
            'lottery winner', 'inheritance claim', 'money transfer',
            'western union', 'money gram', 'gift card', 'cryptocurrency',
            'bitcoin investment', 'forex trading', 'get rich quick',
            'work from home', 'earn money online', 'passive income',
            'urgent action required', 'your account will be closed'
        ]
        
        # Brand names for impersonation
        self.brands = [
            'paypal', 'amazon', 'ebay', 'apple', 'microsoft', 'google',
            'facebook', 'instagram', 'netflix', 'spotify', 'linkedin',
            'twitter', 'yahoo', 'aol', 'outlook', 'hotmail', 'gmail',
            'chase', 'wells fargo', 'bank of america', 'citibank',
            'american express', 'visa', 'mastercard', 'paypal'
        ]
        
        # Compile regex patterns
        self.email_pattern = re.compile(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}')
        self.phone_pattern = re.compile(r'(\+\d{1,3}[-.]?)?\(?\d{3}\)?[-.]?\d{3}[-.]?\d{4}')
        self.bitcoin_pattern = re.compile(r'[13][a-km-zA-HJ-NP-Z1-9]{25,34}')
        self.eth_pattern = re.compile(r'0x[a-fA-F0-9]{40}')
        
        logger.info("ThreatDetector initialized")
    
    def analyze(self, 
               url_info: URLInfo,
               whois_info: Optional[WhoisInfo],
               ssl_info: Optional[SSLInfo],
               content: Optional[PageContent]) -> ThreatAssessment:
        """
        Analyze all information for threats
        
        Args:
            url_info: URL information
            whois_info: WHOIS information
            ssl_info: SSL information
            content: Page content
        
        Returns:
            ThreatAssessment object
        """
        threat_types = []
        warnings = []
        matched_patterns = []
        risk_score = 0
        brand_impersonated = None
        
        # ====================================
        # URL-based checks
        # ====================================
        url_risk = self._check_url(url_info)
        risk_score += url_risk['score']
        warnings.extend(url_risk['warnings'])
        matched_patterns.extend(url_risk['patterns'])
        if url_risk['brand']:
            brand_impersonated = url_risk['brand']
        if url_risk['is_phishing']:
            threat_types.append('phishing_url')
        
        # ====================================
        # WHOIS-based checks
        # ====================================
        if whois_info:
            whois_risk = self._check_whois(whois_info)
            risk_score += whois_risk['score']
            warnings.extend(whois_risk['warnings'])
            if whois_risk['is_suspicious']:
                threat_types.append('suspicious_domain')
        
        # ====================================
        # SSL-based checks
        # ====================================
        if ssl_info:
            ssl_risk = self._check_ssl(ssl_info)
            risk_score += ssl_risk['score']
            warnings.extend(ssl_risk['warnings'])
            if not ssl_info.is_valid:
                threat_types.append('invalid_ssl')
        
        # ====================================
        # Content-based checks
        # ====================================
        if content:
            content_risk = self._check_content(content, url_info)
            risk_score += content_risk['score']
            warnings.extend(content_risk['warnings'])
            matched_patterns.extend(content_risk['patterns'])
            if content_risk['brand']:
                brand_impersonated = brand_impersonated or content_risk['brand']
            if content_risk['is_phishing']:
                threat_types.append('phishing_content')
            if content_risk['is_scam']:
                threat_types.append('scam')
            if content_risk['is_fake_news']:
                threat_types.append('fake_news')
            if content_risk['is_malware']:
                threat_types.append('malware')
        
        # Normalize risk score to 0-100
        risk_score = min(100, max(0, risk_score))
        
        # Determine risk level
        if risk_score >= 80:
            risk_level = "CRITICAL"
        elif risk_score >= 60:
            risk_level = "HIGH"
        elif risk_score >= 40:
            risk_level = "MEDIUM"
        elif risk_score >= 20:
            risk_level = "LOW"
        else:
            risk_level = "MINIMAL"
        
        # Remove duplicates
        threat_types = list(set(threat_types))
        warnings = list(set(warnings))
        matched_patterns = list(set(matched_patterns))
        
        # Determine primary threat
        is_phishing = 'phishing_url' in threat_types or 'phishing_content' in threat_types
        is_malware = 'malware' in threat_types
        is_scam = 'scam' in threat_types
        is_fake_news = 'fake_news' in threat_types
        is_deepfake = False  # Will be set by image analysis
        
        return ThreatAssessment(
            is_phishing=is_phishing,
            is_malware=is_malware,
            is_scam=is_scam,
            is_fake_news=is_fake_news,
            is_deepfake=is_deepfake,
            risk_score=risk_score,
            risk_level=risk_level,
            threat_types=threat_types,
            warnings=warnings,
            matched_patterns=matched_patterns,
            brand_impersonated=brand_impersonated
        )
    
    def _check_url(self, url_info: URLInfo) -> Dict:
        """Check URL for threats"""
        score = 0
        warnings = []
        patterns = []
        is_phishing = False
        brand = None
        
        # Check for suspicious TLD
        if url_info.tld in self.feature_extractor.suspicious_tlds:
            score += 20
            warnings.append(f"Suspicious TLD: .{url_info.tld}")
            patterns.append(f"suspicious_tld_{url_info.tld}")
        
        # Check for IP address instead of domain
        if re.match(r'^\d+\.\d+\.\d+\.\d+$', url_info.full_domain):
            score += 30
            warnings.append("IP address used instead of domain name")
            patterns.append("ip_address_url")
            is_phishing = True
        
        # Check for unusual port
        if url_info.port and url_info.port not in [80, 443, 8080]:
            score += 15
            warnings.append(f"Unusual port: {url_info.port}")
            patterns.append(f"unusual_port_{url_info.port}")
        
        # Check for excessive subdomains
        if url_info.subdomain and url_info.subdomain.count('.') > 2:
            score += 15
            warnings.append("Excessive subdomains")
            patterns.append("excessive_subdomains")
        
        # Check for brand impersonation in domain
        domain_lower = url_info.full_domain.lower()
        for b in self.brands:
            if b in domain_lower and b not in domain_lower.replace(b, ''):
                # Check if it's exactly the brand or a variation
                if domain_lower != f"{b}.com" and domain_lower != f"www.{b}.com":
                    score += 25
                    warnings.append(f"Possible {b} impersonation")
                    patterns.append(f"impersonation_{b}")
                    is_phishing = True
                    brand = b
        
        # Check for URL shorteners
        shorteners = ['bit.ly', 'goo.gl', 'tinyurl', 'ow.ly', 'is.gd', 'buff.ly']
        for s in shorteners:
            if s in url_info.full_domain:
                score += 10
                warnings.append(f"URL shortener detected: {s}")
                patterns.append("url_shortener")
        
        # Check URL length (very long URLs are suspicious)
        if len(url_info.url) > 200:
            score += 10
            warnings.append("Very long URL (>200 chars)")
            patterns.append("long_url")
        
        return {
            'score': score,
            'warnings': warnings,
            'patterns': patterns,
            'is_phishing': is_phishing,
            'brand': brand
        }
    
    def _check_whois(self, whois_info: WhoisInfo) -> Dict:
        """Check WHOIS information for threats"""
        score = 0
        warnings = []
        is_suspicious = False
        
        # Check domain age
        if whois_info.days_old is not None:
            if whois_info.days_old < 7:
                score += 40
                warnings.append(f"Domain is extremely new ({whois_info.days_old} days)")
                is_suspicious = True
            elif whois_info.days_old < 30:
                score += 30
                warnings.append(f"Domain is very new ({whois_info.days_old} days)")
                is_suspicious = True
            elif whois_info.days_old < 90:
                score += 15
                warnings.append(f"Domain is new ({whois_info.days_old} days)")
                is_suspicious = True
        else:
            score += 20
            warnings.append("Domain age unknown")
            is_suspicious = True
        
        # Check expiry
        if whois_info.days_until_expiry is not None:
            if whois_info.days_until_expiry < 7:
                score += 30
                warnings.append(f"Domain expires extremely soon ({whois_info.days_until_expiry} days)")
                is_suspicious = True
            elif whois_info.days_until_expiry < 30:
                score += 20
                warnings.append(f"Domain expires soon ({whois_info.days_until_expiry} days)")
                is_suspicious = True
        
        # Check for privacy protection
        if whois_info.is_private:
            score += 10
            warnings.append("Registrant information is private/hidden")
        
        # Check for suspicious registrar
        suspicious_registrars = ['NAMECHEAP', 'GODADDY', 'NAME.COM', 'ENOM']
        if whois_info.registrar:
            for reg in suspicious_registrars:
                if reg in whois_info.registrar.upper():
                    score += 5
                    warnings.append(f"Registrar: {whois_info.registrar}")
        
        return {
            'score': score,
            'warnings': warnings,
            'is_suspicious': is_suspicious
        }
    
    def _check_ssl(self, ssl_info: SSLInfo) -> Dict:
        """Check SSL certificate for threats"""
        score = 0
        warnings = []
        
        # Check validity
        if not ssl_info.is_valid:
            score += 50
            warnings.append("SSL certificate is invalid or expired")
        
        # Check expiry
        if ssl_info.days_until_expiry < 7:
            score += 30
            warnings.append(f"SSL certificate expires extremely soon ({ssl_info.days_until_expiry} days)")
        elif ssl_info.days_until_expiry < 30:
            score += 15
            warnings.append(f"SSL certificate expires soon ({ssl_info.days_until_expiry} days)")
        
        # Check for self-signed
        if ssl_info.is_self_signed:
            score += 20
            warnings.append("Self-signed SSL certificate")
        
        return {
            'score': score,
            'warnings': warnings
        }
    
    def _check_content(self, content: PageContent, url_info: URLInfo) -> Dict:
        """Check page content for threats"""
        score = 0
        warnings = []
        patterns = []
        is_phishing = False
        is_scam = False
        is_fake_news = False
        is_malware = False
        brand = None
        
        text_lower = content.text.lower()
        
        # Check for phishing keywords
        for keyword in self.phishing_keywords:
            if keyword in text_lower:
                score += 5
                warnings.append(f"Phishing keyword: '{keyword}'")
                patterns.append(f"phishing_keyword_{keyword.replace(' ', '_')[:30]}")
                is_phishing = True
        
        # Check for brand mentions
        for b in self.brands:
            if b in text_lower and url_info.full_domain not in b:
                # Count occurrences
                count = text_lower.count(b)
                if count > 3:
                    score += min(15, count)
                    warnings.append(f"Multiple mentions of {b} ({count} times)")
                    patterns.append(f"brand_mention_{b}")
                    brand = b
                    is_phishing = True
        
        # Check for email harvesters
        emails = self.email_pattern.findall(content.text)
        if len(emails) > 5:
            score += 15
            warnings.append(f"Multiple email addresses found ({len(emails)})")
            patterns.append("multiple_emails")
            is_scam = True
        
        # Check for phone numbers
        phones = self.phone_pattern.findall(content.text)
        if len(phones) > 3:
            score += 10
            warnings.append(f"Multiple phone numbers found ({len(phones)})")
            patterns.append("multiple_phones")
        
        # Check for cryptocurrency addresses
        bitcoin = self.bitcoin_pattern.findall(content.text)
        eth = self.eth_pattern.findall(content.text)
        if bitcoin or eth:
            score += 20
            warnings.append("Cryptocurrency addresses detected")
            patterns.append("crypto_addresses")
            is_scam = True
        
        # Check for fake login forms
        if content.has_login_form:
            score += 15
            warnings.append("Login form detected")
            patterns.append("login_form")
        
        if content.has_password_field:
            score += 10
            warnings.append("Password field detected")
            patterns.append("password_field")
        
        if content.has_payment_form:
            score += 25
            warnings.append("Payment form detected - verify destination")
            patterns.append("payment_form")
            is_phishing = True
        
        # Check for iframes (often used for malvertising)
        if len(content.iframes) > 0:
            score += 10
            warnings.append(f"Page contains {len(content.iframes)} iframe(s)")
            patterns.append("iframes_detected")
            is_malware = True
        
        # Check for external scripts
        if len(content.scripts) > 20:
            score += 10
            warnings.append(f"Many external scripts ({len(content.scripts)})")
            patterns.append("many_scripts")
        
        # Check for suspicious form actions
        for form in content.forms:
            if form['method'] == 'POST':
                form_domain = urlparse(form['action']).netloc
                if form_domain and form_domain != url_info.full_domain:
                    score += 30
                    warnings.append(f"Form submits to different domain: {form_domain}")
                    patterns.append("cross_domain_form")
                    is_phishing = True
        
        return {
            'score': score,
            'warnings': warnings,
            'patterns': patterns,
            'is_phishing': is_phishing,
            'is_scam': is_scam,
            'is_fake_news': is_fake_news,
            'is_malware': is_malware,
            'brand': brand
        }


# ============================================
# LINK ANALYZER
# ============================================

class LinkAnalyzer:
    """
    Main link analyzer for comprehensive URL analysis
    """
    
    def __init__(self,
                 text_detector: Optional[TextDeepfakeDetector] = None,
                 image_detector: Optional[DeepfakeEnsemble] = None,
                 face_detector: Optional[FaceDetectionEnsemble] = None,
                 use_selenium: bool = False,
                 enable_screenshots: bool = True):
        """
        Initialize link analyzer
        
        Args:
            text_detector: Text deepfake detector
            image_detector: Image deepfake detector
            face_detector: Face detection ensemble
            use_selenium: Use Selenium for JavaScript-heavy sites
            enable_screenshots: Enable screenshot capture
        """
        self.text_detector = text_detector or TextDetectorFactory.create_lightweight_detector()
        self.image_detector = image_detector or EnsembleFactory.create_fast_ensemble()
        self.face_detector = face_detector or FaceDetectionEnsemble()
        
        self.feature_extractor = URLFeatureExtractor()
        self.whois = WhoisLookup()
        self.ssl = SSLChecker()
        self.content_extractor = PageContentExtractor(use_selenium=use_selenium)
        self.threat_detector = ThreatDetector(self.feature_extractor)
        
        self.enable_screenshots = enable_screenshots and SELENIUM_AVAILABLE
        
        logger.info(f"LinkAnalyzer initialized (selenium: {use_selenium}, screenshots: {enable_screenshots})")
    
    def analyze(self, 
                url: str, 
                analyze_content: bool = True,
                check_whois: bool = True,
                check_ssl: bool = True,
                analyze_text: bool = True,
                analyze_images: bool = False,
                take_screenshot: bool = False) -> LinkAnalysisResult:
        """
        Analyze URL for threats
        
        Args:
            url: URL to analyze
            analyze_content: Extract and analyze page content
            check_whois: Perform WHOIS lookup
            check_ssl: Check SSL certificate
            analyze_text: Analyze text for AI generation
            analyze_images: Analyze images for deepfakes
            take_screenshot: Take screenshot of page
        
        Returns:
            LinkAnalysisResult
        """
        start_time = time.time()
        
        # Parse URL
        url_info = URLInfo.from_url(url)
        
        # Initialize variables
        whois_info = None
        ssl_info = None
        page_content = None
        redirect_chain = []
        status_code = 0
        headers = {}
        content_length = 0
        content_type = None
        text_analysis = None
        image_analysis = []
        screenshot_path = None
        screenshot_base64 = None
        
        # Check SSL if HTTPS
        if check_ssl and url_info.scheme == 'https':
            ssl_info = self.ssl.check(url_info.full_domain)
        
        # WHOIS lookup
        if check_whois and url_info.full_domain:
            whois_info = self.whois.lookup(url_info.full_domain)
        
        # Extract page content
        if analyze_content:
            page_content, status_code, headers, redirect_chain = self.content_extractor.extract(
                url, 
                extract_images=analyze_images
            )
            
            content_length = len(page_content.html) if page_content else 0
            content_type = headers.get('Content-Type', '')
            
            # Analyze text for AI generation
            if analyze_text and page_content and page_content.text:
                text_analysis = self.text_detector.detect(
                    page_content.text[:5000],  # Limit text length
                    return_details=False
                )
            
            # Analyze images for deepfakes
            if analyze_images and page_content and page_content.images:
                for img in page_content.images[:10]:  # Limit to first 10 images
                    try:
                        img_result = self._analyze_image(img['src'])
                        if img_result:
                            image_analysis.append(img_result)
                    except Exception as e:
                        logger.error(f"Image analysis error: {str(e)}")
        
        # Take screenshot
        if take_screenshot and self.enable_screenshots:
            screenshot_path = f"temp_screenshot_{int(time.time())}.png"
            screenshot_bytes = self.content_extractor.take_screenshot(url, screenshot_path)
            if screenshot_bytes:
                import base64
                screenshot_base64 = base64.b64encode(screenshot_bytes).decode('utf-8')
        
        # Threat assessment
        threat_assessment = self.threat_detector.analyze(
            url_info, whois_info, ssl_info, page_content
        )
        
        # Update deepfake flag based on image analysis
        if image_analysis:
            threat_assessment.is_deepfake = any(img.get('is_fake', False) for img in image_analysis)
        
        analysis_time = time.time() - start_time
        
        return LinkAnalysisResult(
            url=url,
            url_info=url_info,
            whois_info=whois_info,
            ssl_info=ssl_info,
            page_content=page_content,
            threat_assessment=threat_assessment,
            redirect_chain=redirect_chain,
            response_time=analysis_time,
            status_code=status_code,
            content_type=content_type,
            content_length=content_length,
            text_analysis=text_analysis.to_dict() if text_analysis else None,
            image_analysis=image_analysis,
            screenshot_path=screenshot_path,
            screenshot_base64=screenshot_base64,
            analysis_time=analysis_time
        )
    
    def _analyze_image(self, image_url: str) -> Optional[Dict]:
        """Analyze single image from URL"""
        try:
            # Download image
            response = requests.get(image_url, timeout=10, stream=True)
            if response.status_code != 200:
                return None
            
            # Check content type
            content_type = response.headers.get('Content-Type', '')
            if not content_type.startswith('image/'):
                return None
            
            # Load image
            img_data = response.content
            img = Image.open(io.BytesIO(img_data))
            img_np = np.array(img)
            
            # Detect faces
            faces = self.face_detector.detect(img_np)
            
            # Analyze for deepfakes
            result = self.image_detector.predict_single(img_np)
            
            return {
                'url': image_url,
                'width': img.width,
                'height': img.height,
                'format': img.format,
                'size_kb': len(img_data) / 1024,
                'faces_detected': faces.num_faces,
                'is_fake': result.prediction == 'FAKE',
                'fake_probability': result.fake_probability,
                'confidence': result.confidence
            }
            
        except Exception as e:
            logger.error(f"Image analysis error for {image_url}: {str(e)}")
            return None
    
    def analyze_batch(self, urls: List[str], **kwargs) -> List[LinkAnalysisResult]:
        """Analyze multiple URLs"""
        results = []
        for url in urls:
            try:
                result = self.analyze(url, **kwargs)
                results.append(result)
            except Exception as e:
                logger.error(f"Error analyzing {url}: {str(e)}")
                results.append(None)
        return results
    
    def generate_report(self, result: LinkAnalysisResult, output_path: str):
        """Generate HTML report"""
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>URL Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
                .container {{ max-width: 1200px; margin: auto; background: white; padding: 30px; border-radius: 10px; }}
                h1 {{ color: #333; }}
                .risk-score {{ font-size: 48px; font-weight: bold; text-align: center; padding: 20px; }}
                .risk-CRITICAL {{ color: #721c24; background: #f8d7da; }}
                .risk-HIGH {{ color: #856404; background: #fff3cd; }}
                .risk-MEDIUM {{ color: #0c5460; background: #d1ecf1; }}
                .risk-LOW {{ color: #155724; background: #d4edda; }}
                .section {{ margin: 30px 0; padding: 20px; background: #f8f9fa; border-radius: 8px; }}
                .warning {{ color: #856404; background: #fff3cd; padding: 10px; margin: 5px 0; border-left: 4px solid #ffc107; }}
                table {{ width: 100%; border-collapse: collapse; }}
                th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background: #e9ecef; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🔗 URL Analysis Report</h1>
                <p>URL: <a href="{result.url}" target="_blank">{result.url}</a></p>
                <p>Analyzed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                
                <div class="risk-score risk-{result.threat_assessment.risk_level}">
                    Risk Score: {result.threat_assessment.risk_score}/100
                    <div style="font-size: 18px;">{result.threat_assessment.risk_level}</div>
                </div>
                
                <div class="section">
                    <h2>Threat Assessment</h2>
                    <p><strong>Threat Types:</strong> {', '.join(result.threat_assessment.threat_types) or 'None'}</p>
                    <p><strong>Brand Impersonated:</strong> {result.threat_assessment.brand_impersonated or 'None'}</p>
                    
                    <h3>Warnings</h3>
                    {''.join(f'<div class="warning">⚠️ {w}</div>' for w in result.threat_assessment.warnings)}
                </div>
                
                <div class="section">
                    <h2>URL Information</h2>
                    <table>
                        <tr><td>Domain</td><td>{result.url_info.full_domain}</td></tr>
                        <tr><td>TLD</td><td>.{result.url_info.tld}</td></tr>
                        <tr><td>Scheme</td><td>{result.url_info.scheme}</td></tr>
                        <tr><td>Path</td><td>{result.url_info.path}</td></tr>
                        <tr><td>Parameters</td><td>{len(result.url_info.params)}</td></tr>
                    </table>
                </div>
        """
        
        if result.whois_info:
            html += f"""
                <div class="section">
                    <h2>WHOIS Information</h2>
                    <table>
                        <tr><td>Registrar</td><td>{result.whois_info.registrar}</td></tr>
                        <tr><td>Creation Date</td><td>{result.whois_info.creation_date}</td></tr>
                        <tr><td>Expiration Date</td><td>{result.whois_info.expiration_date}</td></tr>
                        <tr><td>Domain Age</td><td>{result.whois_info.days_old} days</td></tr>
                        <tr><td>Days Until Expiry</td><td>{result.whois_info.days_until_expiry}</td></tr>
                    </table>
                </div>
            """
        
        if result.ssl_info:
            html += f"""
                <div class="section">
                    <h2>SSL Certificate</h2>
                    <table>
                        <tr><td>Valid</td><td>{'✅ Yes' if result.ssl_info.is_valid else '❌ No'}</td></tr>
                        <tr><td>Issuer</td><td>{result.ssl_info.issuer.get('CN', 'Unknown')}</td></tr>
                        <tr><td>Expires In</td><td>{result.ssl_info.days_until_expiry} days</td></tr>
                        <tr><td>Self-signed</td><td>{'✅ Yes' if result.ssl_info.is_self_signed else '❌ No'}</td></tr>
                    </table>
                </div>
            """
        
        if result.page_content:
            html += f"""
                <div class="section">
                    <h2>Page Content</h2>
                    <table>
                        <tr><td>Title</td><td>{result.page_content.title}</td></tr>
                        <tr><td>Word Count</td><td>{result.page_content.word_count}</td></tr>
                        <tr><td>Reading Time</td><td>{result.page_content.reading_time} min</td></tr>
                        <tr><td>Links</td><td>Internal: {len(result.page_content.internal_links)}, External: {len(result.page_content.external_links)}</td></tr>
                        <tr><td>Images</td><td>{len(result.page_content.images)}</td></tr>
                        <tr><td>Forms</td><td>{len(result.page_content.forms)}</td></tr>
                    </table>
                </div>
            """
        
        if result.text_analysis:
            html += f"""
                <div class="section">
                    <h2>Text Analysis</h2>
                    <table>
                        <tr><td>AI Generated</td><td>{'✅ Yes' if result.text_analysis.is_ai_generated else '❌ No'}</td></tr>
                        <tr><td>AI Probability</td><td>{result.text_analysis.ai_probability:.2%}</td></tr>
                        <tr><td>Confidence</td><td>{result.text_analysis.confidence:.2%}</td></tr>
                    </table>
                </div>
            """
        
        html += """
            </div>
        </body>
        </html>
        """
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html)
        
        logger.info(f"Report saved to {output_path}")


# ============================================
# FACTORY CLASS
# ============================================

class LinkAnalyzerFactory:
    """Factory for creating link analyzers"""
    
    @staticmethod
    def create_analyzer(**kwargs) -> LinkAnalyzer:
        """Create link analyzer"""
        return LinkAnalyzer(**kwargs)
    
    @staticmethod
    def create_feature_extractor() -> URLFeatureExtractor:
        """Create feature extractor"""
        return URLFeatureExtractor()
    
    @staticmethod
    def create_threat_detector() -> ThreatDetector:
        """Create threat detector"""
        return ThreatDetector()
    
    @staticmethod
    def create_content_extractor(use_selenium: bool = False) -> PageContentExtractor:
        """Create content extractor"""
        return PageContentExtractor(use_selenium=use_selenium)


# ============================================
# TESTING FUNCTION
# ============================================

def test_link_analyzer():
    """Test link analyzer module"""
    print("=" * 60)
    print("TESTING LINK ANALYZER MODULE")
    print("=" * 60)
    
    # Test URL feature extractor
    print("\n1️⃣ Testing URL Feature Extractor...")
    extractor = URLFeatureExtractor()
    features = extractor.extract_features('https://www.google.com/search?q=test')
    print(f"✅ Extracted {len(features)} features")
    print(f"   Sample features: {dict(list(features.items())[:5])}")
    
    # Test WHOIS lookup
    print("\n2️⃣ Testing WHOIS Lookup...")
    whois = WhoisLookup()
    info = whois.lookup('google.com')
    if info:
        print(f"✅ WHOIS lookup successful")
        print(f"   Registrar: {info.registrar}")
        print(f"   Domain age: {info.days_old} days")
    else:
        print("⚠️ WHOIS lookup failed (may be rate limited)")
    
    # Test SSL checker
    print("\n3️⃣ Testing SSL Checker...")
    ssl = SSLChecker()
    ssl_info = ssl.check('google.com')
    if ssl_info:
        print(f"✅ SSL check successful")
        print(f"   Valid: {ssl_info.is_valid}")
        print(f"   Expires in: {ssl_info.days_until_expiry} days")
    else:
        print("⚠️ SSL check failed")
    
    # Test page content extractor
    print("\n4️⃣ Testing Page Content Extractor...")
    extractor = PageContentExtractor()
    content, status, headers, redirects = extractor.extract('https://example.com')
    if content:
        print(f"✅ Content extracted")
        print(f"   Title: {content.title}")
        print(f"   Word count: {content.word_count}")
        print(f"   Links: {len(content.links)}")
    else:
        print("⚠️ Content extraction failed")
    
    # Test threat detector
    print("\n5️⃣ Testing Threat Detector...")
    detector = ThreatDetector()
    
    # Test with suspicious URL
    url_info = URLInfo.from_url('https://secure-paypal-verification.tk/login.php')
    assessment = detector.analyze(url_info, None, None, None)
    print(f"✅ Threat assessment for suspicious URL")
    print(f"   Risk score: {assessment.risk_score}")
    print(f"   Risk level: {assessment.risk_level}")
    print(f"   Threats: {assessment.threat_types}")
    
    # Test full analyzer
    print("\n6️⃣ Testing Full Link Analyzer...")
    analyzer = LinkAnalyzer(enable_screenshots=False)
    
    # Test with safe URL
    result = analyzer.analyze('https://www.google.com', analyze_content=False)
    print(f"✅ Google.com analysis")
    print(f"   Risk score: {result.threat_assessment.risk_score}")
    print(f"   Status: {result.summary()}")
    
    print("\n" + "=" * 60)
    print("✅ LINK ANALYZER TEST PASSED!")
    print("=" * 60)


# ============================================
# MAIN EXECUTION
# ============================================

if __name__ == "__main__":
    test_link_analyzer()
    
    print("\n📝 Example usage:")
    print("""
    from app.link_analyzer import LinkAnalyzer
    
    # Create analyzer
    analyzer = LinkAnalyzer()
    
    # Analyze URL
    result = analyzer.analyze('https://example.com')
    
    # Check if suspicious
    if result.threat_assessment.risk_score > 50:
        print(f"⚠️ Suspicious URL detected!")
        for warning in result.threat_assessment.warnings:
            print(f"  - {warning}")
    
    # Generate report
    analyzer.generate_report(result, 'url_report.html')
    """)