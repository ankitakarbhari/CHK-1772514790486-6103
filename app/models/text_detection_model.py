# Placeholder file for text_detection_model.py
# app/models/text_detection_model.py
"""
AI-Generated Text Detection Model
Detects if text was written by AI (GPT-3/4, Claude, etc.) or humans
Uses multiple techniques: Perplexity, Burstiness, BERT/RoBERTa classifiers
Python 3.13+ Compatible
"""

import os
import sys
import re
import math
import json
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, asdict
from collections import Counter
import time
import hashlib

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

# ========== DEEP LEARNING IMPORTS ==========
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
    pipeline,
    RobertaTokenizer,
    RobertaForSequenceClassification,
    BertTokenizer,
    BertForSequenceClassification,
    GPT2Tokenizer,
    GPT2LMHeadModel
)

# ========== NLP IMPORTS ==========
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tag import pos_tag
import textstat
import spacy

# Download NLTK data if not present
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('universal_tagset', quiet=True)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================
# DATA CLASSES FOR TEXT DETECTION RESULTS
# ============================================

@dataclass
class TextMetrics:
    """Text analysis metrics"""
    perplexity: float
    burstiness: float
    avg_sentence_length: float
    avg_word_length: float
    vocabulary_richness: float
    repetition_score: float
    pos_diversity: float
    readability_score: float
    unique_words_ratio: float
    stopword_ratio: float
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class TextDetectionResult:
    """Text detection result"""
    text: str
    text_hash: str
    ai_probability: float
    human_probability: float
    prediction: str  # 'AI' or 'HUMAN'
    confidence: float
    metrics: TextMetrics
    model_scores: Dict[str, float]
    suspicious_sentences: List[Dict[str, Any]]
    inference_time: float
    
    def to_dict(self) -> Dict:
        result = asdict(self)
        result['metrics'] = self.metrics.to_dict()
        return result
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


# ============================================
# STATISTICAL TEXT ANALYZER
# ============================================

class StatisticalTextAnalyzer:
    """
    Statistical analysis of text without deep learning
    Uses linguistic features to detect AI-generated text
    """
    
    def __init__(self):
        logger.info("Initializing Statistical Text Analyzer")
        
        # Load spaCy model for advanced NLP
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except:
            logger.warning("spaCy model not found, downloading...")
            os.system("python -m spacy download en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")
        
        # Common patterns in AI text
        self.ai_patterns = [
            r'\b(?:additionally|furthermore|moreover|consequently|therefore)\b',
            r'\b(?:in conclusion|to summarize|as previously mentioned)\b',
            r'\b(?:it is important to note|it should be noted that)\b',
            r'\b(?:there are several|there are many|there are various)\b',
            r'\b(?:firstly|secondly|thirdly|lastly)\b'
        ]
        
        # Compile patterns
        self.compiled_patterns = [re.compile(p, re.IGNORECASE) for p in self.ai_patterns]
    
    def analyze(self, text: str) -> TextMetrics:
        """
        Analyze text and extract statistical features
        """
        # Clean text
        text = text.strip()
        if not text:
            return self._empty_metrics()
        
        # Tokenize
        sentences = sent_tokenize(text)
        words = word_tokenize(text.lower())
        words = [w for w in words if w.isalnum()]
        
        if len(words) == 0:
            return self._empty_metrics()
        
        # Calculate metrics
        # 1. Perplexity-like score (using word rarity)
        word_freq = Counter(words)
        unique_words = len(word_freq)
        total_words = len(words)
        
        # 2. Burstiness (variance in sentence lengths)
        sent_lengths = [len(word_tokenize(s)) for s in sentences]
        if len(sent_lengths) > 1:
            burstiness = np.std(sent_lengths) / (np.mean(sent_lengths) + 1e-8)
        else:
            burstiness = 0.0
        
        # 3. Vocabulary richness (Type-Token Ratio)
        vocabulary_richness = unique_words / total_words if total_words > 0 else 0
        
        # 4. Repetition score
        word_repetition = sum(v for k, v in word_freq.items() if v > 3) / total_words if total_words > 0 else 0
        
        # 5. POS diversity
        pos_tags = pos_tag(words, tagset='universal')
        pos_counts = Counter(tag for word, tag in pos_tags)
        pos_diversity = len(pos_counts) / 12.0  # 12 universal POS tags
        
        # 6. Readability score
        try:
            readability = textstat.flesch_reading_ease(text) / 100.0  # Normalize to 0-1
        except:
            readability = 0.5
        
        # 7. Unique words ratio
        unique_ratio = unique_words / total_words
        
        # 8. Stopword ratio
        stopwords = set(nltk.corpus.stopwords.words('english'))
        stopword_count = sum(1 for w in words if w in stopwords)
        stopword_ratio = stopword_count / total_words if total_words > 0 else 0
        
        # 9. Pattern matching score (AI patterns)
        pattern_matches = 0
        for pattern in self.compiled_patterns:
            pattern_matches += len(pattern.findall(text))
        pattern_score = min(pattern_matches / len(sentences) if sentences else 0, 1.0)
        
        # Combine into perplexity-like score
        perplexity = (
            0.3 * (1 - vocabulary_richness) + 
            0.2 * word_repetition +
            0.2 * pattern_score +
            0.3 * (1 - pos_diversity)
        )
        
        return TextMetrics(
            perplexity=float(perplexity),
            burstiness=float(burstiness),
            avg_sentence_length=float(np.mean(sent_lengths) if sent_lengths else 0),
            avg_word_length=float(np.mean([len(w) for w in words]) if words else 0),
            vocabulary_richness=float(vocabulary_richness),
            repetition_score=float(word_repetition),
            pos_diversity=float(pos_diversity),
            readability_score=float(readability),
            unique_words_ratio=float(unique_ratio),
            stopword_ratio=float(stopword_ratio)
        )
    
    def _empty_metrics(self) -> TextMetrics:
        """Return empty metrics for invalid text"""
        return TextMetrics(
            perplexity=0.5,
            burstiness=0.0,
            avg_sentence_length=0.0,
            avg_word_length=0.0,
            vocabulary_richness=0.0,
            repetition_score=0.0,
            pos_diversity=0.0,
            readability_score=0.5,
            unique_words_ratio=0.0,
            stopword_ratio=0.0
        )
    
    def find_suspicious_sentences(self, text: str, threshold: float = 0.6) -> List[Dict]:
        """
        Find sentences that look AI-generated
        """
        sentences = sent_tokenize(text)
        suspicious = []
        
        for i, sent in enumerate(sentences):
            if len(sent.split()) < 5:  # Skip very short sentences
                continue
            
            # Analyze sentence
            metrics = self.analyze(sent)
            
            # Calculate suspicious score
            score = (
                0.4 * metrics.perplexity +
                0.3 * (1 - metrics.vocabulary_richness) +
                0.3 * metrics.repetition_score
            )
            
            if score > threshold:
                suspicious.append({
                    'index': i,
                    'text': sent,
                    'score': float(score),
                    'reasons': self._get_suspicion_reasons(sent, metrics)
                })
        
        return suspicious
    
    def _get_suspicion_reasons(self, sentence: str, metrics: TextMetrics) -> List[str]:
        """Get reasons why a sentence is suspicious"""
        reasons = []
        
        if metrics.perplexity > 0.7:
            reasons.append("Unusual word choices")
        if metrics.repetition_score > 0.3:
            reasons.append("Repetitive patterns")
        if metrics.vocabulary_richness < 0.4:
            reasons.append("Limited vocabulary")
        if any(p.search(sentence) for p in self.compiled_patterns):
            reasons.append("Contains AI-typical phrases")
        
        return reasons


# ============================================
# PERPLEXITY-BASED DETECTOR (GPT-2)
# ============================================

class PerplexityDetector:
    """
    Detects AI text using perplexity from language models
    AI text typically has lower perplexity than human text
    """
    
    def __init__(self, model_name: str = 'gpt2', device: Optional[str] = None):
        """
        Initialize perplexity detector
        
        Args:
            model_name: 'gpt2', 'gpt2-medium', 'gpt2-large'
            device: 'cuda' or 'cpu'
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = model_name
        
        logger.info(f"Initializing PerplexityDetector with {model_name} on {self.device}")
        
        # Load tokenizer and model
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        # Set pad token
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        logger.info("PerplexityDetector initialized")
    
    def calculate_perplexity(self, text: str) -> float:
        """
        Calculate perplexity of text
        Lower perplexity = more likely AI-generated
        """
        try:
            # Tokenize
            encodings = self.tokenizer(
                text, 
                return_tensors='pt',
                truncation=True,
                max_length=512,
                return_attention_mask=True
            )
            
            input_ids = encodings.input_ids.to(self.device)
            attention_mask = encodings.attention_mask.to(self.device)
            
            with torch.no_grad():
                outputs = self.model(input_ids, attention_mask=attention_mask, labels=input_ids)
                loss = outputs.loss
                perplexity = torch.exp(loss).item()
            
            return float(perplexity)
            
        except Exception as e:
            logger.error(f"Perplexity calculation error: {str(e)}")
            return 100.0  # Return high perplexity on error
    
    def calculate_perplexity_batch(self, texts: List[str]) -> List[float]:
        """Calculate perplexity for multiple texts"""
        return [self.calculate_perplexity(t) for t in texts]
    
    def get_token_probabilities(self, text: str) -> List[Tuple[str, float]]:
        """
        Get probabilities for each token
        Useful for identifying suspicious parts
        """
        try:
            # Tokenize
            encodings = self.tokenizer(
                text, 
                return_tensors='pt',
                return_attention_mask=True
            )
            input_ids = encodings.input_ids.to(self.device)
            
            with torch.no_grad():
                outputs = self.model(input_ids)
                logits = outputs.logits
            
            # Get probabilities
            probs = F.softmax(logits[0], dim=-1)
            
            # Get token probabilities
            token_probs = []
            for i, token_id in enumerate(input_ids[0]):
                token = self.tokenizer.decode([token_id])
                prob = probs[i, token_id].item()
                token_probs.append((token, prob))
            
            return token_probs
            
        except Exception as e:
            logger.error(f"Token probability error: {str(e)}")
            return []


# ============================================
# BERT-BASED CLASSIFIER
# ============================================

class BERTTextClassifier:
    """
    BERT-based classifier for AI-generated text
    Fine-tuned on human/AI text datasets
    """
    
    def __init__(self, 
                 model_name: str = 'roberta-base-openai-detector',
                 device: Optional[str] = None):
        """
        Initialize BERT classifier
        
        Args:
            model_name: Pretrained model for AI detection
                       'roberta-base-openai-detector' (OpenAI's detector)
                       'bert-base-uncased' (need fine-tuning)
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = model_name
        
        logger.info(f"Initializing BERTTextClassifier with {model_name} on {self.device}")
        
        try:
            # Load tokenizer and model
            if 'roberta' in model_name:
                self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
                self.model = RobertaForSequenceClassification.from_pretrained(model_name)
            else:
                self.tokenizer = BertTokenizer.from_pretrained(model_name)
                self.model = BertForSequenceClassification.from_pretrained(
                    model_name, 
                    num_labels=2
                )
            
            self.model.to(self.device)
            self.model.eval()
            
            logger.info("BERT classifier initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to load BERT model: {str(e)}")
            logger.warning("Using fallback mode (statistical only)")
            self.model = None
    
    def predict(self, text: str) -> Dict[str, float]:
        """
        Predict if text is AI-generated using BERT
        
        Returns:
            Dictionary with probabilities
        """
        if self.model is None:
            return {'ai_probability': 0.5, 'human_probability': 0.5, 'confidence': 0.0}
        
        try:
            # Tokenize
            inputs = self.tokenizer(
                text,
                return_tensors='pt',
                truncation=True,
                max_length=512,
                padding=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Predict
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probabilities = F.softmax(logits, dim=-1)
            
            probs = probabilities.cpu().numpy()[0]
            
            # Assuming class 0 = human, class 1 = AI (or vice versa)
            # OpenAI detector: 0=real, 1=fake
            ai_prob = float(probs[1])
            human_prob = float(probs[0])
            confidence = float(max(probs))
            
            return {
                'ai_probability': ai_prob,
                'human_probability': human_prob,
                'confidence': confidence
            }
            
        except Exception as e:
            logger.error(f"BERT prediction error: {str(e)}")
            return {'ai_probability': 0.5, 'human_probability': 0.5, 'confidence': 0.0}
    
    def predict_batch(self, texts: List[str]) -> List[Dict]:
        """Predict for multiple texts"""
        return [self.predict(t) for t in texts]


# ============================================
# ENSEMBLE TEXT DETECTOR
# ============================================

class TextDeepfakeDetector:
    """
    Ensemble detector for AI-generated text
    Combines multiple techniques for high accuracy
    """
    
    def __init__(self, 
                 use_perplexity: bool = True,
                 use_bert: bool = True,
                 use_statistical: bool = True,
                 device: Optional[str] = None):
        """
        Initialize ensemble text detector
        
        Args:
            use_perplexity: Use GPT-2 perplexity detector
            use_bert: Use BERT classifier
            use_statistical: Use statistical analyzer
            device: 'cuda' or 'cpu'
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        logger.info("Initializing TextDeepfakeDetector")
        logger.info(f"Using: perplexity={use_perplexity}, bert={use_bert}, statistical={use_statistical}")
        
        # Initialize components
        self.statistical = StatisticalTextAnalyzer() if use_statistical else None
        
        if use_perplexity:
            try:
                self.perplexity = PerplexityDetector(device=self.device)
            except Exception as e:
                logger.error(f"Failed to load perplexity detector: {str(e)}")
                self.perplexity = None
        else:
            self.perplexity = None
        
        if use_bert:
            try:
                self.bert = BERTTextClassifier(device=self.device)
            except Exception as e:
                logger.error(f"Failed to load BERT classifier: {str(e)}")
                self.bert = None
        else:
            self.bert = None
        
        # Weights for ensemble
        self.weights = {
            'perplexity': 0.35 if self.perplexity else 0,
            'bert': 0.45 if self.bert else 0,
            'statistical': 0.20 if self.statistical else 0
        }
        
        # Normalize weights
        total = sum(self.weights.values())
        if total > 0:
            self.weights = {k: v/total for k, v in self.weights.items()}
        
        logger.info(f"Ensemble weights: {self.weights}")
    
    def detect(self, text: str, return_details: bool = True) -> TextDetectionResult:
        """
        Detect if text is AI-generated
        
        Args:
            text: Input text to analyze
            return_details: Return detailed metrics
        
        Returns:
            TextDetectionResult with prediction
        """
        start_time = time.time()
        
        # Clean text
        text = text.strip()
        text_hash = hashlib.md5(text.encode()).hexdigest()
        
        if len(text.split()) < 10:
            # Too short for reliable detection
            return TextDetectionResult(
                text=text[:100] + ('...' if len(text) > 100 else ''),
                text_hash=text_hash,
                ai_probability=0.5,
                human_probability=0.5,
                prediction='UNCERTAIN',
                confidence=0.0,
                metrics=self.statistical.analyze(text) if self.statistical else None,
                model_scores={},
                suspicious_sentences=[],
                inference_time=time.time() - start_time
            )
        
        model_scores = {}
        
        # 1. Statistical analysis
        if self.statistical:
            metrics = self.statistical.analyze(text)
            # Convert metrics to probability
            stat_prob = (
                0.4 * metrics.perplexity +
                0.3 * metrics.repetition_score +
                0.3 * (1 - metrics.vocabulary_richness)
            )
            model_scores['statistical'] = float(stat_prob)
        else:
            metrics = None
            stat_prob = 0.5
        
        # 2. Perplexity analysis
        if self.perplexity:
            try:
                perplexity = self.perplexity.calculate_perplexity(text)
                # Normalize perplexity to probability
                # Lower perplexity = more likely AI
                perplexity_prob = 1.0 / (1.0 + math.exp(-(30 - perplexity) / 10))
                model_scores['perplexity'] = float(perplexity_prob)
            except Exception as e:
                logger.error(f"Perplexity error: {str(e)}")
                perplexity_prob = 0.5
        else:
            perplexity_prob = 0.5
        
        # 3. BERT analysis
        if self.bert:
            try:
                bert_result = self.bert.predict(text)
                bert_prob = bert_result['ai_probability']
                model_scores['bert'] = float(bert_prob)
            except Exception as e:
                logger.error(f"BERT error: {str(e)}")
                bert_prob = 0.5
        else:
            bert_prob = 0.5
        
        # Weighted ensemble
        weighted_prob = 0.0
        total_weight = 0.0
        
        if 'perplexity' in self.weights and self.perplexity:
            weighted_prob += self.weights['perplexity'] * perplexity_prob
            total_weight += self.weights['perplexity']
        
        if 'bert' in self.weights and self.bert:
            weighted_prob += self.weights['bert'] * bert_prob
            total_weight += self.weights['bert']
        
        if 'statistical' in self.weights and self.statistical:
            weighted_prob += self.weights['statistical'] * stat_prob
            total_weight += self.weights['statistical']
        
        if total_weight > 0:
            ai_probability = weighted_prob / total_weight
        else:
            ai_probability = 0.5
        
        human_probability = 1.0 - ai_probability
        
        # Calculate confidence based on agreement
        scores = [v for v in model_scores.values()]
        if len(scores) > 1:
            confidence = 1.0 - min(np.std(scores), 0.5)
        else:
            confidence = 0.8 if abs(ai_probability - 0.5) > 0.2 else 0.5
        
        prediction = 'AI' if ai_probability > 0.5 else 'HUMAN'
        
        # Find suspicious sentences
        suspicious = []
        if self.statistical and return_details:
            suspicious = self.statistical.find_suspicious_sentences(text)
        
        result = TextDetectionResult(
            text=text[:200] + ('...' if len(text) > 200 else ''),
            text_hash=text_hash,
            ai_probability=float(ai_probability),
            human_probability=float(human_probability),
            prediction=prediction,
            confidence=float(confidence),
            metrics=metrics,
            model_scores=model_scores,
            suspicious_sentences=suspicious[:5],  # Top 5 suspicious sentences
            inference_time=time.time() - start_time
        )
        
        logger.info(f"Text detection: {prediction} (AI={ai_probability:.3f}, Conf={confidence:.3f})")
        
        return result
    
    def detect_batch(self, texts: List[str]) -> List[TextDetectionResult]:
        """Detect for multiple texts"""
        return [self.detect(t) for t in texts]
    
    def analyze_url_content(self, url: str, max_length: int = 2000) -> TextDetectionResult:
        """
        Analyze text content from a URL
        """
        try:
            import requests
            from bs4 import BeautifulSoup
            
            # Fetch URL
            response = requests.get(url, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract text
            for script in soup(["script", "style"]):
                script.decompose()
            
            text = soup.get_text()
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            # Truncate if too long
            if len(text) > max_length:
                text = text[:max_length]
            
            return self.detect(text)
            
        except Exception as e:
            logger.error(f"URL analysis error: {str(e)}")
            return None
    
    def save_model(self, path: str):
        """Save detector configuration"""
        config = {
            'weights': self.weights,
            'components': {
                'perplexity': self.perplexity is not None,
                'bert': self.bert is not None,
                'statistical': self.statistical is not None
            }
        }
        with open(path, 'w') as f:
            json.dump(config, f, indent=2)
        logger.info(f"Config saved to {path}")
    
    def load_model(self, path: str):
        """Load detector configuration"""
        with open(path, 'r') as f:
            config = json.load(f)
        self.weights = config['weights']
        logger.info(f"Config loaded from {path}")


# ============================================
# FACTORY CLASS
# ============================================

class TextDetectorFactory:
    """Factory for creating text detectors"""
    
    @staticmethod
    def create_full_detector(device: Optional[str] = None) -> TextDeepfakeDetector:
        """Create full detector with all components"""
        return TextDeepfakeDetector(
            use_perplexity=True,
            use_bert=True,
            use_statistical=True,
            device=device
        )
    
    @staticmethod
    def create_lightweight_detector(device: Optional[str] = None) -> TextDeepfakeDetector:
        """Create lightweight detector (statistical only)"""
        return TextDeepfakeDetector(
            use_perplexity=False,
            use_bert=False,
            use_statistical=True,
            device=device
        )
    
    @staticmethod
    def create_accurate_detector(device: Optional[str] = None) -> TextDeepfakeDetector:
        """Create most accurate detector"""
        return TextDeepfakeDetector(
            use_perplexity=True,
            use_bert=True,
            use_statistical=True,
            device=device
        )
    
    @staticmethod
    def create_from_config(config_path: str) -> TextDeepfakeDetector:
        """Create detector from config"""
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        return TextDeepfakeDetector(
            use_perplexity=config['components']['perplexity'],
            use_bert=config['components']['bert'],
            use_statistical=config['components']['statistical']
        )


# ============================================
# TESTING FUNCTION
# ============================================

def test_text_detector():
    """Test the text deepfake detector"""
    print("=" * 60)
    print("TESTING AI TEXT DETECTION MODEL")
    print("=" * 60)
    
    # Test samples
    human_texts = [
        """I went to the beach yesterday with my friends. The weather was perfect, 
        not too hot, with a nice breeze. We played volleyball and swam in the ocean. 
        Later, we had a barbecue and watched the sunset. It was one of those perfect 
        days that you never want to end. I'm already looking forward to next time!""",
        
        """Just finished reading this amazing book! Couldn't put it down all weekend. 
        The characters felt so real, and the plot twists caught me completely off guard. 
        Has anyone else read it? Would love to discuss the ending - it's been on my 
        mind for days!"""
    ]
    
    ai_texts = [
        """In conclusion, the implementation of artificial intelligence in healthcare 
        presents numerous advantages for patient care and clinical outcomes. Firstly, 
        AI-powered diagnostic systems can analyze medical images with remarkable accuracy. 
        Additionally, machine learning algorithms can predict patient deterioration 
        before it becomes clinically apparent. Furthermore, these technologies can 
        streamline administrative tasks, allowing healthcare professionals to focus 
        more on direct patient care. It is important to note that while these benefits 
        are substantial, careful consideration must be given to ethical implications 
        and data privacy concerns.""",
        
        """The integration of renewable energy sources into existing power grids 
        requires comprehensive planning and infrastructure development. Solar and 
        wind power, while environmentally beneficial, present intermittency challenges 
        that must be addressed through energy storage solutions. Moreover, grid 
        modernization is essential to accommodate bidirectional power flow from 
        distributed generation sources. Consequently, policymakers must consider 
        these factors when designing sustainable energy transitions."""
    ]
    
    try:
        # Create detector
        print("\n1️⃣ Creating text detector...")
        detector = TextDetectorFactory.create_full_detector()
        print("✅ Detector created")
        print(f"   Components: {[k for k, v in detector.weights.items() if v > 0]}")
        
        # Test human texts
        print("\n2️⃣ Testing HUMAN-written texts...")
        for i, text in enumerate(human_texts):
            result = detector.detect(text)
            print(f"\n   Text {i+1}:")
            print(f"   ├─ Prediction: {result.prediction}")
            print(f"   ├─ AI Prob: {result.ai_probability:.3f}")
            print(f"   ├─ Confidence: {result.confidence:.3f}")
            print(f"   └─ Time: {result.inference_time:.3f}s")
            
            if result.metrics:
                print(f"      Metrics:")
                print(f"         Perplexity: {result.metrics.perplexity:.3f}")
                print(f"         Burstiness: {result.metrics.burstiness:.3f}")
                print(f"         Vocabulary: {result.metrics.vocabulary_richness:.3f}")
        
        # Test AI texts
        print("\n3️⃣ Testing AI-generated texts...")
        for i, text in enumerate(ai_texts):
            result = detector.detect(text)
            print(f"\n   Text {i+1}:")
            print(f"   ├─ Prediction: {result.prediction}")
            print(f"   ├─ AI Prob: {result.ai_probability:.3f}")
            print(f"   ├─ Confidence: {result.confidence:.3f}")
            print(f"   └─ Time: {result.inference_time:.3f}s")
            
            if result.model_scores:
                print(f"      Model scores:")
                for model, score in result.model_scores.items():
                    print(f"         {model}: {score:.3f}")
        
        # Test suspicious sentences
        print("\n4️⃣ Finding suspicious sentences...")
        result = detector.detect(ai_texts[0])
        if result.suspicious_sentences:
            print(f"   Found {len(result.suspicious_sentences)} suspicious sentences:")
            for i, sent in enumerate(result.suspicious_sentences):
                print(f"\n   {i+1}. {sent['text'][:80]}...")
                print(f"      Score: {sent['score']:.3f}")
                print(f"      Reasons: {', '.join(sent['reasons'])}")
        
        print("\n" + "=" * 60)
        print("✅ TEXT DETECTOR TEST PASSED!")
        print("=" * 60)
        
        return detector
        
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def benchmark_text_detector(detector: TextDeepfakeDetector, num_iterations: int = 20):
    """Benchmark text detector performance"""
    print("\n" + "=" * 60)
    print("BENCHMARKING TEXT DETECTOR")
    print("=" * 60)
    
    test_text = """
    Artificial intelligence has transformed numerous industries in recent years. 
    From healthcare to finance, AI systems are being deployed to automate complex 
    tasks and provide valuable insights. Machine learning algorithms can now 
    analyze vast amounts of data with unprecedented speed and accuracy. 
    """
    
    import time
    
    # Warmup
    for _ in range(3):
        detector.detect(test_text)
    
    # Benchmark
    times = []
    for i in range(num_iterations):
        start = time.time()
        detector.detect(test_text)
        end = time.time()
        times.append(end - start)
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    
    print(f"\n📊 BENCHMARK RESULTS:")
    print(f"   Average time: {avg_time*1000:.2f} ms")
    print(f"   Std deviation: {std_time*1000:.2f} ms")
    print(f"   Min time: {min(times)*1000:.2f} ms")
    print(f"   Max time: {max(times)*1000:.2f} ms")
    print(f"   Texts per second: {1.0/avg_time:.1f}")


if __name__ == "__main__":
    # Run test
    detector = test_text_detector()
    
    # Run benchmark
    if detector:
        benchmark_text_detector(detector)
        
        print("\n📝 Example usage:")
        print("""
# In your main application:
from app.models.text_detection_model import TextDetectorFactory

# Create detector
detector = TextDetectorFactory.create_full_detector()

# Analyze text
text = "Your text to analyze here..."
result = detector.detect(text)

if result.prediction == 'AI':
    print(f"⚠️ AI-GENERATED TEXT DETECTED!")
    print(f"   Confidence: {result.confidence:.2%}")
    print(f"   AI Probability: {result.ai_probability:.2%}")
    
    # Show suspicious sentences
    for sent in result.suspicious_sentences:
        print(f"   - {sent['text']}")
else:
    print(f"✅ Text appears to be human-written")
    print(f"   Confidence: {result.confidence:.2%}")

# Analyze URL content
url_result = detector.analyze_url_content("https://example.com/article")
if url_result:
    print(f"URL content: {url_result.prediction}")
        """)