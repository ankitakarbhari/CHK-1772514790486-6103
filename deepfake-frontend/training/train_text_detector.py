# training/train_text_detector.py
"""
Complete Training Script for AI-Generated Text Detection Model
Trains BERT/RoBERTa, Perplexity-based detector, and custom neural network
Supports multiple datasets: GPT-3/4, Claude, LLaMA, Human text
"""

import os
import sys
import json
import time
import math
import random
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

# ========== DEEP LEARNING IMPORTS ==========
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler

# ========== TRANSFORMERS IMPORTS ==========
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
    RobertaTokenizer,
    RobertaForSequenceClassification,
    BertTokenizer,
    BertForSequenceClassification,
    GPT2Tokenizer,
    GPT2LMHeadModel,
    GPTNeoForCausalLM,
    T5ForConditionalGeneration,
    AdamW,
    get_linear_schedule_with_warmup
)

# ========== NLP IMPORTS ==========
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import textstat
import spacy

# Download NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)

# ========== SKLEARN METRICS ==========
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)

# ========== IMPORT MODELS ==========
from app.models.text_detection_model import (
    StatisticalTextAnalyzer,
    PerplexityDetector,
    BERTAIDetector,
    TextDeepfakeDetector,
    TextDetectorFactory
)

# Set random seeds
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)


# ============================================
# CONFIGURATION
# ============================================

class Config:
    """Configuration for text detector training"""
    
    # Model configuration
    model_name = 'text_detector'
    use_bert = True
    use_perplexity = True
    use_statistical = True
    use_gptzero = True
    
    # BERT model selection
    bert_model_name = 'roberta-base-openai-detector'  # Options: 'roberta-base', 'bert-base-uncased', 'roberta-base-openai-detector'
    
    # Perplexity model
    perplexity_model = 'gpt2'  # 'gpt2', 'gpt2-medium', 'gpt2-large'
    
    # Data configuration
    data_dir = Path('datasets/text_data')
    train_file = data_dir / 'train.json'
    val_file = data_dir / 'val.json'
    test_file = data_dir / 'test.json'
    
    # Training configuration
    batch_size = 16
    epochs = 10
    learning_rate = 2e-5
    weight_decay = 0.01
    warmup_steps = 500
    
    # Hardware
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_workers = 4
    
    # Mixed precision
    use_amp = True if device == 'cuda' else False
    
    # Checkpointing
    save_dir = Path('checkpoints/text_detector')
    model_save_path = Path('models/text_detection_model.pt')
    bert_save_path = Path('models/bert_text_detector.pt')
    perplexity_config_path = Path('models/perplexity_config.json')
    ensemble_config_path = Path('models/text_ensemble_config.json')
    log_dir = Path('logs/text_detector')
    
    # Text processing
    max_length = 512
    min_text_length = 20
    
    # Early stopping
    early_stopping = True
    early_stopping_patience = 3
    early_stopping_min_delta = 0.001
    
    # Evaluation
    eval_interval = 1
    save_interval = 2
    log_interval = 10
    
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        
        # Create directories
        self.save_dir.mkdir(exist_ok=True, parents=True)
        self.log_dir.mkdir(exist_ok=True, parents=True)
        self.model_save_path.parent.mkdir(exist_ok=True, parents=True)
        self.bert_save_path.parent.mkdir(exist_ok=True, parents=True)
    
    def __str__(self):
        return json.dumps(self.__dict__, indent=2, default=str)
    
    def save(self, path):
        with open(path, 'w') as f:
            json.dump(self.__dict__, f, indent=2, default=str)


# ============================================
# CUSTOM DATASET
# ============================================

class TextDataset(Dataset):
    """Dataset for text classification"""
    
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Validate
        assert len(texts) == len(labels), "Texts and labels must have same length"
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long)
        }


# ============================================
# CUSTOM NEURAL NETWORK
# ============================================

class TextDeepfakeNN(nn.Module):
    """Custom neural network combining embeddings and statistical features"""
    
    def __init__(self, 
                 vocab_size=30522,
                 embedding_dim=256,
                 hidden_dim=512,
                 num_classes=2,
                 num_statistical_features=20,
                 dropout=0.3):
        super(TextDeepfakeNN, self).__init__()
        
        # Text embedding
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.dropout = nn.Dropout(dropout)
        
        # BiLSTM for sequence modeling
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim // 2,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.Tanh(),
            nn.Linear(hidden_dim // 4, 1)
        )
        
        # Statistical features processing
        self.stat_fc = nn.Sequential(
            nn.Linear(num_statistical_features, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Combined classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim + 64, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, input_ids, attention_mask=None, statistical_features=None):
        # Text embedding
        embedded = self.embedding(input_ids)
        embedded = self.dropout(embedded)
        
        # LSTM
        lstm_out, (hidden, cell) = self.lstm(embedded)
        
        # Attention
        attention_weights = F.softmax(self.attention(lstm_out), dim=1)
        attended = torch.sum(attention_weights * lstm_out, dim=1)
        
        # Use concatenated hidden states
        text_features = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        
        # Combine with statistical features
        if statistical_features is not None:
            stat_features = self.stat_fc(statistical_features)
            combined = torch.cat([text_features, stat_features], dim=1)
        else:
            combined = text_features
        
        # Classification
        output = self.classifier(combined)
        
        return output


# ============================================
# STATISTICAL FEATURE EXTRACTOR
# ============================================

class StatisticalFeatureExtractor:
    """Extract statistical features from text"""
    
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.analyzer = StatisticalTextAnalyzer()
        
    def extract_features(self, text):
        """Extract numerical features from text"""
        features = []
        
        # Basic statistics
        words = word_tokenize(text.lower())
        sentences = sent_tokenize(text)
        
        if not words:
            return np.zeros(20)
        
        # Word-level features
        word_count = len(words)
        unique_words = len(set(words))
        char_count = len(text)
        
        features.append(word_count / 1000)  # Normalized
        features.append(unique_words / max(1, word_count))
        features.append(char_count / 5000)  # Normalized
        
        # Sentence-level features
        sent_count = len(sentences)
        avg_sent_len = word_count / max(1, sent_count)
        
        features.append(sent_count / 100)  # Normalized
        features.append(avg_sent_len / 50)  # Normalized
        
        # Readability scores
        try:
            flesch = textstat.flesch_reading_ease(text) / 100
            fog = textstat.gunning_fog(text) / 20
            features.append(flesch)
            features.append(fog)
        except:
            features.append(0.5)
            features.append(0.5)
        
        # Part-of-speech distribution
        pos_tags = nltk.pos_tag(words)
        pos_counts = {}
        for _, tag in pos_tags:
            pos_counts[tag] = pos_counts.get(tag, 0) + 1
        
        # Noun ratio (NN, NNS, NNP, NNPS)
        noun_count = sum(pos_counts.get(tag, 0) for tag in ['NN', 'NNS', 'NNP', 'NNPS'])
        features.append(noun_count / max(1, word_count))
        
        # Verb ratio (VB, VBD, VBG, VBN, VBP, VBZ)
        verb_count = sum(pos_counts.get(tag, 0) for tag in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'])
        features.append(verb_count / max(1, word_count))
        
        # Adjective ratio (JJ, JJR, JJS)
        adj_count = sum(pos_counts.get(tag, 0) for tag in ['JJ', 'JJR', 'JJS'])
        features.append(adj_count / max(1, word_count))
        
        # Adverb ratio (RB, RBR, RBS)
        adv_count = sum(pos_counts.get(tag, 0) for tag in ['RB', 'RBR', 'RBS'])
        features.append(adv_count / max(1, word_count))
        
        # Stopword ratio
        stopword_count = sum(1 for w in words if w in self.stop_words)
        features.append(stopword_count / max(1, word_count))
        
        # Punctuation ratio
        punct_count = sum(1 for c in text if c in '.,;:!?"\'()[]{}')
        features.append(punct_count / max(1, char_count))
        
        # Uppercase ratio
        upper_count = sum(1 for c in text if c.isupper())
        features.append(upper_count / max(1, char_count))
        
        # Digit ratio
        digit_count = sum(1 for c in text if c.isdigit())
        features.append(digit_count / max(1, char_count))
        
        # Vocabulary richness (Type-Token Ratio)
        features.append(unique_words / max(1, word_count))
        
        # Hapax legomena (words appearing once)
        word_freq = {}
        for w in words:
            word_freq[w] = word_freq.get(w, 0) + 1
        hapax = sum(1 for freq in word_freq.values() if freq == 1)
        features.append(hapax / max(1, word_count))
        
        # Average word length
        avg_word_len = char_count / max(1, word_count)
        features.append(avg_word_len / 10)  # Normalized
        
        # Ensure we have exactly 20 features
        while len(features) < 20:
            features.append(0.0)
        
        return np.array(features[:20], dtype=np.float32)


# ============================================
# BERT TRAINER
# ============================================

class BERTTrainer:
    """Train BERT/RoBERTa for text classification"""
    
    def __init__(self, config):
        self.config = config
        self.device = config.device
        
        print(f"\n📦 Initializing BERT trainer...")
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(config.bert_model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            config.bert_model_name,
            num_labels=2
        )
        self.model.to(self.device)
        
        print(f"   Model: {config.bert_model_name}")
        print(f"   Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def train(self, train_texts, train_labels, val_texts, val_labels):
        """Train BERT model"""
        
        # Create datasets
        train_dataset = TextDataset(
            train_texts, train_labels, self.tokenizer, self.config.max_length
        )
        val_dataset = TextDataset(
            val_texts, val_labels, self.tokenizer, self.config.max_length
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers
        )
        
        # Optimizer
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        # Scheduler
        total_steps = len(train_loader) * self.config.epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=total_steps
        )
        
        # Training loop
        best_val_acc = 0.0
        patience_counter = 0
        
        print(f"\n{'='*60}")
        print("TRAINING BERT MODEL")
        print(f"{'='*60}")
        
        for epoch in range(1, self.config.epochs + 1):
            # Training
            self.model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            pbar = tqdm(train_loader, desc=f'Epoch {epoch} [BERT Train]')
            for batch in pbar:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                optimizer.zero_grad()
                
                outputs = self.model(
                    input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                logits = outputs.logits
                
                loss.backward()
                optimizer.step()
                scheduler.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(logits, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
                
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100.*train_correct/train_total:.2f}%'
                })
            
            train_acc = 100. * train_correct / train_total
            train_loss = train_loss / len(train_loader)
            
            # Validation
            self.model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for batch in tqdm(val_loader, desc=f'Epoch {epoch} [BERT Val]'):
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['label'].to(self.device)
                    
                    outputs = self.model(
                        input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    
                    loss = outputs.loss
                    logits = outputs.logits
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(logits, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
            
            val_acc = 100. * val_correct / val_total
            val_loss = val_loss / len(val_loader)
            
            print(f"\nEpoch {epoch} Summary:")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(self.model.state_dict(), self.config.bert_save_path)
                print(f"  ✅ Best BERT model saved! (Acc: {val_acc:.2f}%)")
                patience_counter = 0
            else:
                patience_counter += 1
                
                if self.config.early_stopping and patience_counter >= self.config.early_stopping_patience:
                    print(f"  ⏹️ Early stopping triggered")
                    break
        
        print(f"\n✅ BERT training complete!")
        print(f"   Best validation accuracy: {best_val_acc:.2f}%")
        print(f"   Model saved to: {self.config.bert_save_path}")
        
        return best_val_acc / 100


# ============================================
# PERPLEXITY DETECTOR TRAINER
# ============================================

class PerplexityTrainer:
    """Train perplexity-based detector"""
    
    def __init__(self, config):
        self.config = config
        self.device = config.device
        
        print(f"\n📦 Initializing perplexity detector...")
        
        # Initialize perplexity detector
        self.detector = PerplexityDetector(
            model_name=config.perplexity_model,
            device=self.device
        )
        
        print(f"   Model: {config.perplexity_model}")
    
    def find_optimal_threshold(self, train_texts, train_labels):
        """Find optimal perplexity threshold"""
        
        print(f"\n{'='*60}")
        print("FINDING OPTIMAL PERPLEXITY THRESHOLD")
        print(f"{'='*60}")
        
        # Calculate perplexities
        perplexities = []
        for text in tqdm(train_texts, desc="Calculating perplexities"):
            ppl = self.detector.calculate_perplexity(text[:1000])  # Limit length
            perplexities.append(ppl)
        
        # Try different thresholds
        thresholds = np.linspace(20, 200, 50)
        best_threshold = 50
        best_acc = 0
        
        for threshold in thresholds:
            preds = [1 if ppl < threshold else 0 for ppl in perplexities]
            acc = accuracy_score(train_labels, preds)
            
            if acc > best_acc:
                best_acc = acc
                best_threshold = threshold
        
        print(f"\n📊 Results:")
        print(f"   Optimal threshold: {best_threshold:.2f}")
        print(f"   Training accuracy: {best_acc:.4f}")
        
        # Save threshold
        config = {
            'threshold': float(best_threshold),
            'model_name': self.config.perplexity_model,
            'accuracy': float(best_acc)
        }
        
        with open(self.config.perplexity_config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"   Config saved to: {self.config.perplexity_config_path}")
        
        return best_threshold, best_acc


# ============================================
# CUSTOM NN TRAINER
# ============================================

class CustomNNTrainer:
    """Train custom neural network"""
    
    def __init__(self, config):
        self.config = config
        self.device = config.device
        self.feature_extractor = StatisticalFeatureExtractor()
        
        print(f"\n📦 Initializing custom NN trainer...")
    
    def prepare_data(self, texts, labels):
        """Prepare data with statistical features"""
        
        print(f"   Extracting statistical features...")
        
        all_features = []
        valid_texts = []
        valid_labels = []
        
        for text, label in tqdm(zip(texts, labels), total=len(texts), desc="Extracting features"):
            if len(text.split()) >= 5:  # Minimum length
                features = self.feature_extractor.extract_features(text)
                all_features.append(features)
                valid_texts.append(text)
                valid_labels.append(label)
        
        return valid_texts, valid_labels, np.array(all_features)
    
    def train(self, train_texts, train_labels, val_texts, val_labels):
        """Train custom neural network"""
        
        # Prepare data
        train_texts, train_labels, train_features = self.prepare_data(train_texts, train_labels)
        val_texts, val_labels, val_features = self.prepare_data(val_texts, val_labels)
        
        # Tokenizer for text
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        
        # Tokenize texts
        train_encodings = tokenizer(
            train_texts,
            truncation=True,
            padding=True,
            max_length=self.config.max_length,
            return_tensors='pt'
        )
        
        val_encodings = tokenizer(
            val_texts,
            truncation=True,
            padding=True,
            max_length=self.config.max_length,
            return_tensors='pt'
        )
        
        # Create dataset class
        class CustomDataset(Dataset):
            def __init__(self, encodings, features, labels):
                self.encodings = encodings
                self.features = torch.FloatTensor(features)
                self.labels = torch.LongTensor(labels)
            
            def __len__(self):
                return len(self.labels)
            
            def __getitem__(self, idx):
                return {
                    'input_ids': self.encodings['input_ids'][idx],
                    'attention_mask': self.encodings['attention_mask'][idx],
                    'features': self.features[idx],
                    'label': self.labels[idx]
                }
        
        train_dataset = CustomDataset(train_encodings, train_features, train_labels)
        val_dataset = CustomDataset(val_encodings, val_features, val_labels)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers
        )
        
        # Initialize model
        model = TextDeepfakeNN(
            vocab_size=tokenizer.vocab_size,
            num_statistical_features=20
        )
        model.to(self.device)
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        # Training loop
        best_val_acc = 0.0
        patience_counter = 0
        
        print(f"\n{'='*60}")
        print("TRAINING CUSTOM NEURAL NETWORK")
        print(f"{'='*60}")
        
        for epoch in range(1, self.config.epochs + 1):
            # Training
            model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            pbar = tqdm(train_loader, desc=f'Epoch {epoch} [CustomNN Train]')
            for batch in pbar:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                features = batch['features'].to(self.device)
                labels = batch['label'].to(self.device)
                
                optimizer.zero_grad()
                
                outputs = model(input_ids, attention_mask, features)
                loss = criterion(outputs, labels)
                
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
                
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100.*train_correct/train_total:.2f}%'
                })
            
            train_acc = 100. * train_correct / train_total
            train_loss = train_loss / len(train_loader)
            
            # Validation
            model.eval()
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for batch in tqdm(val_loader, desc=f'Epoch {epoch} [CustomNN Val]'):
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    features = batch['features'].to(self.device)
                    labels = batch['label'].to(self.device)
                    
                    outputs = model(input_ids, attention_mask, features)
                    
                    _, predicted = torch.max(outputs, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
            
            val_acc = 100. * val_correct / val_total
            
            print(f"\nEpoch {epoch} Summary:")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"  Val Acc: {val_acc:.2f}%")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                
                # Save model
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'config': {
                        'vocab_size': tokenizer.vocab_size,
                        'num_statistical_features': 20,
                        'num_classes': 2
                    },
                    'best_acc': best_val_acc / 100
                }, self.config.model_save_path)
                
                print(f"  ✅ Best custom NN model saved! (Acc: {val_acc:.2f}%)")
                patience_counter = 0
            else:
                patience_counter += 1
                
                if self.config.early_stopping and patience_counter >= self.config.early_stopping_patience:
                    print(f"  ⏹️ Early stopping triggered")
                    break
        
        print(f"\n✅ Custom NN training complete!")
        print(f"   Best validation accuracy: {best_val_acc:.2f}%")
        print(f"   Model saved to: {self.config.model_save_path}")
        
        return best_val_acc / 100, model


# ============================================
# DATA LOADER
# ============================================

class TextDataLoader:
    """Load text datasets"""
    
    @staticmethod
    def load_json(file_path):
        """Load texts and labels from JSON"""
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        texts = []
        labels = []
        
        for item in data:
            texts.append(item['text'])
            labels.append(item['label'])  # 0=human, 1=AI
        
        return texts, labels
    
    @staticmethod
    def create_sample_data(num_samples=1000):
        """Create sample data for testing"""
        texts = []
        labels = []
        
        # Human texts (simulated)
        human_templates = [
            "I went to the store yesterday and bought some groceries. The weather was nice.",
            "My friend called me last night and we talked for hours about our plans.",
            "The movie was really good! I especially liked the ending.",
            "I need to finish this project by Friday, but I'm not sure if I'll make it.",
            "Have you seen my keys? I can't find them anywhere.",
            "Let's meet for coffee tomorrow morning around 10.",
            "I just finished reading this amazing book! Highly recommend it.",
            "The concert last night was incredible. The band played all their best songs.",
            "I'm thinking about learning a new language. Any recommendations?",
            "Just got back from vacation. It was so relaxing to get away."
        ]
        
        # AI texts (simulated)
        ai_templates = [
            "In conclusion, the implementation of artificial intelligence in healthcare presents numerous advantages.",
            "Furthermore, it is important to consider the ethical implications of such technologies.",
            "Additionally, there are several factors that contribute to the effectiveness of this approach.",
            "Moreover, the integration of these systems requires careful planning and consideration.",
            "It should be noted that while these benefits are substantial, there are also challenges to address.",
            "The results indicate a significant correlation between the variables under investigation.",
            "Based on the analysis, we can conclude that the proposed method outperforms existing approaches.",
            "In the context of modern society, these developments have far-reaching implications.",
            "To summarize, the evidence suggests that further research is needed in this area.",
            "It is worth noting that these findings align with previous studies in the field."
        ]
        
        for _ in range(num_samples // 2):
            texts.append(random.choice(human_templates))
            labels.append(0)
            
            # Add variations
            if random.random() > 0.5:
                texts[-1] += " " + random.choice(human_templates)
        
        for _ in range(num_samples // 2):
            texts.append(random.choice(ai_templates))
            labels.append(1)
            
            if random.random() > 0.5:
                texts[-1] += " " + random.choice(ai_templates)
        
        # Shuffle
        combined = list(zip(texts, labels))
        random.shuffle(combined)
        texts, labels = zip(*combined)
        
        return list(texts), list(labels)
    
    @staticmethod
    def save_json(texts, labels, file_path):
        """Save texts and labels to JSON"""
        data = []
        for text, label in zip(texts, labels):
            data.append({
                'text': text,
                'label': label
            })
        
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)


# ============================================
# MAIN TRAINER
# ============================================

class TextDetectorTrainer:
    """Main trainer for text detector"""
    
    def __init__(self, config):
        self.config = config
        self.writer = SummaryWriter(log_dir=config.log_dir)
        
        print(f"\n{'='*60}")
        print("TEXT DETECTOR TRAINER INITIALIZED")
        print(f"{'='*60}")
        print(f"Configuration:\n{config}")
        
        # Initialize component trainers
        self.bert_trainer = BERTTrainer(config) if config.use_bert else None
        self.perplexity_trainer = PerplexityTrainer(config) if config.use_perplexity else None
        self.custom_trainer = CustomNNTrainer(config) if config.use_statistical else None
        
        self.results = {}
    
    def load_data(self):
        """Load training data"""
        print(f"\n📂 Loading data...")
        
        if self.config.train_file.exists():
            train_texts, train_labels = TextDataLoader.load_json(self.config.train_file)
            val_texts, val_labels = TextDataLoader.load_json(self.config.val_file)
            
            if self.config.test_file.exists():
                test_texts, test_labels = TextDataLoader.load_json(self.config.test_file)
            else:
                test_texts, test_labels = [], []
        else:
            print("   No data files found. Creating sample data...")
            texts, labels = TextDataLoader.create_sample_data(2000)
            
            # Split
            split = int(0.7 * len(texts))
            split2 = int(0.85 * len(texts))
            
            train_texts = texts[:split]
            train_labels = labels[:split]
            
            val_texts = texts[split:split2]
            val_labels = labels[split:split2]
            
            test_texts = texts[split2:]
            test_labels = labels[split2:]
            
            # Save for future use
            self.config.data_dir.mkdir(exist_ok=True)
            TextDataLoader.save_json(train_texts, train_labels, self.config.train_file)
            TextDataLoader.save_json(val_texts, val_labels, self.config.val_file)
            TextDataLoader.save_json(test_texts, test_labels, self.config.test_file)
        
        print(f"\n📊 Dataset statistics:")
        print(f"   Train: {len(train_texts)} samples")
        print(f"   Validation: {len(val_texts)} samples")
        print(f"   Test: {len(test_texts)} samples")
        
        # Show class distribution
        train_real = sum(1 for l in train_labels if l == 0)
        train_ai = sum(1 for l in train_labels if l == 1)
        print(f"\n   Train - Real: {train_real}, AI: {train_ai}")
        
        return train_texts, train_labels, val_texts, val_labels, test_texts, test_labels
    
    def train(self):
        """Train all models"""
        
        # Load data
        train_texts, train_labels, val_texts, val_labels, test_texts, test_labels = self.load_data()
        
        # Train BERT
        if self.bert_trainer:
            bert_acc = self.bert_trainer.train(train_texts, train_labels, val_texts, val_labels)
            self.results['bert_accuracy'] = bert_acc
        
        # Train perplexity detector
        if self.perplexity_trainer:
            ppl_threshold, ppl_acc = self.perplexity_trainer.find_optimal_threshold(
                train_texts, train_labels
            )
            self.results['perplexity_threshold'] = ppl_threshold
            self.results['perplexity_accuracy'] = ppl_acc
        
        # Train custom NN
        if self.custom_trainer:
            nn_acc, nn_model = self.custom_trainer.train(
                train_texts, train_labels, val_texts, val_labels
            )
            self.results['nn_accuracy'] = nn_acc
        
        # Create ensemble config
        self._create_ensemble_config()
        
        # Evaluate on test set
        if test_texts:
            self.evaluate(test_texts, test_labels)
        
        print(f"\n{'='*60}")
        print("TRAINING COMPLETE")
        print(f"{'='*60}")
        print(f"Results: {json.dumps(self.results, indent=2)}")
        
        return self.results
    
    def _create_ensemble_config(self):
        """Create ensemble configuration"""
        
        # Default weights
        weights = {
            'bert': 0.4 if self.bert_trainer else 0,
            'perplexity': 0.3 if self.perplexity_trainer else 0,
            'nn': 0.3 if self.custom_trainer else 0
        }
        
        # Normalize
        total = sum(weights.values())
        if total > 0:
            weights = {k: v/total for k, v in weights.items()}
        
        config = {
            'bert_model': str(self.config.bert_save_path) if self.bert_trainer else None,
            'perplexity_config': str(self.config.perplexity_config_path) if self.perplexity_trainer else None,
            'nn_model': str(self.config.model_save_path) if self.custom_trainer else None,
            'weights': weights,
            'results': self.results
        }
        
        with open(self.config.ensemble_config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"\n📝 Ensemble config saved to: {self.config.ensemble_config_path}")
    
    def evaluate(self, test_texts, test_labels):
        """Evaluate on test set"""
        print(f"\n{'='*60}")
        print("EVALUATING ON TEST SET")
        print(f"{'='*60}")
        
        # Create full detector
        detector = TextDeepfakeDetector(
            use_perplexity=self.config.use_perplexity,
            use_bert=self.config.use_bert,
            use_statistical=self.config.use_statistical
        )
        
        # Evaluate
        predictions = []
        probabilities = []
        
        for text in tqdm(test_texts, desc="Testing"):
            result = detector.detect(text, return_details=False)
            predictions.append(1 if result.prediction == 'AI' else 0)
            probabilities.append(result.ai_probability)
        
        # Compute metrics
        accuracy = accuracy_score(test_labels, predictions)
        precision = precision_score(test_labels, predictions)
        recall = recall_score(test_labels, predictions)
        f1 = f1_score(test_labels, predictions)
        auc = roc_auc_score(test_labels, probabilities)
        
        print(f"\n📊 Test Results:")
        print(f"   Accuracy:  {accuracy:.4f}")
        print(f"   Precision: {precision:.4f}")
        print(f"   Recall:    {recall:.4f}")
        print(f"   F1 Score:  {f1:.4f}")
        print(f"   AUC-ROC:   {auc:.4f}")
        
        # Confusion matrix
        cm = confusion_matrix(test_labels, predictions)
        print(f"\n📊 Confusion Matrix:")
        print(f"   {cm}")
        
        # Save results
        self.results['test_accuracy'] = float(accuracy)
        self.results['test_precision'] = float(precision)
        self.results['test_recall'] = float(recall)
        self.results['test_f1'] = float(f1)
        self.results['test_auc'] = float(auc)
        self.results['confusion_matrix'] = cm.tolist()


# ============================================
# MAIN FUNCTION
# ============================================

def main():
    parser = argparse.ArgumentParser(description='Train Text Deepfake Detector')
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--data_dir', type=str, help='Path to data directory')
    parser.add_argument('--batch_size', type=int, help='Batch size')
    parser.add_argument('--epochs', type=int, help='Number of epochs')
    parser.add_argument('--lr', type=float, help='Learning rate')
    parser.add_argument('--device', type=str, help='Device to use')
    parser.add_argument('--no_bert', action='store_true', help='Disable BERT')
    parser.add_argument('--no_perplexity', action='store_true', help='Disable perplexity')
    parser.add_argument('--no_statistical', action='store_true', help='Disable statistical')
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        with open(args.config, 'r') as f:
            config_dict = json.load(f)
        config = Config(**config_dict)
    else:
        config = Config()
    
    # Override with command line arguments
    if args.data_dir:
        config.data_dir = Path(args.data_dir)
        config.train_file = config.data_dir / 'train.json'
        config.val_file = config.data_dir / 'val.json'
        config.test_file = config.data_dir / 'test.json'
    
    if args.batch_size:
        config.batch_size = args.batch_size
    
    if args.epochs:
        config.epochs = args.epochs
    
    if args.lr:
        config.learning_rate = args.lr
    
    if args.device:
        config.device = args.device
    
    if args.no_bert:
        config.use_bert = False
    
    if args.no_perplexity:
        config.use_perplexity = False
    
    if args.no_statistical:
        config.use_statistical = False
    
    # Create trainer and train
    trainer = TextDetectorTrainer(config)
    results = trainer.train()
    
    print(f"\n✅ Training complete!")
    print(f"   Models saved to: {config.model_save_path.parent}")
    print(f"   Best accuracies: BERT={results.get('bert_accuracy', 0):.4f}, "
          f"Perplexity={results.get('perplexity_accuracy', 0):.4f}, "
          f"NN={results.get('nn_accuracy', 0):.4f}")


if __name__ == "__main__":
    main()