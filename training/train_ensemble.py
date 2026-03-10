# training/train_ensemble.py
"""
Complete Training Script for Ensemble Deepfake Detection Model
Combines MobileNetV2, Xception, and EfficientNet with attention mechanism
Supports multi-GPU training, mixed precision, and comprehensive logging
"""

import os
import sys
import json
import time
import math
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

# ========== TORCH IMPORTS ==========
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler

# ========== TORCHVISION ==========
import torchvision
from torchvision import transforms, models
from torchvision.datasets import ImageFolder

# ========== SKLEARN METRICS ==========
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)

# ========== IMPORT MODELS ==========
from app.models.ensemble import DeepfakeEnsemble, EnsembleFactory, EnsembleResult
from app.models.mobilenet_model import MobileNetFactory
from app.models.xception_model import XceptionFactory
from app.models.efficientnet_model import EfficientNetFactory

# ========== UTILS ==========
from training.dataset_prep import FaceExtractor, DataAugmenter

# Set random seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)


# ============================================
# CONFIGURATION
# ============================================

class Config:
    """Configuration class for training"""
    
    # Model configuration
    model_name = 'ensemble_b3'  # ensemble_b0 to ensemble_b7
    num_classes = 2
    pretrained = True
    
    # Data configuration
    data_dir = Path('datasets/processed')
    train_dir = data_dir / 'train'
    val_dir = data_dir / 'val'
    test_dir = data_dir / 'test'
    
    # Training configuration
    batch_size = 16  # Adjust based on GPU memory
    epochs = 50
    learning_rate = 1e-4
    weight_decay = 1e-4
    momentum = 0.9
    
    # Learning rate scheduler
    lr_scheduler = 'cosine'  # 'cosine', 'step', 'plateau'
    lr_step_size = 10
    lr_gamma = 0.1
    lr_patience = 5
    lr_min = 1e-7
    
    # Optimizer
    optimizer = 'adamw'  # 'adam', 'adamw', 'sgd'
    
    # Loss function
    loss_function = 'cross_entropy'  # 'cross_entropy', 'focal', 'weighted'
    
    # Class weights (for imbalanced datasets)
    class_weights = None  # Will be computed automatically
    
    # Hardware
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_workers = 4
    pin_memory = True
    
    # Mixed precision training
    use_amp = True if device == 'cuda' else False
    
    # Distributed training
    distributed = False
    world_size = -1
    
    # Checkpointing
    save_dir = Path('checkpoints/ensemble')
    model_save_path = Path('models/ensemble_model.pt')
    log_dir = Path('logs/ensemble')
    resume_from = None  # Path to checkpoint to resume from
    
    # Augmentation
    use_augmentation = True
    cutmix_prob = 0.5
    mixup_prob = 0.5
    
    # Evaluation
    eval_interval = 1
    save_interval = 5
    log_interval = 10
    
    # Early stopping
    early_stopping = True
    early_stopping_patience = 10
    early_stopping_min_delta = 0.001
    
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        
        # Create directories
        self.save_dir.mkdir(exist_ok=True, parents=True)
        self.log_dir.mkdir(exist_ok=True, parents=True)
        self.model_save_path.parent.mkdir(exist_ok=True, parents=True)
    
    def __str__(self):
        return json.dumps(self.__dict__, indent=2, default=str)
    
    def save(self, path):
        """Save configuration to JSON"""
        with open(path, 'w') as f:
            json.dump(self.__dict__, f, indent=2, default=str)
    
    @classmethod
    def load(cls, path):
        """Load configuration from JSON"""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(**data)


# ============================================
# CUSTOM DATASET WITH AUGMENTATIONS
# ============================================

class DeepfakeDataset(Dataset):
    """Enhanced dataset for deepfake training with augmentations"""
    
    def __init__(self, root_dir, transform=None, is_train=True, config=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.is_train = is_train
        self.config = config
        
        # Get all image paths and labels
        self.samples = []
        self.classes = ['real', 'fake']
        
        for class_idx, class_name in enumerate(self.classes):
            class_dir = self.root_dir / class_name
            if class_dir.exists():
                images = list(class_dir.glob('*.jpg')) + \
                        list(class_dir.glob('*.png')) + \
                        list(class_dir.glob('*.jpeg'))
                
                for img_path in images:
                    self.samples.append((str(img_path), class_idx))
        
        # Compute class weights if needed
        if config and config.class_weights == 'auto':
            class_counts = [0, 0]
            for _, label in self.samples:
                class_counts[label] += 1
            
            total = len(self.samples)
            self.class_weights = torch.tensor([
                total / (2 * class_counts[0]),
                total / (2 * class_counts[1])
            ])
        else:
            self.class_weights = None
        
        print(f"Loaded {len(self.samples)} images from {root_dir}")
        print(f"  Real: {sum(1 for _, l in self.samples if l == 0)}")
        print(f"  Fake: {sum(1 for _, l in self.samples if l == 1)}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # Load image
        from PIL import Image
        image = Image.open(img_path).convert('RGB')
        
        # Apply CutMix or MixUp for training
        if self.is_train and self.config:
            if random.random() < getattr(self.config, 'cutmix_prob', 0):
                return self._cutmix(image, label)
            elif random.random() < getattr(self.config, 'mixup_prob', 0):
                return self._mixup(image, label)
        
        # Regular transformation
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(label, dtype=torch.long)
    
    def _cutmix(self, image, label):
        """Apply CutMix augmentation"""
        # Get random image from same batch (simplified)
        # In practice, this should be implemented in collate_fn
        return image, label
    
    def _mixup(self, image, label):
        """Apply MixUp augmentation"""
        # Similar to CutMix, should be implemented in collate_fn
        return image, label


# ============================================
# CUSTOM LOSS FUNCTIONS
# ============================================

class FocalLoss(nn.Module):
    """Focal Loss for class imbalance"""
    
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class LabelSmoothingLoss(nn.Module):
    """Cross entropy with label smoothing"""
    
    def __init__(self, num_classes, smoothing=0.1):
        super(LabelSmoothingLoss, self).__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
    
    def forward(self, inputs, targets):
        log_probs = F.log_softmax(inputs, dim=-1)
        
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.smoothing / (self.num_classes - 1))
            true_dist.scatter_(1, targets.unsqueeze(1), self.confidence)
        
        return torch.mean(torch.sum(-true_dist * log_probs, dim=-1))


# ============================================
# METRICS TRACKER
# ============================================

class MetricsTracker:
    """Track and compute metrics during training"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.losses = []
        self.predictions = []
        self.targets = []
    
    def update(self, loss, preds, targets):
        self.losses.append(loss)
        self.predictions.extend(preds.cpu().numpy())
        self.targets.extend(targets.cpu().numpy())
    
    def compute(self):
        """Compute all metrics"""
        if not self.predictions:
            return {}
        
        y_true = np.array(self.targets)
        y_pred = np.array(self.predictions)
        
        # Convert probabilities to class predictions
        if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
            y_pred_class = np.argmax(y_pred, axis=1)
            y_pred_prob = y_pred[:, 1] if y_pred.shape[1] == 2 else y_pred
        else:
            y_pred_class = (y_pred > 0.5).astype(int)
            y_pred_prob = y_pred
        
        metrics = {
            'loss': np.mean(self.losses),
            'accuracy': accuracy_score(y_true, y_pred_class),
            'precision': precision_score(y_true, y_pred_class, average='binary'),
            'recall': recall_score(y_true, y_pred_class, average='binary'),
            'f1': f1_score(y_true, y_pred_class, average='binary'),
        }
        
        # Add AUC if possible
        try:
            if len(np.unique(y_true)) > 1:
                metrics['auc'] = roc_auc_score(y_true, y_pred_prob)
        except:
            metrics['auc'] = 0.0
        
        return metrics


# ============================================
# TRAINER CLASS
# ============================================

class EnsembleTrainer:
    """Complete trainer for ensemble model"""
    
    def __init__(self, config):
        self.config = config
        self.device = config.device
        self.writer = SummaryWriter(log_dir=config.log_dir)
        
        print(f"\n{'='*60}")
        print("ENSEMBLE MODEL TRAINER INITIALIZED")
        print(f"{'='*60}")
        print(f"Configuration:\n{config}")
        
        # Initialize model
        self._init_model()
        
        # Initialize data
        self._init_data()
        
        # Initialize loss, optimizer, scheduler
        self._init_training_components()
        
        # Initialize scaler for mixed precision
        self.scaler = GradScaler() if config.use_amp else None
        
        # Metrics trackers
        self.train_metrics = MetricsTracker()
        self.val_metrics = MetricsTracker()
        
        # Best model tracking
        self.best_val_acc = 0.0
        self.best_val_loss = float('inf')
        self.early_stopping_counter = 0
        
        print(f"\n✅ Trainer initialized successfully")
    
    def _init_model(self):
        """Initialize ensemble model"""
        print(f"\n📦 Initializing ensemble model...")
        
        self.model = EnsembleFactory.create_accurate_ensemble(device=self.device)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.model.parameters() if p.requires_grad)
        
        print(f"   Total parameters: {total_params:,}")
        print(f"   Trainable parameters: {trainable_params:,}")
        
        # Log to tensorboard
        self.writer.add_text('model/architecture', str(self.model.model))
        self.writer.add_scalar('model/total_params', total_params)
        self.writer.add_scalar('model/trainable_params', trainable_params)
    
    def _init_data(self):
        """Initialize data loaders"""
        print(f"\n📂 Initializing data loaders...")
        
        # Define transforms
        train_transform = self._get_train_transform()
        val_transform = self._get_val_transform()
        
        # Create datasets
        self.train_dataset = DeepfakeDataset(
            self.config.train_dir,
            transform=train_transform,
            is_train=True,
            config=self.config
        )
        
        self.val_dataset = DeepfakeDataset(
            self.config.val_dir,
            transform=val_transform,
            is_train=False
        )
        
        if self.config.test_dir.exists():
            self.test_dataset = DeepfakeDataset(
                self.config.test_dir,
                transform=val_transform,
                is_train=False
            )
        else:
            self.test_dataset = None
        
        # Create data loaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            drop_last=True
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory
        )
        
        if self.test_dataset:
            self.test_loader = DataLoader(
                self.test_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=self.config.num_workers,
                pin_memory=self.config.pin_memory
            )
        
        print(f"\n📊 Dataset statistics:")
        print(f"   Train: {len(self.train_dataset)} images")
        print(f"   Validation: {len(self.val_dataset)} images")
        if self.test_dataset:
            print(f"   Test: {len(self.test_dataset)} images")
    
    def _get_train_transform(self):
        """Get training transforms with augmentation"""
        transforms_list = [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1
            ),
        ]
        
        if self.config.use_augmentation:
            transforms_list.extend([
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
            ])
        
        transforms_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.1))
        ])
        
        return transforms.Compose(transforms_list)
    
    def _get_val_transform(self):
        """Get validation transforms"""
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def _init_training_components(self):
        """Initialize loss, optimizer, scheduler"""
        
        # Loss function
        if self.config.loss_function == 'focal':
            self.criterion = FocalLoss(alpha=1, gamma=2)
        elif self.config.loss_function == 'label_smoothing':
            self.criterion = LabelSmoothingLoss(num_classes=self.config.num_classes)
        else:
            self.criterion = nn.CrossEntropyLoss()
        
        # Optimizer
        if self.config.optimizer == 'adam':
            self.optimizer = optim.Adam(
                self.model.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer == 'adamw':
            self.optimizer = optim.AdamW(
                self.model.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer == 'sgd':
            self.optimizer = optim.SGD(
                self.model.model.parameters(),
                lr=self.config.learning_rate,
                momentum=self.config.momentum,
                weight_decay=self.config.weight_decay
            )
        
        # Learning rate scheduler
        if self.config.lr_scheduler == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.epochs,
                eta_min=self.config.lr_min
            )
        elif self.config.lr_scheduler == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.lr_step_size,
                gamma=self.config.lr_gamma
            )
        elif self.config.lr_scheduler == 'plateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=self.config.lr_patience,
                min_lr=self.config.lr_min
            )
        else:
            self.scheduler = None
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.model.train()
        self.train_metrics.reset()
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch} [Train]')
        
        for batch_idx, (images, labels) in enumerate(pbar):
            # Move to device
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass with mixed precision
            if self.scaler:
                with autocast():
                    outputs = self.model.model(images)
                    loss = self.criterion(outputs, labels)
                
                # Backward pass with scaler
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
            
            # Get predictions
            probs = F.softmax(outputs, dim=1)
            
            # Update metrics
            self.train_metrics.update(loss.item(), probs.detach(), labels)
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{self.train_metrics.losses[-1]:.4f}',
                'avg_loss': f'{np.mean(self.train_metrics.losses):.4f}'
            })
            
            # Log to tensorboard
            global_step = (epoch - 1) * len(self.train_loader) + batch_idx
            if global_step % self.config.log_interval == 0:
                self.writer.add_scalar('train/step_loss', loss.item(), global_step)
                self.writer.add_scalar('train/lr', self.optimizer.param_groups[0]['lr'], global_step)
        
        # Compute epoch metrics
        metrics = self.train_metrics.compute()
        
        # Log to tensorboard
        self.writer.add_scalar('train/epoch_loss', metrics['loss'], epoch)
        self.writer.add_scalar('train/accuracy', metrics['accuracy'], epoch)
        self.writer.add_scalar('train/precision', metrics['precision'], epoch)
        self.writer.add_scalar('train/recall', metrics['recall'], epoch)
        self.writer.add_scalar('train/f1', metrics['f1'], epoch)
        if 'auc' in metrics:
            self.writer.add_scalar('train/auc', metrics['auc'], epoch)
        
        return metrics
    
    def validate(self, epoch):
        """Validate model"""
        self.model.model.eval()
        self.val_metrics.reset()
        
        pbar = tqdm(self.val_loader, desc=f'Epoch {epoch} [Val]')
        
        with torch.no_grad():
            for images, labels in pbar:
                # Move to device
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                outputs = self.model.model(images)
                loss = self.criterion(outputs, labels)
                
                # Get predictions
                probs = F.softmax(outputs, dim=1)
                
                # Update metrics
                self.val_metrics.update(loss.item(), probs, labels)
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f'{self.val_metrics.losses[-1]:.4f}',
                    'avg_loss': f'{np.mean(self.val_metrics.losses):.4f}'
                })
        
        # Compute metrics
        metrics = self.val_metrics.compute()
        
        # Log to tensorboard
        self.writer.add_scalar('val/epoch_loss', metrics['loss'], epoch)
        self.writer.add_scalar('val/accuracy', metrics['accuracy'], epoch)
        self.writer.add_scalar('val/precision', metrics['precision'], epoch)
        self.writer.add_scalar('val/recall', metrics['recall'], epoch)
        self.writer.add_scalar('val/f1', metrics['f1'], epoch)
        if 'auc' in metrics:
            self.writer.add_scalar('val/auc', metrics['auc'], epoch)
        
        return metrics
    
    def train(self):
        """Full training loop"""
        print(f"\n{'='*60}")
        print("STARTING TRAINING")
        print(f"{'='*60}\n")
        
        # Save initial config
        self.config.save(self.config.save_dir / 'config.json')
        
        for epoch in range(1, self.config.epochs + 1):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch}/{self.config.epochs}")
            print(f"{'='*60}")
            
            # Train
            train_metrics = self.train_epoch(epoch)
            
            # Validate
            if epoch % self.config.eval_interval == 0:
                val_metrics = self.validate(epoch)
                
                # Print metrics
                print(f"\n📊 Epoch {epoch} Summary:")
                print(f"   Train - Loss: {train_metrics['loss']:.4f}, "
                      f"Acc: {train_metrics['accuracy']:.4f}, "
                      f"F1: {train_metrics['f1']:.4f}")
                print(f"   Val   - Loss: {val_metrics['loss']:.4f}, "
                      f"Acc: {val_metrics['accuracy']:.4f}, "
                      f"F1: {val_metrics['f1']:.4f}")
                
                # Check for best model
                if val_metrics['accuracy'] > self.best_val_acc:
                    self.best_val_acc = val_metrics['accuracy']
                    self.best_val_loss = val_metrics['loss']
                    self.early_stopping_counter = 0
                    
                    # Save best model
                    self.model.save_model(str(self.config.model_save_path))
                    
                    # Save metrics
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'best_val_acc': self.best_val_acc,
                        'best_val_loss': self.best_val_loss,
                        'train_metrics': train_metrics,
                        'val_metrics': val_metrics,
                        'config': self.config.__dict__
                    }, self.config.save_dir / 'best_model.pt')
                    
                    print(f"   ✅ New best model! Accuracy: {self.best_val_acc:.4f}")
                else:
                    self.early_stopping_counter += 1
                    
                    # Early stopping
                    if self.config.early_stopping and \
                       self.early_stopping_counter >= self.config.early_stopping_patience:
                        print(f"\n⏹️ Early stopping triggered after {epoch} epochs")
                        break
            
            # Update scheduler
            if self.scheduler:
                if self.config.lr_scheduler == 'plateau' and val_metrics:
                    self.scheduler.step(val_metrics['loss'])
                else:
                    self.scheduler.step()
            
            # Save checkpoint
            if epoch % self.config.save_interval == 0:
                checkpoint_path = self.config.save_dir / f'checkpoint_epoch{epoch}.pt'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
                    'best_val_acc': self.best_val_acc,
                    'best_val_loss': self.best_val_loss,
                    'config': self.config.__dict__
                }, checkpoint_path)
                print(f"   💾 Checkpoint saved to {checkpoint_path}")
        
        print(f"\n{'='*60}")
        print("TRAINING COMPLETE")
        print(f"{'='*60}")
        print(f"Best validation accuracy: {self.best_val_acc:.4f}")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print(f"Model saved to: {self.config.model_save_path}")
        
        # Final evaluation on test set
        if self.test_dataset:
            self.evaluate()
        
        self.writer.close()
        return self.model
    
    def evaluate(self):
        """Evaluate model on test set"""
        print(f"\n{'='*60}")
        print("EVALUATING ON TEST SET")
        print(f"{'='*60}")
        
        self.model.model.eval()
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for images, labels in tqdm(self.test_loader, desc='Testing'):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model.model(images)
                probs = F.softmax(outputs, dim=1)
                
                all_preds.extend(torch.argmax(probs, dim=1).cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())
        
        # Compute metrics
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds)
        recall = recall_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds)
        auc = roc_auc_score(all_labels, all_probs)
        
        print(f"\n📊 Test Results:")
        print(f"   Accuracy:  {accuracy:.4f}")
        print(f"   Precision: {precision:.4f}")
        print(f"   Recall:    {recall:.4f}")
        print(f"   F1 Score:  {f1:.4f}")
        print(f"   AUC-ROC:   {auc:.4f}")
        
        # Confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        print(f"\n📊 Confusion Matrix:")
        print(f"   {cm}")
        
        # Classification report
        print(f"\n📊 Classification Report:")
        print(classification_report(all_labels, all_preds, 
                                    target_names=['Real', 'Fake']))
        
        # Save results
        results = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'auc': float(auc),
            'confusion_matrix': cm.tolist()
        }
        
        with open(self.config.save_dir / 'test_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        # Plot ROC curve
        fpr, tpr, _ = roc_curve(all_labels, all_probs)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve - Test Set')
        plt.legend()
        plt.grid(True)
        plt.savefig(self.config.save_dir / 'roc_curve.png')
        plt.close()
        
        return results
    
    def resume(self, checkpoint_path):
        """Resume training from checkpoint"""
        print(f"\n📦 Resuming from checkpoint: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model
        self.model.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler
        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Load metrics
        self.best_val_acc = checkpoint['best_val_acc']
        self.best_val_loss = checkpoint['best_val_loss']
        
        start_epoch = checkpoint['epoch'] + 1
        
        print(f"   Resuming from epoch {start_epoch}")
        print(f"   Best validation accuracy: {self.best_val_acc:.4f}")
        
        return start_epoch


# ============================================
# MAIN FUNCTION
# ============================================

def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Train Ensemble Deepfake Detector')
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')
    parser.add_argument('--data_dir', type=str, help='Path to data directory')
    parser.add_argument('--batch_size', type=int, help='Batch size')
    parser.add_argument('--epochs', type=int, help='Number of epochs')
    parser.add_argument('--lr', type=float, help='Learning rate')
    parser.add_argument('--device', type=str, help='Device to use')
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        config = Config.load(args.config)
    else:
        config = Config()
    
    # Override with command line arguments
    if args.data_dir:
        config.data_dir = Path(args.data_dir)
        config.train_dir = config.data_dir / 'train'
        config.val_dir = config.data_dir / 'val'
        config.test_dir = config.data_dir / 'test'
    
    if args.batch_size:
        config.batch_size = args.batch_size
    
    if args.epochs:
        config.epochs = args.epochs
    
    if args.lr:
        config.learning_rate = args.lr
    
    if args.device:
        config.device = args.device
    
    # Create trainer
    trainer = EnsembleTrainer(config)
    
    # Resume if specified
    start_epoch = 1
    if args.resume:
        start_epoch = trainer.resume(args.resume)
    
    # Train
    model = trainer.train()
    
    print(f"\n✅ Training complete!")
    print(f"   Model saved to: {config.model_save_path}")
    print(f"   Best accuracy: {trainer.best_val_acc:.4f}")


if __name__ == "__main__":
    main()