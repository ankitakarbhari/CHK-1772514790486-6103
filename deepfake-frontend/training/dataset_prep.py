# Placeholder file for dataset_prep.py
# training/dataset_prep.py
"""
Dataset Preparation Script for Deepfake Detection
Downloads and preprocesses datasets for training
Supports FaceForensics++, Celeb-DF, DFDC, and custom datasets
"""

import os
import sys
import json
import time
import shutil
import logging
import requests
import zipfile
import tarfile
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
import hashlib
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from app.utils.face_detection import FaceDetectionEnsemble, FacePreprocessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================
# DATASET CONFIGURATION
# ============================================

DATASET_CONFIGS = {
    'faceforensics': {
        'url': 'https://github.com/ondyari/FaceForensics',
        'description': 'FaceForensics++ dataset for face manipulation',
        'classes': ['real', 'fake'],
        'size_gb': 150,
        'frames': 300
    },
    'celeba_df': {
        'url': 'https://github.com/yuezunli/celeb-deepfakeforensics',
        'description': 'Celeb-DF dataset for deepfake detection',
        'classes': ['real', 'fake'],
        'size_gb': 75,
        'frames': 300
    },
    'dfdc': {
        'url': 'https://www.kaggle.com/c/deepfake-detection-challenge/data',
        'description': 'Deepfake Detection Challenge dataset',
        'classes': ['real', 'fake'],
        'size_gb': 470,
        'frames': 300
    },
    'custom': {
        'description': 'Custom dataset',
        'classes': ['real', 'fake'],
        'size_gb': 0
    }
}


# ============================================
# DATASET DOWNLOADER
# ============================================

class DatasetDownloader:
    """Download deepfake datasets"""
    
    def __init__(self, download_dir: str = 'datasets'):
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(exist_ok=True, parents=True)
        
    def download_faceforensics(self, num_videos: int = 100):
        """Download FaceForensics++ dataset"""
        logger.info("Downloading FaceForensics++ dataset...")
        
        # Note: FaceForensics requires manual download
        logger.warning("FaceForensics++ requires manual download from:")
        logger.warning("  https://github.com/ondyari/FaceForensics")
        
        # Create directory structure
        ff_dir = self.download_dir / 'faceforensics'
        ff_dir.mkdir(exist_ok=True)
        
        for split in ['train', 'val', 'test']:
            (ff_dir / split / 'real').mkdir(exist_ok=True, parents=True)
            (ff_dir / split / 'fake').mkdir(exist_ok=True, parents=True)
        
        return ff_dir
    
    def download_kaggle_dataset(self, dataset_name: str):
        """Download dataset from Kaggle"""
        logger.info(f"Downloading {dataset_name} from Kaggle...")
        
        try:
            import kaggle
            
            # Download dataset
            kaggle.api.authenticate()
            kaggle.api.dataset_download_files(
                dataset_name,
                path=str(self.download_dir),
                unzip=True
            )
            logger.info(f"✅ Downloaded {dataset_name}")
            
        except ImportError:
            logger.error("Kaggle API not installed. Install with: pip install kaggle")
        except Exception as e:
            logger.error(f"Kaggle download error: {str(e)}")
    
    def create_sample_dataset(self, output_dir: str = 'datasets/sample', 
                              num_samples: int = 1000):
        """Create a small sample dataset for testing"""
        logger.info(f"Creating sample dataset with {num_samples} images...")
        
        sample_dir = Path(output_dir)
        sample_dir.mkdir(exist_ok=True, parents=True)
        
        # Create directories
        for split in ['train', 'val', 'test']:
            for cls in ['real', 'fake']:
                (sample_dir / split / cls).mkdir(exist_ok=True, parents=True)
        
        # Generate random images
        logger.info("Generating random images...")
        
        splits = {
            'train': int(0.7 * num_samples),
            'val': int(0.15 * num_samples),
            'test': int(0.15 * num_samples)
        }
        
        for split, count in splits.items():
            for cls in ['real', 'fake']:
                cls_count = count // 2
                cls_dir = sample_dir / split / cls
                
                for i in range(cls_count):
                    # Create random image
                    img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
                    
                    # Save
                    img_path = cls_dir / f"{cls}_{i:04d}.jpg"
                    cv2.imwrite(str(img_path), img)
                    
                    if i % 100 == 0:
                        logger.info(f"  Generated {i}/{cls_count} {cls} images for {split}")
        
        logger.info(f"✅ Sample dataset created at {sample_dir}")
        return sample_dir


# ============================================
# FACE EXTRACTOR
# ============================================

class FaceExtractor:
    """Extract faces from videos/images for training"""
    
    def __init__(self, face_detector=None, target_size=(224, 224)):
        self.face_detector = face_detector or FaceDetectionEnsemble()
        self.preprocessor = FacePreprocessor(target_size=target_size)
        
    def extract_from_video(self, video_path: str, output_dir: str, 
                           max_frames: int = 30) -> List[str]:
        """Extract faces from video frames"""
        import cv2
        
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_indices = np.linspace(0, total_frames-1, max_frames, dtype=int)
        
        extracted_paths = []
        
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            
            if ret:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Detect faces
                face_result = self.face_detector.detect(frame_rgb)
                
                for i, face in enumerate(face_result.faces):
                    # Extract face
                    face_img = self.preprocessor.align_face(face, frame_rgb)
                    
                    # Save
                    face_path = Path(output_dir) / f"frame_{idx}_face_{i}.jpg"
                    cv2.imwrite(str(face_path), cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR))
                    extracted_paths.append(str(face_path))
        
        cap.release()
        return extracted_paths
    
    def extract_from_image(self, image_path: str, output_dir: str) -> List[str]:
        """Extract faces from single image"""
        # Load image
        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        face_result = self.face_detector.detect(img_rgb)
        
        extracted_paths = []
        
        for i, face in enumerate(face_result.faces):
            # Extract face
            face_img = self.preprocessor.align_face(face, img_rgb)
            
            # Save
            face_path = Path(output_dir) / f"{Path(image_path).stem}_face_{i}.jpg"
            cv2.imwrite(str(face_path), cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR))
            extracted_paths.append(str(face_path))
        
        return extracted_paths
    
    def process_directory(self, input_dir: str, output_dir: str, 
                         pattern: str = "*.jpg", recursive: bool = True):
        """Process all images in directory"""
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        
        # Find all images
        if recursive:
            images = list(input_path.rglob(pattern))
        else:
            images = list(input_path.glob(pattern))
        
        logger.info(f"Processing {len(images)} images...")
        
        for img_path in tqdm(images, desc="Extracting faces"):
            try:
                self.extract_from_image(str(img_path), str(output_path))
            except Exception as e:
                logger.error(f"Error processing {img_path}: {str(e)}")
        
        logger.info(f"✅ Processed {len(images)} images, faces saved to {output_dir}")


# ============================================
# DATA SPLITTER
# ============================================

class DataSplitter:
    """Split dataset into train/val/test sets"""
    
    def __init__(self, train_ratio: float = 0.7, val_ratio: float = 0.15, 
                 test_ratio: float = 0.15, random_seed: int = 42):
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.random_seed = random_seed
        
    def split_files(self, file_paths: List[str], labels: List[int]) -> Dict:
        """Split files into train/val/test"""
        from sklearn.model_selection import train_test_split
        
        # First split: train vs temp
        X_train, X_temp, y_train, y_temp = train_test_split(
            file_paths, labels,
            test_size=(self.val_ratio + self.test_ratio),
            random_state=self.random_seed,
            stratify=labels
        )
        
        # Second split: val vs test
        val_size = self.val_ratio / (self.val_ratio + self.test_ratio)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp,
            test_size=1-val_size,
            random_state=self.random_seed,
            stratify=y_temp
        )
        
        return {
            'train': list(zip(X_train, y_train)),
            'val': list(zip(X_val, y_val)),
            'test': list(zip(X_test, y_test))
        }
    
    def split_directory(self, input_dir: str, output_dir: str):
        """Split directory with class subfolders"""
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        
        # Get classes
        classes = [d for d in input_path.iterdir() if d.is_dir()]
        class_names = [c.name for c in classes]
        
        all_files = []
        all_labels = []
        
        for class_idx, class_dir in enumerate(classes):
            files = list(class_dir.glob("*.[jJ][pP][gG]")) + \
                    list(class_dir.glob("*.[pP][nN][gG]")) + \
                    list(class_dir.glob("*.[jJ][pP][eE][gG]"))
            
            all_files.extend(files)
            all_labels.extend([class_idx] * len(files))
        
        # Split
        split_data = self.split_files(all_files, all_labels)
        
        # Create output directories
        for split in ['train', 'val', 'test']:
            for class_name in class_names:
                (output_path / split / class_name).mkdir(exist_ok=True, parents=True)
        
        # Copy files
        for split, data in split_data.items():
            for file_path, label in tqdm(data, desc=f"Copying {split}"):
                dest_dir = output_path / split / class_names[label]
                dest_path = dest_dir / file_path.name
                
                # Copy or create symlink
                if not dest_path.exists():
                    shutil.copy2(str(file_path), str(dest_path))
        
        logger.info(f"✅ Dataset split complete:")
        logger.info(f"   Train: {len(split_data['train'])}")
        logger.info(f"   Val: {len(split_data['val'])}")
        logger.info(f"   Test: {len(split_data['test'])}")
        
        return split_data


# ============================================
# DATA AUGMENTATION
# ============================================

class DataAugmenter:
    """Apply augmentations to training data"""
    
    def __init__(self):
        import albumentations as A
        
        self.train_aug = A.Compose([
            A.RandomResizedCrop(224, 224, scale=(0.8, 1.0)),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=15, p=0.3),
            A.OneOf([
                A.GaussianBlur(blur_limit=3, p=0.3),
                A.MedianBlur(blur_limit=3, p=0.3),
                A.GaussNoise(var_limit=(10.0, 30.0), p=0.2),
            ], p=0.3),
            A.OneOf([
                A.CLAHE(clip_limit=2, p=0.3),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
                A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.3),
            ], p=0.3),
            A.CoarseDropout(max_holes=8, max_height=32, max_width=32, fill_value=0, p=0.2),
        ])
        
        self.val_aug = A.Compose([
            A.Resize(224, 224),
        ])
    
    def augment_image(self, image: np.ndarray, train: bool = True) -> np.ndarray:
        """Apply augmentation to single image"""
        if train:
            augmented = self.train_aug(image=image)
        else:
            augmented = self.val_aug(image=image)
        
        return augmented['image']
    
    def augment_dataset(self, input_dir: str, output_dir: str, 
                       augment_factor: int = 3):
        """Augment entire dataset"""
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        
        # Get all images
        images = list(input_path.rglob("*.[jJ][pP][gG]")) + \
                 list(input_path.rglob("*.[pP][nN][gG]"))
        
        logger.info(f"Augmenting {len(images)} images (factor: {augment_factor})...")
        
        for img_path in tqdm(images, desc="Augmenting"):
            # Load image
            img = cv2.imread(str(img_path))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Save original
            rel_path = img_path.relative_to(input_path)
            dest_path = output_path / rel_path
            dest_path.parent.mkdir(exist_ok=True, parents=True)
            cv2.imwrite(str(dest_path), img)
            
            # Generate augmented versions
            for i in range(augment_factor):
                aug_img = self.augment_image(img_rgb, train=True)
                aug_img_bgr = cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR)
                
                aug_path = dest_path.parent / f"{img_path.stem}_aug{i}{img_path.suffix}"
                cv2.imwrite(str(aug_path), aug_img_bgr)
        
        logger.info(f"✅ Augmentation complete. Generated {len(images) * (augment_factor + 1)} images")


# ============================================
# DATASET ANALYZER
# ============================================

class DatasetAnalyzer:
    """Analyze dataset statistics"""
    
    def __init__(self):
        pass
    
    def analyze_directory(self, data_dir: str) -> Dict:
        """Analyze dataset directory"""
        data_path = Path(data_dir)
        
        stats = {
            'total_images': 0,
            'classes': {},
            'splits': {}
        }
        
        # Analyze splits
        for split in ['train', 'val', 'test']:
            split_path = data_path / split
            if split_path.exists():
                split_stats = self._analyze_split(split_path)
                stats['splits'][split] = split_stats
                stats['total_images'] += sum(split_stats['class_counts'].values())
        
        return stats
    
    def _analyze_split(self, split_path: Path) -> Dict:
        """Analyze a single split"""
        class_counts = {}
        class_sizes = {}
        
        for class_dir in split_path.iterdir():
            if class_dir.is_dir():
                # Count images
                images = list(class_dir.glob("*.[jJ][pP][gG]")) + \
                         list(class_dir.glob("*.[pP][nN][gG]"))
                
                class_counts[class_dir.name] = len(images)
                
                # Calculate average size
                sizes = []
                for img_path in images[:100]:  # Sample first 100
                    sizes.append(img_path.stat().st_size)
                
                class_sizes[class_dir.name] = {
                    'avg_bytes': np.mean(sizes) if sizes else 0,
                    'total_mb': sum(sizes) / (1024*1024)
                }
        
        return {
            'class_counts': class_counts,
            'total': sum(class_counts.values()),
            'class_sizes': class_sizes
        }
    
    def generate_report(self, data_dir: str, output_path: str):
        """Generate HTML report"""
        stats = self.analyze_directory(data_dir)
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Dataset Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1 {{ color: #333; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <h1>Dataset Analysis Report</h1>
            <p>Total Images: {stats['total_images']}</p>
            
            <h2>Splits</h2>
        """
        
        for split, split_stats in stats['splits'].items():
            html += f"<h3>{split.capitalize()}</h3>"
            html += "<table>"
            html += "<tr><th>Class</th><th>Count</th><th>Avg Size (KB)</th><th>Total (MB)</th></tr>"
            
            for class_name, count in split_stats['class_counts'].items():
                class_size = split_stats['class_sizes'].get(class_name, {})
                avg_kb = class_size.get('avg_bytes', 0) / 1024
                total_mb = class_size.get('total_mb', 0)
                
                html += f"<tr><td>{class_name}</td><td>{count}</td><td>{avg_kb:.2f}</td><td>{total_mb:.2f}</td></tr>"
            
            html += f"<tr><td><strong>Total</strong></td><td><strong>{split_stats['total']}</strong></td><td></td><td></td></tr>"
            html += "</table>"
        
        html += """
        </body>
        </html>
        """
        
        with open(output_path, 'w') as f:
            f.write(html)
        
        logger.info(f"Report saved to {output_path}")


# ============================================
# MAIN EXECUTION
# ============================================

def main():
    """Main dataset preparation function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Prepare datasets for deepfake detection')
    parser.add_argument('--download', type=str, help='Download dataset (faceforensics/celeba_df/dfdc)')
    parser.add_argument('--extract-faces', type=str, help='Extract faces from videos/images')
    parser.add_argument('--output-dir', type=str, default='datasets/processed', help='Output directory')
    parser.add_argument('--split', type=str, help='Split dataset into train/val/test')
    parser.add_argument('--augment', type=str, help='Augment dataset')
    parser.add_argument('--analyze', type=str, help='Analyze dataset')
    parser.add_argument('--sample', type=int, help='Create sample dataset with N images')
    
    args = parser.parse_args()
    
    if args.download:
        downloader = DatasetDownloader()
        if args.download == 'faceforensics':
            downloader.download_faceforensics()
        else:
            downloader.download_kaggle_dataset(args.download)
    
    if args.extract_faces:
        extractor = FaceExtractor()
        extractor.process_directory(args.extract_faces, args.output_dir)
    
    if args.split:
        splitter = DataSplitter()
        splitter.split_directory(args.split, args.output_dir)
    
    if args.augment:
        augmenter = DataAugmenter()
        augmenter.augment_dataset(args.augment, args.output_dir)
    
    if args.analyze:
        analyzer = DatasetAnalyzer()
        analyzer.generate_report(args.analyze, 'dataset_report.html')
    
    if args.sample:
        downloader = DatasetDownloader()
        downloader.create_sample_dataset(num_samples=args.sample)


if __name__ == "__main__":
    main()