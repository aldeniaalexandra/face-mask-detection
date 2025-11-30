"""
This module handles data loading and preprocessing for the face mask detection task.
We parse XML annotations (Pascal VOC format) and extract face regions with their labels.
"""

import os
import cv2
import torch
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
from torch.utils.data import Dataset
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2


class FaceMaskDataset(Dataset):
    """
    Custom Dataset for Face Mask Detection
    
    The dataset contains images with XML annotations (Pascal VOC format).
    Each annotation contains bounding boxes with one of three classes:
    - with_mask: person wearing a mask
    - without_mask: person not wearing a mask
    - mask_weared_incorrect: mask worn incorrectly
    
    Design Decision: We treat this as a classification problem by extracting
    face regions from bounding boxes rather than object detection, as this
    simplifies the model architecture and focuses on the mask classification task.
    """
    
    def __init__(self, image_dir, annotation_dir, transform=None, target_size=(224, 224)):
        """
        Args:
            image_dir: Directory containing images
            annotation_dir: Directory containing XML annotations
            transform: Albumentations transform pipeline
            target_size: Size to resize images (default: 224x224 for pretrained models)
        """
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.transform = transform
        self.target_size = target_size
        
        # Class mapping - we use 3 classes as per dataset
        self.class_to_idx = {
            'with_mask': 0,
            'without_mask': 1,
            'mask_weared_incorrect': 2
        }
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        
        # Load all annotations
        self.samples = self._load_annotations()
        
        print(f"Loaded {len(self.samples)} face samples")
        print(f"Class distribution:")
        labels = [s['label'] for s in self.samples]
        for class_name, idx in self.class_to_idx.items():
            count = labels.count(idx)
            print(f"  {class_name}: {count} ({count/len(labels)*100:.1f}%)")
    
    def _load_annotations(self):
        """
        Parse XML annotations and extract face bounding boxes with labels
        
        Design Decision: We extract each face as a separate sample rather than
        treating this as multi-object detection. This allows us to use standard
        classification architectures and creates more training samples.
        """
        samples = []
        
        annotation_files = [f for f in os.listdir(self.annotation_dir) if f.endswith('.xml')]
        
        for ann_file in annotation_files:
            ann_path = os.path.join(self.annotation_dir, ann_file)
            tree = ET.parse(ann_path)
            root = tree.getroot()
            
            # Get image filename
            filename = root.find('filename').text
            image_path = os.path.join(self.image_dir, filename)
            
            # Skip if image doesn't exist
            if not os.path.exists(image_path):
                continue
            
            # Extract all bounding boxes from this image
            for obj in root.findall('object'):
                label = obj.find('name').text
                
                # Skip unknown labels
                if label not in self.class_to_idx:
                    continue
                
                # Get bounding box coordinates
                bbox = obj.find('bndbox')
                xmin = int(bbox.find('xmin').text)
                ymin = int(bbox.find('ymin').text)
                xmax = int(bbox.find('xmax').text)
                ymax = int(bbox.find('ymax').text)
                
                samples.append({
                    'image_path': image_path,
                    'bbox': (xmin, ymin, xmax, ymax),
                    'label': self.class_to_idx[label],
                    'label_name': label
                })
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        Load and preprocess a single sample
        
        Process:
        1. Load full image
        2. Crop face region using bounding box
        3. Resize to target size
        4. Apply augmentations (if training)
        5. Normalize and convert to tensor
        """
        sample = self.samples[idx]
        
        # Load image
        image = cv2.imread(sample['image_path'])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Crop face region with small padding
        # Design Decision: Add 10% padding around bbox to include context
        xmin, ymin, xmax, ymax = sample['bbox']
        h, w = image.shape[:2]
        
        # Add padding (10% of bbox size)
        bbox_w = xmax - xmin
        bbox_h = ymax - ymin
        pad_x = int(bbox_w * 0.1)
        pad_y = int(bbox_h * 0.1)
        
        xmin = max(0, xmin - pad_x)
        ymin = max(0, ymin - pad_y)
        xmax = min(w, xmax + pad_x)
        ymax = min(h, ymax + pad_y)
        
        face_crop = image[ymin:ymax, xmin:xmax]
        
        # Resize to target size
        face_crop = cv2.resize(face_crop, self.target_size)
        
        # Apply augmentations
        if self.transform:
            augmented = self.transform(image=face_crop)
            face_crop = augmented['image']
        
        label = sample['label']
        
        return face_crop, label


def get_transforms(mode='train'):
    """
    Get augmentation pipeline for train/validation
    
    Design Decisions:
    - Training augmentations: We use moderate augmentations to improve generalization
      without distorting faces too much (rotation, brightness, contrast)
    - Validation: Only normalization, no augmentation
    - Normalization: ImageNet stats since we use pretrained models
    """
    
    if mode == 'train':
        return A.Compose([
            # Geometric augmentations
            A.HorizontalFlip(p=0.5),  # Faces can be flipped
            A.Rotate(limit=15, p=0.5),  # Small rotation for natural variation
            
            # Color augmentations - moderate to handle different lighting
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.3),
            
            # Blur to handle different image qualities
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
            
            # Normalize using ImageNet statistics (standard for pretrained models)
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    else:  # validation/test
        return A.Compose([
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])


def create_data_loaders(data_dir, batch_size=32, num_workers=4, train_split=0.8):
    """
    Create train and validation data loaders
    
    Args:
        data_dir: Root directory containing 'images' and 'annotations' folders
        batch_size: Batch size for training
        num_workers: Number of worker processes for data loading
        train_split: Fraction of data to use for training
    
    Design Decision: We use 80-20 train-val split by default. This provides
    enough validation data to monitor overfitting while maximizing training data.
    """
    from sklearn.model_selection import train_test_split
    from torch.utils.data import DataLoader, Subset
    
    image_dir = os.path.join(data_dir, 'images')
    annotation_dir = os.path.join(data_dir, 'annotations')
    
    # Create full dataset
    full_dataset = FaceMaskDataset(
        image_dir=image_dir,
        annotation_dir=annotation_dir,
        transform=None,  # Will be set per split
        target_size=(224, 224)
    )
    
    # Split indices
    indices = list(range(len(full_dataset)))
    train_indices, val_indices = train_test_split(
        indices, 
        train_size=train_split, 
        random_state=42,
        stratify=[full_dataset.samples[i]['label'] for i in indices]  # Stratified split
    )
    
    # Create train dataset with augmentation
    train_dataset = FaceMaskDataset(
        image_dir=image_dir,
        annotation_dir=annotation_dir,
        transform=get_transforms('train'),
        target_size=(224, 224)
    )
    train_dataset.samples = [full_dataset.samples[i] for i in train_indices]
    
    # Create validation dataset without augmentation
    val_dataset = FaceMaskDataset(
        image_dir=image_dir,
        annotation_dir=annotation_dir,
        transform=get_transforms('val'),
        target_size=(224, 224)
    )
    val_dataset.samples = [full_dataset.samples[i] for i in val_indices]
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, full_dataset.class_to_idx