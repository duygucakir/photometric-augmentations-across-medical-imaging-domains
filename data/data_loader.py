"""
Phase 3 - Unified Data Loading and Splitting Module
====================================================
Handles all three datasets: APTOS, ISIC Skin, HyperKvasir
Generates reproducible 5-fold stratified splits (60/20/20)
Saves fold assignments to disk for traceability.
"""

import os
import json
import logging
import hashlib
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from collections import Counter

from sklearn.model_selection import StratifiedKFold
from PIL import Image, ImageFile
from torch.utils.data import Dataset
from torchvision import transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True
logger = logging.getLogger(__name__)


def _portable_path(path, base_dir):
    """Store paths inside the repository relative to its root."""
    absolute_path = os.path.abspath(path)
    try:
        if os.path.commonpath([absolute_path, base_dir]) == base_dir:
            return os.path.relpath(absolute_path, base_dir)
    except ValueError:
        pass
    return absolute_path


# =============================================================================
# Dataset Classes
# =============================================================================

class BaseImageDataset(Dataset):
    """Base dataset class with image paths and labels."""
    
    def __init__(self, image_paths, labels, transform=None, preprocessing_fn=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.preprocessing_fn = preprocessing_fn
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        img = Image.open(img_path).convert('RGB')
        
        if self.preprocessing_fn is not None:
            img = self.preprocessing_fn(img)
        
        if self.transform is not None:
            img = self.transform(img)
            
        return img, int(label)


# =============================================================================
# Data Loading Functions
# =============================================================================

def load_aptos_data(config):
    """Load APTOS dataset file paths and labels."""
    base = config['data']['base_dir']
    ds_cfg = config['data']['datasets']['aptos']
    
    # Load from CSV files
    train_csv = os.path.join(base, ds_cfg['train_csv'])
    valid_csv = os.path.join(base, ds_cfg['valid_csv'])
    test_csv = os.path.join(base, ds_cfg['test_csv'])
    
    train_img_dir = os.path.join(base, ds_cfg['train_img_dir'])
    val_img_dir = os.path.join(base, ds_cfg['val_img_dir'])
    
    all_paths = []
    all_labels = []
    
    # Combine train and validation for cross-validation
    for csv_path, img_dir in [(train_csv, train_img_dir), (valid_csv, val_img_dir)]:
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            for _, row in df.iterrows():
                img_name = str(row.iloc[0])
                label = int(row.iloc[1])
                img_path = os.path.join(img_dir, f"{img_name}.png")
                if os.path.exists(img_path):
                    all_paths.append(img_path)
                    all_labels.append(label)
    
    logger.info(f"APTOS: loaded {len(all_paths)} images, "
                f"class distribution: {dict(Counter(all_labels))}")
    
    return all_paths, all_labels


def load_skin_data(config):
    """Load ISIC Skin dataset file paths and labels."""
    base = config['data']['base_dir']
    ds_cfg = config['data']['datasets']['skin']
    
    train_dir = os.path.join(base, ds_cfg['train_dir'])
    val_dir = os.path.join(base, ds_cfg['val_dir'])
    
    all_paths = []
    all_labels = []
    class_to_idx = {"benign": 0, "malignant": 1}
    
    for data_dir in [train_dir, val_dir]:
        if os.path.exists(data_dir):
            for class_name in sorted(os.listdir(data_dir)):
                class_dir = os.path.join(data_dir, class_name)
                if not os.path.isdir(class_dir) or class_name.startswith('.'):
                    continue
                label = class_to_idx.get(class_name.lower(), -1)
                if label == -1:
                    continue
                for fname in sorted(os.listdir(class_dir)):
                    if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                        all_paths.append(os.path.join(class_dir, fname))
                        all_labels.append(label)
    
    logger.info(f"Skin: loaded {len(all_paths)} images, "
                f"class distribution: {dict(Counter(all_labels))}")
    
    return all_paths, all_labels


def load_hyperkvasir_data(config, subtask):
    """Load HyperKvasir dataset for a specific binary subtask."""
    base = config['data']['base_dir']
    ds_cfg = config['data']['datasets']['hyperkvasir']
    hk_base = os.path.join(base, ds_cfg['base_dir'])
    subtask_cfg = ds_cfg['subtasks'][subtask]
    
    all_paths = []
    all_labels = []
    
    # Positive class
    pos_dir = os.path.join(hk_base, subtask_cfg['positive_dir'])
    if os.path.exists(pos_dir):
        for fname in sorted(os.listdir(pos_dir)):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                all_paths.append(os.path.join(pos_dir, fname))
                all_labels.append(1)
    
    # Negative class
    neg_dir = os.path.join(hk_base, subtask_cfg['negative_dir'])
    if os.path.exists(neg_dir):
        for fname in sorted(os.listdir(neg_dir)):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                all_paths.append(os.path.join(neg_dir, fname))
                all_labels.append(0)
    
    logger.info(f"HyperKvasir [{subtask}]: loaded {len(all_paths)} images, "
                f"class distribution: {dict(Counter(all_labels))}")
    
    return all_paths, all_labels


def load_dataset(config, dataset_name, task=None):
    """Unified dataset loading interface."""
    if dataset_name == 'aptos':
        return load_aptos_data(config)
    elif dataset_name == 'skin':
        return load_skin_data(config)
    elif dataset_name == 'hyperkvasir':
        if task is None:
            raise ValueError("HyperKvasir requires a subtask specification")
        return load_hyperkvasir_data(config, task)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


# =============================================================================
# Splitting
# =============================================================================

def generate_cv_splits(image_paths, labels, config, dataset_name, task=None):
    """
    Generate 5-fold stratified cross-validation splits.
    60% train, 20% validation, 20% test per fold.
    
    We achieve 60/20/20 by:
    1. StratifiedKFold with 5 folds gives 80/20 splits
    2. Within the 80% train portion, we use another split to get 75/25 = 60/20 of total
    """
    seed = config['seed']
    n_folds = config['cv']['n_folds']
    
    labels_arr = np.array(labels)
    indices = np.arange(len(labels_arr))
    
    # Use 5-fold: each fold is 20% test
    # Then split the remaining 80% into 75% train / 25% val = 60% / 20% of total
    outer_skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    
    splits = []
    
    for fold_idx, (trainval_idx, test_idx) in enumerate(outer_skf.split(indices, labels_arr)):
        # Now split trainval into train (75%) and val (25%)
        # 75% of 80% = 60% of total, 25% of 80% = 20% of total
        trainval_labels = labels_arr[trainval_idx]
        
        inner_skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=seed + fold_idx)
        inner_splits = list(inner_skf.split(trainval_idx, trainval_labels))
        train_inner_idx, val_inner_idx = inner_splits[0]
        
        train_idx = trainval_idx[train_inner_idx]
        val_idx = trainval_idx[val_inner_idx]
        
        fold_info = {
            'fold': fold_idx,
            'train_indices': train_idx.tolist(),
            'val_indices': val_idx.tolist(),
            'test_indices': test_idx.tolist(),
            'train_size': len(train_idx),
            'val_size': len(val_idx),
            'test_size': len(test_idx),
            'train_class_dist': dict(Counter(labels_arr[train_idx].tolist())),
            'val_class_dist': dict(Counter(labels_arr[val_idx].tolist())),
            'test_class_dist': dict(Counter(labels_arr[test_idx].tolist())),
        }
        splits.append(fold_info)
        
        logger.info(
            f"Fold {fold_idx}: train={len(train_idx)}, val={len(val_idx)}, "
            f"test={len(test_idx)}"
        )
    
    return splits


def save_splits(splits, image_paths, labels, config, dataset_name, task=None):
    """Save fold assignments to disk."""
    base = config['data']['base_dir']
    splits_dir = os.path.join(base, config['data'].get('splits_dir', 'splits'))
    os.makedirs(splits_dir, exist_ok=True)
    
    task_str = f"_{task}" if task else ""
    splits_file = os.path.join(splits_dir, f"{dataset_name}{task_str}_splits.json")
    
    save_data = {
        'dataset': dataset_name,
        'task': task,
        'seed': config['seed'],
        'n_folds': config['cv']['n_folds'],
        'total_samples': len(image_paths),
        'stratification': config['data']['datasets'].get(
            dataset_name, {}
        ).get('stratification', 'image_level'),
        'created_at': datetime.now().isoformat(),
        'splits': splits,
        'image_paths': [_portable_path(path, base) for path in image_paths],
        'labels': labels,
    }
    
    with open(splits_file, 'w') as f:
        json.dump(save_data, f, indent=2)
    
    logger.info(f"Splits saved to {splits_file}")
    return splits_file


def load_splits(config, dataset_name, task=None):
    """Load previously saved splits from disk."""
    base = config['data']['base_dir']
    splits_dir = os.path.join(base, config['data'].get('splits_dir', 'splits'))
    task_str = f"_{task}" if task else ""
    splits_file = os.path.join(splits_dir, f"{dataset_name}{task_str}_splits.json")
    
    if os.path.exists(splits_file):
        with open(splits_file, 'r') as f:
            data = json.load(f)
        data['image_paths'] = [
            path if os.path.isabs(path) else os.path.join(base, path)
            for path in data.get('image_paths', [])
        ]
        logger.info(f"Loaded existing splits from {splits_file}")
        return data
    return None


def get_or_create_splits(config, dataset_name, task=None):
    """Get existing splits or create new ones."""
    existing = load_splits(config, dataset_name, task)
    if existing is not None:
        return existing
    
    image_paths, labels = load_dataset(config, dataset_name, task)
    splits = generate_cv_splits(image_paths, labels, config, dataset_name, task)
    save_splits(splits, image_paths, labels, config, dataset_name, task)
    
    return {
        'splits': splits,
        'image_paths': image_paths,
        'labels': labels,
        'dataset': dataset_name,
        'task': task,
    }


# =============================================================================
# Transforms
# =============================================================================

def get_transforms(img_size=299, is_train=True, perturbation=None):
    """Build transforms, applying optional photometric noise before normalization."""
    if is_train:
        transform_steps = [
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
        ]
        if perturbation is not None:
            transform_steps.append(perturbation)
        transform_steps.append(
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225],
            )
        )
        return transforms.Compose(transform_steps)
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])


# =============================================================================
# Dataset Manifest
# =============================================================================

def save_dataset_manifest(config, dataset_name, task, image_paths, labels):
    """Save a dataset manifest (file list + hash) for reproducibility."""
    base = config['data']['base_dir']
    manifests_dir = os.path.join(
        base, config['data'].get('manifests_dir', 'outputs/manifests')
    )
    os.makedirs(manifests_dir, exist_ok=True)
    
    task_str = f"_{task}" if task else ""
    manifest_file = os.path.join(manifests_dir, f"{dataset_name}{task_str}_manifest.json")
    
    # Create hash of all paths for integrity check
    paths_hash = hashlib.md5('\n'.join(image_paths).encode()).hexdigest()
    
    manifest = {
        'dataset': dataset_name,
        'task': task,
        'n_samples': len(image_paths),
        'class_distribution': dict(Counter(labels)),
        'paths_hash': paths_hash,
        'created_at': datetime.now().isoformat(),
        'sample_paths': [
            _portable_path(path, base) for path in image_paths[:5]
        ],
    }
    
    with open(manifest_file, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    logger.info(f"Manifest saved to {manifest_file}")
    return manifest_file
