"""
Phase 3 - Training Module with Checkpointing and Resume
========================================================
Handles training loop, early stopping, checkpoint saving/loading,
and resume-after-interruption logic.
"""

import os
import json
import time
import logging
import numpy as np
from datetime import datetime
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler

logger = logging.getLogger(__name__)


# =============================================================================
# Training & Evaluation Functions
# =============================================================================

def train_one_epoch(model, loader, criterion, optimizer, device, scheduler=None,
                    gradient_clip=None):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, labels) in enumerate(loader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        
        if gradient_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip)
        
        optimizer.step()
        
        if scheduler is not None:
            scheduler.step()
        
        running_loss += loss.item() * inputs.size(0)
        preds = torch.argmax(outputs, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def evaluate(model, loader, criterion, device):
    """Evaluate model on a data loader."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)
            
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    
    return (epoch_loss, epoch_acc, 
            np.array(all_labels), np.array(all_preds), np.array(all_probs))


# =============================================================================
# Checkpoint Management
# =============================================================================

def get_checkpoint_path(config, dataset_name, task, condition_name, fold):
    """Get the checkpoint file path for a specific experiment."""
    base = config['data']['base_dir']
    ckpt_dir = os.path.join(base, config['outputs']['checkpoints_dir'])
    task_str = f"_{task}" if task else ""
    os.makedirs(ckpt_dir, exist_ok=True)
    return os.path.join(ckpt_dir, 
                        f"{dataset_name}{task_str}_{condition_name}_fold{fold}.pth")


def save_checkpoint(model, optimizer, scheduler, epoch, metrics_history,
                    best_val_loss, patience_counter, checkpoint_path):
    """Save a complete checkpoint for resume capability."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'best_val_loss': best_val_loss,
        'patience_counter': patience_counter,
        'metrics_history': metrics_history,
        'saved_at': datetime.now().isoformat(),
    }
    torch.save(checkpoint, checkpoint_path)
    logger.debug(f"Checkpoint saved: epoch={epoch}, path={checkpoint_path}")


def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None, device='cpu'):
    """Load a checkpoint for resuming training."""
    if not os.path.exists(checkpoint_path):
        return None
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler and checkpoint.get('scheduler_state_dict'):
        try:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        except Exception:
            logger.warning("Could not load scheduler state, starting fresh")
    
    logger.info(f"Checkpoint loaded: epoch={checkpoint['epoch']}, "
                f"path={checkpoint_path}")
    return checkpoint


# =============================================================================
# Experiment Status Management
# =============================================================================

def get_experiment_status_path(config, dataset_name, task):
    """Get the experiment status file path."""
    base = config['data']['base_dir']
    status_dir = os.path.join(base, config['outputs']['logs_dir'])
    os.makedirs(status_dir, exist_ok=True)
    task_str = f"_{task}" if task else ""
    return os.path.join(status_dir, f"status_{dataset_name}{task_str}.json")


def load_experiment_status(config, dataset_name, task):
    """Load experiment status to check what's been completed."""
    status_path = get_experiment_status_path(config, dataset_name, task)
    if os.path.exists(status_path):
        with open(status_path, 'r') as f:
            return json.load(f)
    return {}


def save_experiment_status(config, dataset_name, task, status_dict):
    """Save experiment status."""
    status_path = get_experiment_status_path(config, dataset_name, task)
    with open(status_path, 'w') as f:
        json.dump(status_dict, f, indent=2)


def is_fold_completed(status_dict, condition_name, fold):
    """Check if a specific fold is already completed."""
    key = f"{condition_name}_fold{fold}"
    entry = status_dict.get(key, {})
    return entry.get('status') == 'completed'


def mark_fold_status(status_dict, condition_name, fold, status, info=None):
    """Mark fold status (pending, running, completed, failed, resumed)."""
    key = f"{condition_name}_fold{fold}"
    status_dict[key] = {
        'status': status,
        'timestamp': datetime.now().isoformat(),
        'info': info or {},
    }


# =============================================================================
# Main Training Function
# =============================================================================

def train_fold(model, train_loader, val_loader, config, condition_config,
               checkpoint_path, device, fold_idx):
    """
    Train one fold with early stopping and checkpointing.
    
    Returns:
        metrics_history: dict with lists of epoch metrics
        best_epoch: epoch with best validation loss
    """
    train_cfg = config.get('training', {})
    epochs = train_cfg.get('epochs', 30)
    patience = train_cfg.get('patience', 7)
    
    # Optimizer
    opt_cfg = train_cfg.get('optimizer', {})
    optimizer = optim.AdamW(
        model.parameters(),
        lr=opt_cfg.get('lr', 1e-4),
        weight_decay=opt_cfg.get('weight_decay', 0.01)
    )
    
    # Scheduler
    sched_cfg = train_cfg.get('scheduler', {})
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=sched_cfg.get('max_lr', 1e-4),
        steps_per_epoch=len(train_loader),
        epochs=epochs
    )
    
    # Loss function
    label_smoothing = train_cfg.get('label_smoothing', 0.1)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    
    # Gradient clipping
    grad_clip_cfg = train_cfg.get('gradient_clipping', {})
    gradient_clip = grad_clip_cfg.get('max_norm', 1.0) if grad_clip_cfg.get('enabled', True) else None
    
    # Metrics tracking
    metrics_history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': [],
        'lr': [], 'epoch_time': [],
    }
    
    best_val_loss = float('inf')
    patience_counter = 0
    start_epoch = 0
    best_epoch = 0
    
    # Try to resume from checkpoint
    existing_ckpt = load_checkpoint(checkpoint_path, model, optimizer, scheduler, device)
    if existing_ckpt:
        start_epoch = existing_ckpt['epoch'] + 1
        best_val_loss = existing_ckpt['best_val_loss']
        patience_counter = existing_ckpt['patience_counter']
        metrics_history = existing_ckpt.get('metrics_history', metrics_history)
        logger.info(f"Resuming fold {fold_idx} from epoch {start_epoch}")
    
    # Training loop
    for epoch in range(start_epoch, epochs):
        epoch_start = time.time()
        
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            scheduler=scheduler, gradient_clip=gradient_clip
        )
        
        val_loss, val_acc, _, _, _ = evaluate(model, val_loader, criterion, device)
        
        epoch_time = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]['lr']
        
        metrics_history['train_loss'].append(train_loss)
        metrics_history['val_loss'].append(val_loss)
        metrics_history['train_acc'].append(train_acc)
        metrics_history['val_acc'].append(val_acc)
        metrics_history['lr'].append(current_lr)
        metrics_history['epoch_time'].append(epoch_time)
        
        logger.info(
            f"Fold {fold_idx} | Epoch {epoch+1}/{epochs} | "
            f"Train L:{train_loss:.4f} A:{train_acc:.4f} | "
            f"Val L:{val_loss:.4f} A:{val_acc:.4f} | "
            f"LR:{current_lr:.6f} | Time:{epoch_time:.1f}s"
        )
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_epoch = epoch
            
            # Save best model checkpoint
            save_checkpoint(
                model, optimizer, scheduler, epoch, metrics_history,
                best_val_loss, patience_counter, checkpoint_path
            )
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1} "
                           f"(best was epoch {best_epoch+1})")
                break
        
        # Save periodic checkpoint for crash recovery
        if (epoch + 1) % 5 == 0:
            periodic_path = checkpoint_path.replace('.pth', f'_periodic_e{epoch+1}.pth')
            save_checkpoint(
                model, optimizer, scheduler, epoch, metrics_history,
                best_val_loss, patience_counter, periodic_path
            )
    
    # Load best model
    load_checkpoint(checkpoint_path, model, device=device)
    
    return metrics_history, best_epoch


def create_weighted_sampler(labels):
    """Create WeightedRandomSampler for class imbalance."""
    labels_arr = np.array(labels)
    class_counts = np.bincount(labels_arr)
    class_weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)
    sample_weights = class_weights[labels_arr]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
    return sampler
