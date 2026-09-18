"""
Photometric Augmentation Experiment Pipeline
============================================
Config-driven, resume-safe experimental pipeline.
Runs all conditions across all datasets with full logging,
checkpointing, evaluation, statistical analysis, and visualization.

Usage:
    python run_pipeline.py                          # Run all
    python run_pipeline.py --dataset aptos          # Run one dataset
    python run_pipeline.py --condition baseline_none # Run one condition
"""

import os
import sys
import json
import time
import random
import logging
import argparse
import warnings
from datetime import datetime
from pathlib import Path
from collections import defaultdict

import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
warnings.filterwarnings('ignore')

# Add phase3 to path
PHASE3_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PHASE3_DIR)

from data.data_loader import (
    load_dataset, get_or_create_splits, BaseImageDataset, 
    get_transforms, save_dataset_manifest
)
from preprocessing.preprocessor import create_preprocessing_fn, log_preprocessing_config
from augmentation.perturbation import (
    PerturbedTrainDataset, resolve_sigma, log_perturbation_config
)
from models.xception_model import build_model, get_num_classes
from training.trainer import (
    train_fold, evaluate, create_weighted_sampler,
    get_checkpoint_path, load_experiment_status, save_experiment_status,
    is_fold_completed, mark_fold_status, load_checkpoint
)
from evaluation.evaluator import (
    compute_metrics, compute_fold_summary, save_fold_results,
    save_condition_summary, save_predictions, generate_comparison_table,
    load_all_condition_summaries, update_markdown_status
)
from stats_analysis.statistical_tests import (
    run_pairwise_analysis, save_statistical_analysis, generate_latex_stats_table
)
from analysis.channel_analysis import (
    run_channel_analysis, extract_embeddings, compute_tsne, compute_umap,
    save_embeddings
)
from visualization.plot_generator import (
    plot_training_curves, plot_confusion_matrix, plot_embedding,
    generate_all_plots, plot_qualitative_grid, plot_sigma_sensitivity,
    generate_results_latex_table, save_latex_table
)


# =============================================================================
# Setup
# =============================================================================

def get_device():
    """Get the best available device."""
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_seed(seed, deterministic=False):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def load_config(config_path=None):
    """Load base configuration and resolve its project root."""
    if config_path is None:
        config_path = os.path.join(PHASE3_DIR, 'configs', 'base_config.yaml')
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    base_dir = os.path.expandvars(os.path.expanduser(
        config.get('data', {}).get('base_dir', '.')
    ))
    if not os.path.isabs(base_dir):
        base_dir = os.path.join(PHASE3_DIR, base_dir)
    config['data']['base_dir'] = os.path.abspath(base_dir)
    return config


def load_experiment_registry(registry_path=None):
    """Load experiment registry."""
    if registry_path is None:
        registry_path = os.path.join(PHASE3_DIR, 'configs', 'experiment_registry.yaml')
    
    with open(registry_path, 'r') as f:
        registry = yaml.safe_load(f)
    return registry


def setup_logging(config, dataset_name=None, task=None):
    """Setup structured logging."""
    base = config['data']['base_dir']
    log_dir = os.path.join(base, config['outputs']['logs_dir'])
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ds_str = f"_{dataset_name}" if dataset_name else ""
    task_str = f"_{task}" if task else ""
    
    log_file = os.path.join(log_dir, f"pipeline{ds_str}{task_str}_{timestamp}.log")
    
    # Reset handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ],
        force=True
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging to: {log_file}")
    logger.info(f"Timestamp: {timestamp}")
    logger.info(f"Device: {get_device()}")
    logger.info(f"PyTorch version: {torch.__version__}")
    
    return logger


def merge_configs(base_config, condition_config):
    """Merge condition-specific overrides into base config."""
    import copy
    merged = copy.deepcopy(base_config)
    
    for key, value in condition_config.items():
        if key == 'description':
            continue
        if isinstance(value, dict) and key in merged:
            merged[key].update(value)
        else:
            merged[key] = value
    
    return merged


# =============================================================================
# Qualitative Examples
# =============================================================================

def generate_qualitative_examples(config, image_paths, n_examples=5, seed=42):
    """Generate qualitative examples showing different perturbation effects."""
    from preprocessing.preprocessor import apply_preprocessing
    
    rng = np.random.RandomState(seed)
    sample_indices = rng.choice(len(image_paths), min(n_examples, len(image_paths)), 
                                replace=False)
    
    results = {}
    sigma_med = config.get('perturbation', {}).get('sigma_levels', {}).get('medium', 51.0)
    
    for idx in sample_indices:
        img_path = image_paths[idx]
        img = Image.open(img_path).convert('RGB')
        img_np = np.array(img)
        
        examples = {'original': img_np.copy()}
        
        # Preprocessed
        pre_config = {**config, 'preprocessing': {**config.get('preprocessing', {}), 'enabled': True}}
        pre_img = apply_preprocessing(img, pre_config)
        examples['preprocessed'] = np.array(pre_img)
        
        # Channel perturbations (applied to preprocessed)
        for ch_name, ch_idx in [('R_jitter', 0), ('G_jitter', 1), ('B_jitter', 2)]:
            perturbed = examples['preprocessed'].copy().astype(np.float64)
            noise = rng.normal(0, sigma_med, perturbed[:, :, ch_idx].shape)
            perturbed[:, :, ch_idx] += noise
            perturbed = np.clip(perturbed, 0, 255).astype(np.uint8)
            examples[ch_name] = perturbed
        
        # RGB perturbation
        perturbed = examples['preprocessed'].copy().astype(np.float64)
        for ch_idx in range(3):
            noise = rng.normal(0, sigma_med, perturbed[:, :, ch_idx].shape)
            perturbed[:, :, ch_idx] += noise
        perturbed = np.clip(perturbed, 0, 255).astype(np.uint8)
        examples['RGB_jitter'] = perturbed
        
        results[os.path.basename(img_path)] = examples
    
    return results


# =============================================================================
# Single Experiment Runner
# =============================================================================

def run_single_experiment(config, condition_name, condition_config,
                         dataset_name, task, splits_data, device, logger):
    """
    Run a single experiment: one condition on one dataset/task.
    Handles all 5 folds with resume capability.
    
    Returns:
        fold_metrics: list of metric dicts per fold
        fold_histories: list of training histories per fold
    """
    experiment_config = merge_configs(config, condition_config)
    num_classes = get_num_classes(config, dataset_name, task)
    
    image_paths = splits_data['image_paths']
    labels = splits_data['labels']
    splits = splits_data['splits']
    
    # Log configuration
    task_str = f"/{task}" if task else ""
    logger.info(f"\n{'='*80}")
    logger.info(f"EXPERIMENT: {dataset_name}{task_str} | {condition_name}")
    logger.info(f"Description: {condition_config.get('description', 'N/A')}")
    logger.info(f"{'='*80}")
    
    log_preprocessing_config(experiment_config, dataset_name, task)
    sigma = resolve_sigma(config, condition_config)
    log_perturbation_config(condition_config, sigma)
    
    # Load experiment status for resume
    status_dict = load_experiment_status(config, dataset_name, task)
    
    # Preprocessing function
    preprocess_fn = create_preprocessing_fn(experiment_config)
    
    # Image size for Xception
    img_size = config.get('model', {}).get('input_size', 299)
    
    fold_metrics = []
    fold_histories = []
    
    for fold_idx, fold_split in enumerate(splits):
        fold_key = f"{condition_name}_fold{fold_idx}"
        
        # Check if already completed
        if is_fold_completed(status_dict, condition_name, fold_idx):
            logger.info(f"Fold {fold_idx} already completed, loading results...")
            # Load saved results
            base = config['data']['base_dir']
            metrics_dir = os.path.join(base, config['outputs']['metrics_dir'])
            task_str_file = f"_{task}" if task else ""
            result_file = os.path.join(
                metrics_dir,
                f"{dataset_name}{task_str_file}_{condition_name}_fold{fold_idx}.json"
            )
            if os.path.exists(result_file):
                with open(result_file, 'r') as f:
                    saved = json.load(f)
                fold_metrics.append(saved['metrics'])
                fold_histories.append({})  # History not re-loaded
                continue
        
        logger.info(f"\n--- Fold {fold_idx+1}/{len(splits)} ---")
        mark_fold_status(status_dict, condition_name, fold_idx, 'running')
        save_experiment_status(config, dataset_name, task, status_dict)
        # Update markdown status
        update_markdown_status(config, dataset_name, task, status_dict)
        
        try:
            # Set seed for this fold
            set_seed(config['seed'] + fold_idx)
            
            # Create datasets
            train_indices = fold_split['train_indices']
            val_indices = fold_split['val_indices']
            test_indices = fold_split['test_indices']
            
            train_paths = [image_paths[i] for i in train_indices]
            train_labels = [labels[i] for i in train_indices]
            val_paths = [image_paths[i] for i in val_indices]
            val_labels = [labels[i] for i in val_indices]
            test_paths = [image_paths[i] for i in test_indices]
            test_labels = [labels[i] for i in test_indices]
            
            # Create base datasets
            train_ds = BaseImageDataset(
                train_paths, train_labels,
                transform=get_transforms(img_size, is_train=True),
                preprocessing_fn=preprocess_fn
            )
            val_ds = BaseImageDataset(
                val_paths, val_labels,
                transform=get_transforms(img_size, is_train=False),
                preprocessing_fn=preprocess_fn
            )
            test_ds = BaseImageDataset(
                test_paths, test_labels,
                transform=get_transforms(img_size, is_train=False),
                preprocessing_fn=preprocess_fn
            )
            
            # Apply perturbation to training data only
            pert_cfg = condition_config.get('perturbation', {})
            if pert_cfg.get('enabled', False) and sigma is not None:
                channels = pert_cfg.get('channels', [])
                train_ds = PerturbedTrainDataset(train_ds, channels, sigma)
                logger.info(f"  Perturbation applied: channels={channels}, sigma={sigma}")
            else:
                logger.info("  No perturbation applied")
            
            # Create data loaders
            batch_size = config.get('training', {}).get('batch_size', 32)
            num_workers = config.get('training', {}).get('num_workers', 4)  # Safe with spawn multiprocessing
            
            # The archived experiments used inverse-frequency weighted sampling.
            use_weighted_sampler = config.get('training', {}).get(
                'use_weighted_sampler', True
            )
            train_sampler = (
                create_weighted_sampler(train_labels)
                if use_weighted_sampler else None
            )
            
            train_loader = DataLoader(
                train_ds, batch_size=batch_size, sampler=train_sampler,
                shuffle=train_sampler is None,
                num_workers=num_workers, pin_memory=True
            )
            val_loader = DataLoader(
                val_ds, batch_size=batch_size, shuffle=False,
                num_workers=num_workers, pin_memory=True
            )
            test_loader = DataLoader(
                test_ds, batch_size=batch_size, shuffle=False,
                num_workers=num_workers, pin_memory=True
            )
            
            # Build model
            model = build_model(config, num_classes=num_classes).to(device)
            
            # Checkpoint path
            checkpoint_path = get_checkpoint_path(
                config, dataset_name, task, condition_name, fold_idx
            )
            
            # Train
            train_start = time.time()
            metrics_history, best_epoch = train_fold(
                model, train_loader, val_loader, config, condition_config,
                checkpoint_path, device, fold_idx
            )
            train_duration = time.time() - train_start
            
            # Save training curves
            base = config['data']['base_dir']
            figures_dir = os.path.join(base, config['outputs']['figures_dir'])
            tc_dir = os.path.join(figures_dir, 'training_curves')
            os.makedirs(tc_dir, exist_ok=True)
            task_str_file = f"_{task}" if task else ""
            plot_training_curves(
                metrics_history,
                os.path.join(tc_dir, 
                            f"{dataset_name}{task_str_file}_{condition_name}_fold{fold_idx}.png"),
                title=f"{condition_name} Fold {fold_idx}"
            )
            
            # Evaluate on test set
            criterion = nn.CrossEntropyLoss()
            _, _, y_true, y_pred, y_probs = evaluate(model, test_loader, criterion, device)
            
            # Get class names
            if dataset_name == 'aptos':
                class_names = config['data']['datasets']['aptos']['class_names']
            elif dataset_name == 'skin':
                class_names = config['data']['datasets']['skin']['class_names']
            else:
                class_names = ['negative', 'positive']
            
            # Compute all metrics
            fold_result = compute_metrics(y_true, y_pred, y_probs, num_classes, class_names)
            fold_result['best_epoch'] = best_epoch
            fold_result['train_duration_seconds'] = train_duration
            fold_result['early_stopping_epoch'] = len(metrics_history['train_loss'])
            
            fold_metrics.append(fold_result)
            fold_histories.append(metrics_history)
            
            # Save fold results
            save_fold_results(fold_result, config, dataset_name, task, 
                            condition_name, fold_idx)
            
            # Save predictions
            save_predictions(y_true, y_pred, y_probs, config, dataset_name, task,
                           condition_name, fold_idx)
            
            # Save confusion matrix plot
            cm_dir = os.path.join(figures_dir, 'confusion_matrices')
            os.makedirs(cm_dir, exist_ok=True)
            plot_confusion_matrix(
                np.array(fold_result['confusion_matrix']),
                class_names,
                os.path.join(cm_dir,
                            f"{dataset_name}{task_str_file}_{condition_name}_fold{fold_idx}_cm.png"),
                title=f"{condition_name} Fold {fold_idx}"
            )
            
            # Mark fold as completed
            mark_fold_status(status_dict, condition_name, fold_idx, 'completed', {
                'accuracy': fold_result['accuracy'],
                'macro_f1': fold_result['macro_f1'],
                'auc': fold_result.get('auc', 'N/A'),
                'train_duration': train_duration,
            })
            save_experiment_status(config, dataset_name, task, status_dict)
            # Update markdown status
            update_markdown_status(config, dataset_name, task, status_dict)
            
            logger.info(f"Fold {fold_idx}: Acc={fold_result['accuracy']:.4f}, "
                       f"F1={fold_result['macro_f1']:.4f}, "
                       f"AUC={fold_result.get('auc', 'N/A')}")
            
        except Exception as e:
            logger.error(f"Fold {fold_idx} FAILED: {e}", exc_info=True)
            mark_fold_status(status_dict, condition_name, fold_idx, 'failed', 
                           {'error': str(e)})
            save_experiment_status(config, dataset_name, task, status_dict)
            raise
    
    # Compute and save condition summary
    if fold_metrics:
        summary = compute_fold_summary(fold_metrics)
        save_condition_summary(summary, config, dataset_name, task, condition_name)
        
        logger.info(f"\n{'='*60}")
        logger.info(f"SUMMARY: {condition_name} on {dataset_name}{task_str}")
        for metric in ['accuracy', 'macro_f1', 'auc', 'balanced_accuracy', 'mcc']:
            if metric in summary:
                logger.info(f"  {metric}: {summary[metric]['mean']:.4f} "
                           f"± {summary[metric]['std']:.4f}")
        logger.info(f"{'='*60}\n")
    
    return fold_metrics, fold_histories


# =============================================================================
# Main Pipeline
# =============================================================================

def run_pipeline(args):
    """Run the complete experimental pipeline."""
    
    # Load configs
    config = load_config(args.config)
    registry = load_experiment_registry(args.registry)
    
    # Setup
    device = get_device()
    set_seed(config['seed'], config.get('deterministic', False))
    logger = setup_logging(config)
    
    logger.info("="*80)
    logger.info("PHOTOMETRIC AUGMENTATION EXPERIMENTAL PIPELINE")
    logger.info(f"Started at: {datetime.now().isoformat()}")
    logger.info(f"Device: {device}")
    logger.info(f"Seed: {config['seed']}")
    logger.info("="*80)
    
    # Save reproducibility info
    base = config['data']['base_dir']
    repro_file = os.path.join(
        base, config['outputs']['base_dir'], 'reproducibility_report.json'
    )
    os.makedirs(os.path.dirname(repro_file), exist_ok=True)
    repro_info = {
        'torch_version': torch.__version__,
        'numpy_version': np.__version__,
        'python_version': sys.version,
        'device': str(device),
        'seed': config['seed'],
        'started_at': datetime.now().isoformat(),
        'config': config,
    }
    with open(repro_file, 'w') as f:
        json.dump(repro_info, f, indent=2, default=str)
    
    # Determine what to run
    dataset_tasks = registry.get('dataset_tasks', [])
    all_conditions = registry.get('conditions', {})
    
    # Filter by args
    if args.dataset:
        dataset_tasks = [dt for dt in dataset_tasks if dt['dataset'] == args.dataset]
    if args.task:
        dataset_tasks = [dt for dt in dataset_tasks if dt.get('task') == args.task]
    
    if args.condition:
        if args.condition not in all_conditions:
            available = ', '.join(sorted(all_conditions))
            raise ValueError(
                f"Unknown condition '{args.condition}'. Available: {available}"
            )
        conditions_to_run = {args.condition: all_conditions[args.condition]}
    else:
        conditions_to_run = all_conditions
    
    # === MAIN LOOP: Dataset-Task × Condition ===
    all_results = {}
    
    for dt in dataset_tasks:
        dataset_name = dt['dataset']
        task = dt.get('task')
        
        logger.info(f"\n{'#'*80}")
        logger.info(f"DATASET: {dataset_name}, TASK: {task or 'default'}")
        logger.info(f"{'#'*80}\n")
        
        # Load/create splits (shared across all conditions)
        splits_data = get_or_create_splits(config, dataset_name, task)
        
        # Save dataset manifest
        save_dataset_manifest(config, dataset_name, task,
                            splits_data['image_paths'], 
                            splits_data['labels'])
        
        # Run channel analysis once per dataset
        if not args.skip_analysis:
            try:
                channel_analysis = run_channel_analysis(
                    splits_data['image_paths'], config, dataset_name, task
                )
            except Exception as e:
                logger.warning(f"Channel analysis failed: {e}")
                channel_analysis = None
        else:
            channel_analysis = None
        
        # Collect fold-level results for statistical analysis
        task_fold_results = defaultdict(lambda: defaultdict(list))
        task_summaries = {}
        
        # Run each condition
        for condition_name, condition_config in conditions_to_run.items():
            # Skip nopre experiments if not explicitly requested (excluding baseline_none which is a baseline)
            if not args.condition and condition_name.startswith('nopre_'):
                # logger.info(f"Skipping {condition_name} as requested (nopre experiment)")
                continue
            try:
                fold_metrics, fold_histories = run_single_experiment(
                    config, condition_name, condition_config,
                    dataset_name, task, splits_data, device, logger
                )
                
                # Collect fold-level values for stats
                for fold_idx, fm in enumerate(fold_metrics):
                    for metric_key, metric_val in fm.items():
                        if isinstance(metric_val, (int, float)):
                            task_fold_results[condition_name][metric_key].append(metric_val)
                
                # Load summary for comparison
                task_summaries[condition_name] = compute_fold_summary(fold_metrics)
                
            except Exception as e:
                logger.error(f"Condition {condition_name} FAILED: {e}", exc_info=True)
                continue
        
        # === POST-TRAINING ANALYSIS ===
        
        if len(task_summaries) >= 2 and not args.skip_analysis:
            task_str = f"_{task}" if task else ""
            
            # 1. Statistical analysis vs baseline
            try:
                stats_vs_baseline = run_pairwise_analysis(
                    dict(task_fold_results), 
                    baseline_key='baseline_none'
                )
                save_statistical_analysis(
                    stats_vs_baseline, config, dataset_name, task,
                    label='vs_baseline'
                )
                
                # Also compare with preprocessing baseline
                stats_vs_pre = run_pairwise_analysis(
                    dict(task_fold_results),
                    baseline_key='baseline_pre'
                )
                save_statistical_analysis(
                    stats_vs_pre, config, dataset_name, task,
                    label='vs_preprocessing'
                )
                
            except Exception as e:
                logger.warning(f"Statistical analysis failed: {e}")
            
            # 2. Generate all plots
            try:
                generate_all_plots(
                    config, dataset_name, task,
                    dict(task_fold_results),
                    task_summaries,
                    channel_analysis=channel_analysis,
                    class_names=(config['data']['datasets'].get(dataset_name, {})
                                .get('class_names'))
                )
            except Exception as e:
                logger.warning(f"Plot generation failed: {e}")
            
            # 3. Sigma sensitivity analysis (per-channel)
            try:
                figures_dir = os.path.join(base, config['outputs']['figures_dir'])
                sigma_dir = os.path.join(figures_dir, 'sigma_sensitivity')
                os.makedirs(sigma_dir, exist_ok=True)

                # Define all channel groups for sigma sweeps
                sigma_channel_groups = {
                    'B':     ('pre_B_low',    'pre_B_med',    'pre_B_high'),
                    'R':     ('pre_R_low',    'pre_R_med',    'pre_R_high'),
                    'G':     ('pre_G_low',    'pre_G_med',    'pre_G_high'),
                    'RGB':   ('pre_RGB_low',  'pre_RGB_med',  'pre_RGB_high'),
                    
                    #'B_nopre': ('nopre_B_low', 'nopre_B_med', 'nopre_B_high'),
                    #'R_nopre': ('nopre_R_low', 'nopre_R_med', 'nopre_R_high'),
                    #'G_nopre': ('nopre_G_low', 'nopre_G_med', 'nopre_G_high'),
                    #'RGB_nopre': ('nopre_RGB_low', 'nopre_RGB_med', 'nopre_RGB_high'),
                    
                }

                for ch_label, (low_key, med_key, high_key) in sigma_channel_groups.items():
                    sigma_conditions = {
                        'Low (σ=12.75)':  task_summaries.get(low_key),
                        'Medium (σ=51.0)': task_summaries.get(med_key),
                        'High (σ=102.0)': task_summaries.get(high_key),
                    }
                    sigma_conditions = {k: v for k, v in sigma_conditions.items() if v}

                    if len(sigma_conditions) >= 2:
                        for metric in ['macro_f1', 'auc', 'accuracy']:
                            plot_sigma_sensitivity(
                                sigma_conditions,
                                os.path.join(sigma_dir,
                                    f"{dataset_name}{task_str}_sigma_{ch_label}_{metric}.png"),
                                metric=metric,
                                title=f"{dataset_name}: σ Sensitivity – {ch_label} ({metric})"
                            )
            except Exception as e:
                logger.warning(f"Sigma sensitivity plots failed: {e}")
        
        # 4. Qualitative examples (once per dataset-task)
        task_str_q = f"_{task}" if task else ""
        if not args.skip_analysis:
            try:
                qual_examples = generate_qualitative_examples(
                    config, splits_data['image_paths'], n_examples=5, seed=config['seed']
                )
                
                figures_dir = os.path.join(base, config['outputs']['figures_dir'])
                qual_dir = os.path.join(figures_dir, 'qualitative')
                os.makedirs(qual_dir, exist_ok=True)
                
                # Save individual examples
                for img_name, examples in qual_examples.items():
                    plot_qualitative_grid(
                        {k: [Image.fromarray(v)] for k, v in examples.items()},
                        os.path.join(qual_dir,
                                    f"{dataset_name}{task_str_q}_{Path(img_name).stem}_qualitative.png"),
                        title=f"Qualitative: {img_name}"
                    )
            except Exception as e:
                logger.warning(f"Qualitative examples failed: {e}")
        
        all_results[f"{dataset_name}_{task}"] = {
            'summaries': task_summaries,
            'fold_results': dict(task_fold_results),
        }
    
    # === MASTER SUMMARY ===
    logger.info("\n" + "="*80)
    logger.info("PIPELINE COMPLETE")
    logger.info("="*80)
    
    # Save master summary
    master_summary_path = os.path.join(
        base, config['outputs']['base_dir'], 'master_summary.json'
    )
    master_summary = {
        'completed_at': datetime.now().isoformat(),
        'datasets_processed': [dt['dataset'] for dt in dataset_tasks],
        'conditions_run': list(conditions_to_run.keys()),
        'results': {},
    }
    
    for key, data in all_results.items():
        master_summary['results'][key] = {
            'conditions': list(data['summaries'].keys()),
            'summary': {
                cond: {
                    metric: {
                        'mean': vals.get('mean', 'N/A'),
                        'std': vals.get('std', 'N/A'),
                    }
                    for metric, vals in summary_data.items()
                    if isinstance(vals, dict) and 'mean' in vals
                }
                for cond, summary_data in data['summaries'].items()
            }
        }
    
    with open(master_summary_path, 'w') as f:
        json.dump(master_summary, f, indent=2, default=str)
    
    logger.info(f"Master summary saved: {master_summary_path}")
    logger.info(f"Pipeline finished at: {datetime.now().isoformat()}")
    
    return all_results


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Photometric augmentation experimental pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_pipeline.py                              # Run everything
  python run_pipeline.py --dataset aptos              # Only APTOS
  python run_pipeline.py --dataset hyperkvasir --task polyps  # Specific subtask
  python run_pipeline.py --condition baseline_none     # One condition only
  python run_pipeline.py --skip-analysis               # Training only
        """
    )
    parser.add_argument('--config', type=str, default=None,
                        help='Path to base config YAML')
    parser.add_argument('--registry', type=str, default=None,
                        help='Path to experiment registry YAML')
    parser.add_argument('--dataset', type=str, default=None,
                        choices=['aptos', 'skin', 'hyperkvasir'],
                        help='Run only this dataset')
    parser.add_argument('--task', type=str, default=None,
                        help='Run only this task (for hyperkvasir)')
    parser.add_argument('--condition', type=str, default=None,
                        help='Run only this condition')
    parser.add_argument('--skip-analysis', action='store_true',
                        help='Skip post-training analysis (stats, plots)')
    return parser.parse_args()


if __name__ == '__main__':
    import torch.multiprocessing as mp
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
        
    args = parse_args()
    
    try:
        results = run_pipeline(args)
    except KeyboardInterrupt:
        print("\n\nPipeline interrupted. Progress has been saved.")
        print("Re-run the same command to resume from last checkpoint.")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Pipeline failed: {e}", exc_info=True)
        print(f"\nPipeline failed: {e}")
        print("Check logs for details. Re-run to resume from last checkpoint.")
        sys.exit(1)
