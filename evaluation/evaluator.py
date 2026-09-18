"""
Phase 3 - Comprehensive Evaluation Module
==========================================
Computes all primary and additional metrics.
Saves fold-wise scores and mean±std summaries.
"""

import os
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from collections import defaultdict

from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix, classification_report,
    balanced_accuracy_score, matthews_corrcoef, average_precision_score
)

logger = logging.getLogger(__name__)


def compute_metrics(y_true, y_pred, y_probs, num_classes, class_names=None):
    """
    Compute all metrics for a single evaluation.
    
    Args:
        y_true: ground truth labels
        y_pred: predicted labels
        y_probs: prediction probabilities [N, num_classes]
        num_classes: number of classes
        class_names: optional list of class names
    
    Returns:
        dict of all metrics
    """
    metrics = {}
    
    # Primary metrics
    metrics['accuracy'] = float(accuracy_score(y_true, y_pred))
    metrics['macro_f1'] = float(f1_score(y_true, y_pred, average='macro', zero_division=0))
    metrics['weighted_f1'] = float(f1_score(y_true, y_pred, average='weighted', zero_division=0))
    
    # AUC
    try:
        if num_classes == 2:
            metrics['auc'] = float(roc_auc_score(y_true, y_probs[:, 1]))
        else:
            metrics['auc'] = float(roc_auc_score(y_true, y_probs, multi_class='ovr', 
                                                  average='macro'))
    except Exception as e:
        logger.warning(f"Could not compute AUC: {e}")
        metrics['auc'] = float('nan')
    
    # Additional metrics
    metrics['balanced_accuracy'] = float(balanced_accuracy_score(y_true, y_pred))
    metrics['mcc'] = float(matthews_corrcoef(y_true, y_pred))
    metrics['precision_macro'] = float(precision_score(y_true, y_pred, average='macro', zero_division=0))
    metrics['recall_macro'] = float(recall_score(y_true, y_pred, average='macro', zero_division=0))
    metrics['precision_weighted'] = float(precision_score(y_true, y_pred, average='weighted', zero_division=0))
    metrics['recall_weighted'] = float(recall_score(y_true, y_pred, average='weighted', zero_division=0))
    
    # PR-AUC / Average Precision for binary tasks
    if num_classes == 2:
        try:
            metrics['pr_auc'] = float(average_precision_score(y_true, y_probs[:, 1]))
        except Exception:
            metrics['pr_auc'] = float('nan')
    
    # Per-class F1
    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    for i, f1_val in enumerate(per_class_f1):
        class_label = class_names[i] if class_names else f"class_{i}"
        metrics[f'f1_{class_label}'] = float(f1_val)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    metrics['confusion_matrix'] = cm.tolist()
    
    return metrics


def compute_fold_summary(fold_metrics_list, metric_names=None):
    """
    Compute mean ± std across folds for each metric.
    
    Args:
        fold_metrics_list: list of metric dicts, one per fold
        metric_names: optional list of metrics to summarize (defaults to all numeric)
    
    Returns:
        dict with mean, std, and per-fold values for each metric
    """
    if not fold_metrics_list:
        return {}
    
    if metric_names is None:
        metric_names = [k for k in fold_metrics_list[0].keys() 
                       if isinstance(fold_metrics_list[0][k], (int, float))
                       and k != 'confusion_matrix']
    
    summary = {}
    for metric in metric_names:
        values = [m.get(metric, float('nan')) for m in fold_metrics_list]
        values = [v for v in values if not np.isnan(v)]
        
        if values:
            summary[metric] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'per_fold': values,
            }
    
    return summary


def save_fold_results(results, config, dataset_name, task, condition_name, fold):
    """Save results for a specific fold."""
    base = config['data']['base_dir']
    metrics_dir = os.path.join(base, config['outputs']['metrics_dir'])
    os.makedirs(metrics_dir, exist_ok=True)
    
    task_str = f"_{task}" if task else ""
    results_file = os.path.join(
        metrics_dir, 
        f"{dataset_name}{task_str}_{condition_name}_fold{fold}.json"
    )
    
    save_data = {
        'dataset': dataset_name,
        'task': task,
        'condition': condition_name,
        'fold': fold,
        'metrics': results,
        'saved_at': datetime.now().isoformat(),
    }
    
    with open(results_file, 'w') as f:
        json.dump(save_data, f, indent=2, default=str)
    
    logger.info(f"Fold results saved: {results_file}")
    return results_file


def save_condition_summary(summary, config, dataset_name, task, condition_name):
    """Save summary results for a condition (mean±std across folds)."""
    base = config['data']['base_dir']
    metrics_dir = os.path.join(base, config['outputs']['metrics_dir'])
    os.makedirs(metrics_dir, exist_ok=True)
    
    task_str = f"_{task}" if task else ""
    summary_file = os.path.join(
        metrics_dir,
        f"{dataset_name}{task_str}_{condition_name}_summary.json"
    )
    
    save_data = {
        'dataset': dataset_name,
        'task': task,
        'condition': condition_name,
        'summary': summary,
        'saved_at': datetime.now().isoformat(),
    }
    
    with open(summary_file, 'w') as f:
        json.dump(save_data, f, indent=2)
    
    logger.info(f"Condition summary saved: {summary_file}")
    return summary_file


def save_predictions(y_true, y_pred, y_probs, config, dataset_name, task,
                     condition_name, fold, split='test'):
    """Save raw predictions for later analysis."""
    base = config['data']['base_dir']
    metrics_dir = os.path.join(base, config['outputs']['metrics_dir'], 'predictions')
    os.makedirs(metrics_dir, exist_ok=True)
    
    task_str = f"_{task}" if task else ""
    pred_file = os.path.join(
        metrics_dir,
        f"{dataset_name}{task_str}_{condition_name}_fold{fold}_{split}_predictions.npz"
    )
    
    np.savez(pred_file, y_true=y_true, y_pred=y_pred, y_probs=y_probs)
    logger.debug(f"Predictions saved: {pred_file}")


def load_all_condition_summaries(config, dataset_name, task):
    """Load all condition summaries for a dataset/task for comparison."""
    base = config['data']['base_dir']
    metrics_dir = os.path.join(base, config['outputs']['metrics_dir'])
    task_str = f"_{task}" if task else ""
    
    summaries = {}
    if os.path.exists(metrics_dir):
        for fname in os.listdir(metrics_dir):
            if fname.startswith(f"{dataset_name}{task_str}_") and fname.endswith('_summary.json'):
                condition = fname.replace(f"{dataset_name}{task_str}_", '').replace('_summary.json', '')
                with open(os.path.join(metrics_dir, fname), 'r') as f:
                    summaries[condition] = json.load(f)
    
    return summaries


def generate_comparison_table(summaries, metric_names=None):
    """
    Generate a comparison table across conditions.
    
    Args:
        summaries: dict of condition_name -> summary dict
        metric_names: list of metrics to include
    
    Returns:
        pandas DataFrame
    """
    if metric_names is None:
        metric_names = ['accuracy', 'macro_f1', 'auc', 'balanced_accuracy', 'mcc']
    
    rows = []
    for condition, data in summaries.items():
        summary = data.get('summary', data)
        row = {'condition': condition}
        for metric in metric_names:
            if metric in summary:
                mean = summary[metric]['mean']
                std = summary[metric]['std']
                row[f'{metric}_mean'] = mean
                row[f'{metric}_std'] = std
                row[f'{metric}'] = f"{mean:.4f} ± {std:.4f}"
        rows.append(row)
    
    df = pd.DataFrame(rows)
    return df


def update_markdown_status(config, dataset_name, task, status_dict):
    """
    Update (or create) the [dataset/task]_experiments_status.md file.
    """
    # 1. Determine filename and path
    if dataset_name == 'hyperkvasir':
        filename = f"{task}_experiments_status.md"
    else:
        filename = f"{dataset_name}_experiments_status.md"
    
    # Write to the same directory as run_pipeline.py 
    # (which is likely the current working directory or the parent of this file)
    md_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", filename))
    
    # 2. Organize status information
    # Group conditions by name
    cond_status = defaultdict(list)
    for key, info in status_dict.items():
        # key format: {condition}_fold{idx}
        if '_fold' in key:
            parts = key.split('_fold')
            c_name = parts[0]
            cond_status[c_name].append(info)
        else:
            # Handle keys that might not have fold (though unlikely in current setup)
            cond_status[key].append(info)

    def get_summary_str(c_name, metrics=['accuracy', 'macro_f1', 'auc']):
        folds = cond_status.get(c_name, [])
        completed = [f for f in folds if f.get('status') == 'completed' and f.get('info')]
        
        if len(completed) == 5:
            # Compute summary
            fold_results = [f['info'] for f in completed]
            stats = compute_fold_summary(fold_results)
            parts = []
            for m in metrics:
                if m in stats:
                    parts.append(f"{stats[m]['mean']:.4f} ± {stats[m]['std']:.4f}")
                else:
                    parts.append("---")
            return "✅ Finished", "5 / 5", parts
        
        running = any(f.get('status') == 'running' for f in folds)
        count = len(completed)
        status_icon = "🏃 Running" if running else ("⏸️ Partial" if count > 0 else "❌ Not started")
        
        # Get latest accuracy if available
        acc_str = "---"
        if completed:
            last_info = completed[-1].get('info', {})
            acc = last_info.get('accuracy')
            if acc is not None:
                acc_str = f"Acc: {acc:.3f}"
        elif running:
            acc_str = "Calculating..."
            
        return status_icon, f"{count} / 5", [acc_str, "---", "---"]

    # 3. Generate content sections
    title = f"# {dataset_name.upper()} {task.capitalize() if task else ''} Experiments Status".strip()
    
    # --- Section 1: Baselines ---
    baselines_table = [
        "### 1. Baselines",
        "| Setup | Status | Folds | Accuracy | Macro F1 | AUC |",
        "|---|---|---|---|---|---|"
    ]
    for b_name in ['baseline_none', 'baseline_pre']:
        status_label, fold_str, metrics_list = get_summary_str(b_name)
        baselines_table.append(f"| {b_name} | {status_label} | {fold_str} | {' | '.join(metrics_list)} |")

    # --- Section 2: Perturbation Grid ---
    grid_header = [
        "### 2. Perturbation Grid (Status Summary)",
        "| Preprocessing | Color | low sigma | med sigma | high sigma |",
        "|---|---|---|---|---|"
    ]
    
    def get_grid_cell(pre, color, sigma):
        c_name = f"{pre}_{color}_{sigma}"
        status_label, fold_str, _ = get_summary_str(c_name)
        
        icon = "❌"
        if "✅" in status_label: icon = "✅"
        elif "🏃" in status_label: icon = "🏃"
        elif "⏸️" in status_label: icon = "⏸️"
        
        if icon == "✅":
            # For finished, just show Acc
            folds = cond_status.get(c_name, [])
            completed = [f for f in folds if f.get('status') == 'completed' and f.get('info')]
            acc = compute_fold_summary([f['info'] for f in completed])['accuracy']['mean']
            return f"✅ (Acc: {acc:.3f})"
        elif icon == "🏃" or icon == "⏸️":
            return f"{icon} ({fold_str})"
        else:
            return "❌ (0/5)"

    grid_rows = []
    for pre in ['nopre', 'pre']:
        for color in ['R', 'G', 'B', 'RGB']:
            row = [f"**{pre}**", color]
            for sigma in ['low', 'med', 'high']:
                row.append(get_grid_cell(pre, color, sigma))
            grid_rows.append(f"| {' | '.join(row)} |")

    # --- Section 3: Detailed Results ---
    detailed_table = [
        "### 3. Detailed Results for Finished Conditions",
        "| Configuration | Accuracy | Macro F1 | AUC |",
        "|---|---|---|---|---|"
    ]
    finished_found = False
    # Sort conditions for consistency
    all_cond_names = sorted(cond_status.keys())
    for c_name in all_cond_names:
        status_label, _, metrics_list = get_summary_str(c_name)
        if "✅" in status_label:
            detailed_table.append(f"| `{c_name}` | {' | '.join(metrics_list)} |")
            finished_found = True
    
    if not finished_found:
        detailed_table.append("| *(None finished yet)* | --- | --- | --- |")

    # 4. Assemble and write
    legend = [
        "",
        "*Legend:*",
        "- ✅ = Finished (all 5 folds completed)",
        "- ❌ = Not started (no folds completed)",
        "- 🏃 = Running (Currently computing folds)",
        "- ⏸️ = Paused/Partial (Some folds completed)",
        ""
    ]
    
    footer = [
        f"*Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*"
    ]
    
    full_content = [
        title, "",
        "\n".join(baselines_table), "",
        "\n".join(grid_header),
        "\n".join(grid_rows), "",
        "\n".join(legend),
        "\n".join(detailed_table), "",
        "\n".join(footer)
    ]
    
    with open(md_path, 'w') as f:
        f.write("\n".join(full_content))
    
    logger.info(f"Updated markdown status report: {md_path}")
