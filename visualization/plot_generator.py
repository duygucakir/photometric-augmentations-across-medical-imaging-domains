"""
Phase 3 - Visualization & Plot Generation Module
=================================================
Generates all publication-ready figures and LaTeX tables.
"""

import os
import json
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

# Set publication-quality defaults
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})


# =============================================================================
# Training Curves
# =============================================================================

def plot_training_curves(metrics_history, save_path, title="Training Curves"):
    """Plot training and validation loss/accuracy curves."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss curves
    axes[0].plot(metrics_history['train_loss'], label='Train', linewidth=2)
    axes[0].plot(metrics_history['val_loss'], label='Validation', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title(f'{title} - Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy curves
    axes[1].plot(metrics_history['train_acc'], label='Train', linewidth=2)
    axes[1].plot(metrics_history['val_acc'], label='Validation', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title(f'{title} - Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    logger.info(f"Training curves saved: {save_path}")


# =============================================================================
# Confusion Matrices
# =============================================================================

def plot_confusion_matrix(cm, class_names, save_path, title="Confusion Matrix"):
    """Plot a publication-ready confusion matrix."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                ax=ax, linewidths=0.5)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(title)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    logger.info(f"Confusion matrix saved: {save_path}")


# =============================================================================
# Channel Histograms
# =============================================================================

def plot_channel_histograms(histograms, save_path, title="RGB Channel Distribution"):
    """Plot RGB channel histogram distributions."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    colors = {'R': '#e74c3c', 'G': '#27ae60', 'B': '#2980b9'}
    
    for idx, (ch_name, ax) in enumerate(zip(['R', 'G', 'B'], axes)):
        hist_data = histograms[ch_name]['histogram']
        bin_edges = histograms[ch_name]['bin_edges']
        bin_centers = (np.array(bin_edges[:-1]) + np.array(bin_edges[1:])) / 2
        
        ax.fill_between(bin_centers, hist_data, alpha=0.3, color=colors[ch_name])
        ax.plot(bin_centers, hist_data, color=colors[ch_name], linewidth=2)
        ax.set_xlabel('Pixel Intensity')
        ax.set_ylabel('Density')
        ax.set_title(f'{ch_name} Channel')
        
        mean = histograms[ch_name]['mean']
        std = histograms[ch_name]['std']
        ax.text(0.95, 0.95, f'μ={mean:.1f}\nσ={std:.1f}',
                transform=ax.transAxes, ha='right', va='top',
                fontsize=9, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        ax.grid(True, alpha=0.3)
    
    fig.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    logger.info(f"Channel histograms saved: {save_path}")


def plot_channel_overlay(histograms, save_path, title="RGB Channel Overlay"):
    """Plot all three channels overlaid for comparison."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = {'R': '#e74c3c', 'G': '#27ae60', 'B': '#2980b9'}
    
    for ch_name in ['R', 'G', 'B']:
        hist_data = histograms[ch_name]['histogram']
        bin_edges = histograms[ch_name]['bin_edges']
        bin_centers = (np.array(bin_edges[:-1]) + np.array(bin_edges[1:])) / 2
        
        ax.fill_between(bin_centers, hist_data, alpha=0.15, color=colors[ch_name])
        ax.plot(bin_centers, hist_data, color=colors[ch_name], linewidth=2, label=ch_name)
    
    ax.set_xlabel('Pixel Intensity')
    ax.set_ylabel('Density')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    logger.info(f"Channel overlay saved: {save_path}")


# =============================================================================
# Sigma Sensitivity Plots
# =============================================================================

def plot_sigma_sensitivity(summary_data, save_path, metric='macro_f1',
                           title="Sigma Sensitivity Analysis"):
    """
    Plot performance vs sigma level.
    
    Args:
        summary_data: dict of {sigma_label: {metric: {mean, std}}}
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    
    sigma_labels = list(summary_data.keys())
    means = [summary_data[s][metric]['mean'] for s in sigma_labels]
    stds = [summary_data[s][metric]['std'] for s in sigma_labels]
    
    x = range(len(sigma_labels))
    ax.bar(x, means, yerr=stds, capsize=5, color='#3498db', alpha=0.7, edgecolor='black')
    ax.set_xticks(x)
    ax.set_xticklabels(sigma_labels)
    ax.set_xlabel('Sigma Level')
    ax.set_ylabel(metric.replace('_', ' ').title())
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    logger.info(f"Sigma sensitivity plot saved: {save_path}")


# =============================================================================
# t-SNE / UMAP Plots
# =============================================================================

def plot_embedding(coords, labels, class_names, save_path, method='t-SNE',
                   title=None):
    """Plot t-SNE or UMAP embedding visualization."""
    if coords is None:
        return
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    unique_labels = np.unique(labels)
    colors = plt.cm.Set2(np.linspace(0, 1, len(unique_labels)))
    
    for i, label in enumerate(unique_labels):
        mask = labels == label
        class_name = class_names[label] if label < len(class_names) else f"Class {label}"
        ax.scatter(coords[mask, 0], coords[mask, 1], 
                  c=[colors[i]], label=class_name, alpha=0.6, s=20)
    
    ax.set_xlabel(f'{method} Dimension 1')
    ax.set_ylabel(f'{method} Dimension 2')
    ax.set_title(title or f'{method} Visualization')
    ax.legend(markerscale=2)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    logger.info(f"{method} plot saved: {save_path}")


# =============================================================================
# Variance / Fold Stability Plots
# =============================================================================

def plot_fold_variability(fold_results, save_path, metric='macro_f1',
                          title="Fold Variability"):
    """Plot per-fold scores for each condition to show variability."""
    fig, ax = plt.subplots(figsize=(14, 6))
    
    conditions = list(fold_results.keys())
    n_conditions = len(conditions)
    x = np.arange(n_conditions)
    
    for fold_idx in range(5):
        fold_values = []
        for cond in conditions:
            values = fold_results[cond].get(metric, [])
            if fold_idx < len(values):
                fold_values.append(values[fold_idx])
            else:
                fold_values.append(float('nan'))
        ax.scatter(x, fold_values, alpha=0.6, s=40, label=f'Fold {fold_idx+1}')
    
    # Plot means
    means = [np.mean(fold_results[c].get(metric, [0])) for c in conditions]
    ax.plot(x, means, 'k-o', linewidth=2, markersize=8, label='Mean', zorder=5)
    
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, rotation=45, ha='right')
    ax.set_ylabel(metric.replace('_', ' ').title())
    ax.set_title(title)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    logger.info(f"Fold variability plot saved: {save_path}")


# =============================================================================
# Bar Plots with Error Bars
# =============================================================================

def plot_comparison_bars(summaries, save_path, metric='macro_f1',
                         title="Condition Comparison"):
    """Bar plot comparing all conditions for a given metric."""
    fig, ax = plt.subplots(figsize=(14, 6))
    
    conditions = list(summaries.keys())
    means = []
    stds = []
    
    for cond in conditions:
        summary = summaries[cond].get('summary', summaries[cond])
        if metric in summary:
            means.append(summary[metric]['mean'])
            stds.append(summary[metric]['std'])
        else:
            means.append(0)
            stds.append(0)
    
    x = np.arange(len(conditions))
    colors = plt.cm.Set3(np.linspace(0, 1, len(conditions)))
    
    bars = ax.bar(x, means, yerr=stds, capsize=4, color=colors, 
                  edgecolor='black', linewidth=0.5)
    
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, rotation=45, ha='right')
    ax.set_ylabel(metric.replace('_', ' ').title())
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, mean, std in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + std + 0.005,
                f'{mean:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    logger.info(f"Comparison bar plot saved: {save_path}")


# =============================================================================
# Qualitative Image Grid
# =============================================================================

def plot_qualitative_grid(images_dict, save_path, title="Qualitative Examples"):
    """
    Plot a grid of image examples showing different conditions.
    
    Args:
        images_dict: dict of {condition_name: list_of_PIL_images}
    """
    conditions = list(images_dict.keys())
    n_conditions = len(conditions)
    n_images = len(next(iter(images_dict.values())))
    
    fig, axes = plt.subplots(n_conditions, n_images, 
                             figsize=(3 * n_images, 3 * n_conditions))
    
    if n_conditions == 1:
        axes = axes.reshape(1, -1)
    if n_images == 1:
        axes = axes.reshape(-1, 1)
    
    for i, condition in enumerate(conditions):
        for j, img in enumerate(images_dict[condition][:n_images]):
            axes[i, j].imshow(img)
            axes[i, j].axis('off')
            if j == 0:
                axes[i, j].set_ylabel(condition, fontsize=10, rotation=0, 
                                       labelpad=80, va='center')
    
    fig.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    logger.info(f"Qualitative grid saved: {save_path}")


# =============================================================================
# LaTeX Table Generation
# =============================================================================

def generate_results_latex_table(summaries, dataset_name, task=None,
                                  metrics=None):
    """Generate a LaTeX-ready results table."""
    if metrics is None:
        metrics = ['accuracy', 'macro_f1', 'auc', 'balanced_accuracy']
    
    task_str = f" ({task})" if task else ""
    
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Results for " + dataset_name + task_str + r"}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{l" + "c" * len(metrics) + r"}",
        r"\hline",
        "Condition & " + " & ".join([m.replace('_', ' ').title() for m in metrics]) + r" \\",
        r"\hline",
    ]
    
    for condition, data in summaries.items():
        summary = data.get('summary', data)
        cells = [condition.replace('_', r'\_')]
        
        for metric in metrics:
            if metric in summary:
                mean = summary[metric]['mean']
                std = summary[metric]['std']
                cells.append(f"${mean:.4f} \\pm {std:.4f}$")
            else:
                cells.append("--")
        
        lines.append(" & ".join(cells) + r" \\")
    
    lines.extend([
        r"\hline",
        r"\end{tabular}}",
        r"\end{table}",
    ])
    
    return '\n'.join(lines)


def save_latex_table(latex_str, save_path):
    """Save LaTeX table to file."""
    with open(save_path, 'w') as f:
        f.write(latex_str)
    logger.info(f"LaTeX table saved: {save_path}")


# =============================================================================
# Master Plot Generation
# =============================================================================

def generate_all_plots(config, dataset_name, task, fold_results, summaries,
                       channel_analysis=None, class_names=None):
    """
    Generate all publication-ready plots for a dataset/task.
    """
    base = config['data']['base_dir']
    figures_dir = os.path.join(base, config['outputs']['figures_dir'])
    latex_dir = os.path.join(base, config['outputs']['latex_dir'])
    task_str = f"_{task}" if task else ""
    prefix = f"{dataset_name}{task_str}"
    
    # Create subdirectories
    for subdir in ['training_curves', 'confusion_matrices', 'histograms',
                   'sigma_sensitivity', 'tsne_umap', 'variance', 'qualitative']:
        os.makedirs(os.path.join(figures_dir, subdir), exist_ok=True)
    os.makedirs(latex_dir, exist_ok=True)
    
    # 1. Comparison bar plots for each primary metric
    for metric in ['accuracy', 'macro_f1', 'auc', 'balanced_accuracy']:
        plot_comparison_bars(
            summaries,
            os.path.join(figures_dir, f"{prefix}_{metric}_comparison.png"),
            metric=metric,
            title=f"{dataset_name}{' - ' + task if task else ''}: "
                  f"{metric.replace('_', ' ').title()}"
        )
    
    # 2. Fold variability plots
    for metric in ['macro_f1', 'auc']:
        plot_fold_variability(
            fold_results,
            os.path.join(figures_dir, 'variance', f"{prefix}_{metric}_fold_variability.png"),
            metric=metric,
            title=f"{dataset_name}: Fold Variability ({metric.replace('_', ' ').title()})"
        )
    
    # 3. Channel histograms
    if channel_analysis:
        plot_channel_histograms(
            channel_analysis['histograms'],
            os.path.join(figures_dir, 'histograms', f"{prefix}_channel_histograms.png"),
            title=f"{dataset_name}: RGB Distributions"
        )
        plot_channel_overlay(
            channel_analysis['histograms'],
            os.path.join(figures_dir, 'histograms', f"{prefix}_channel_overlay.png"),
            title=f"{dataset_name}: RGB Overlay"
        )
    
    # 4. LaTeX results table
    latex_str = generate_results_latex_table(summaries, dataset_name, task)
    save_latex_table(
        latex_str,
        os.path.join(latex_dir, f"{prefix}_results_table.tex")
    )
    
    logger.info(f"All plots generated for {prefix}")
