"""
Phase 3 - Channel Distribution & Embedding Analysis Module
===========================================================
RGB histogram analysis, KL divergence, Wasserstein distance,
t-SNE and UMAP embedding visualizations.
"""

import os
import json
import logging
import numpy as np
from datetime import datetime
from collections import defaultdict

from PIL import Image
from scipy import stats as scipy_stats
from scipy.spatial.distance import jensenshannon

logger = logging.getLogger(__name__)


# =============================================================================
# Channel Distribution Analysis
# =============================================================================

def compute_channel_histograms(image_paths, n_samples=200, n_bins=256, seed=42):
    """
    Compute per-channel histograms aggregated over sampled images.
    
    Args:
        image_paths: list of image file paths
        n_samples: number of images to sample
        n_bins: number of histogram bins
        seed: random seed
    
    Returns:
        histograms dict: {channel: normalized_histogram}
    """
    rng = np.random.RandomState(seed)
    
    if len(image_paths) > n_samples:
        sample_indices = rng.choice(len(image_paths), n_samples, replace=False)
        sampled_paths = [image_paths[i] for i in sample_indices]
    else:
        sampled_paths = image_paths
    
    channel_pixels = {'R': [], 'G': [], 'B': []}
    
    for path in sampled_paths:
        try:
            img = np.array(Image.open(path).convert('RGB'))
            channel_pixels['R'].append(img[:, :, 0].ravel())
            channel_pixels['G'].append(img[:, :, 1].ravel())
            channel_pixels['B'].append(img[:, :, 2].ravel())
        except Exception as e:
            logger.warning(f"Could not load image {path}: {e}")
    
    histograms = {}
    for ch_name in ['R', 'G', 'B']:
        all_pixels = np.concatenate(channel_pixels[ch_name])
        hist, bin_edges = np.histogram(all_pixels, bins=n_bins, range=(0, 255), density=True)
        histograms[ch_name] = {
            'histogram': hist.tolist(),
            'bin_edges': bin_edges.tolist(),
            'mean': float(np.mean(all_pixels)),
            'std': float(np.std(all_pixels)),
            'median': float(np.median(all_pixels)),
        }
    
    logger.info(f"Computed histograms from {len(sampled_paths)} images")
    return histograms


def compute_kl_divergence(hist1, hist2, epsilon=1e-10):
    """
    Compute KL divergence between two normalized histograms.
    Uses Jensen-Shannon divergence (symmetric) as well.
    """
    h1 = np.array(hist1) + epsilon
    h2 = np.array(hist2) + epsilon
    
    # Normalize
    h1 = h1 / h1.sum()
    h2 = h2 / h2.sum()
    
    # KL divergence (asymmetric)
    kl_12 = float(np.sum(h1 * np.log(h1 / h2)))
    kl_21 = float(np.sum(h2 * np.log(h2 / h1)))
    
    # Jensen-Shannon divergence (symmetric)
    js = float(jensenshannon(h1, h2) ** 2)
    
    return {'kl_12': kl_12, 'kl_21': kl_21, 'js_divergence': js}


def compute_wasserstein_distance(hist1, hist2):
    """Compute Wasserstein (Earth Mover's) distance between histograms."""
    h1 = np.array(hist1)
    h2 = np.array(hist2)
    
    # Normalize
    h1 = h1 / (h1.sum() + 1e-10)
    h2 = h2 / (h2.sum() + 1e-10)
    
    bins = np.arange(len(h1))
    return float(scipy_stats.wasserstein_distance(bins, bins, h1, h2))


def compute_channel_distances(histograms):
    """
    Compute pairwise KL and Wasserstein distances between R, G, B channels.
    
    Returns:
        distances dict with all pairwise comparisons
    """
    channels = ['R', 'G', 'B']
    distances = {}
    
    for i, ch1 in enumerate(channels):
        for ch2 in channels[i+1:]:
            h1 = histograms[ch1]['histogram']
            h2 = histograms[ch2]['histogram']
            
            kl_result = compute_kl_divergence(h1, h2)
            wasserstein = compute_wasserstein_distance(h1, h2)
            
            key = f"{ch1}_vs_{ch2}"
            distances[key] = {
                **kl_result,
                'wasserstein': wasserstein,
            }
    
    return distances


def run_channel_analysis(image_paths, config, dataset_name, task=None):
    """
    Run full channel distribution analysis and save results.
    """
    base = config['data']['base_dir']
    analysis_cfg = config.get('analysis', {}).get('channel_histograms', {})
    
    n_samples = analysis_cfg.get('n_samples', 200)
    n_bins = analysis_cfg.get('n_bins', 256)
    
    # Compute histograms
    histograms = compute_channel_histograms(image_paths, n_samples, n_bins, config['seed'])
    
    # Compute distances
    distances = compute_channel_distances(histograms)
    
    # Save results
    results = {
        'dataset': dataset_name,
        'task': task,
        'n_samples_used': min(n_samples, len(image_paths)),
        'histograms': histograms,
        'distances': distances,
        'computed_at': datetime.now().isoformat(),
    }
    
    task_str = f"_{task}" if task else ""
    output_dir = os.path.join(base, config['outputs']['tables_dir'])
    os.makedirs(output_dir, exist_ok=True)
    
    results_file = os.path.join(output_dir, 
                                f"{dataset_name}{task_str}_channel_analysis.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Channel analysis saved: {results_file}")
    return results


# =============================================================================
# Embedding Analysis (t-SNE, UMAP)
# =============================================================================

def extract_embeddings(model, data_loader, device, n_samples=500):
    """
    Extract penultimate-layer embeddings from the model.
    
    Returns:
        embeddings: numpy array [N, D]
        labels: numpy array [N]
        indices: list of sample indices
    """
    import torch
    
    model.eval()
    all_embeddings = []
    all_labels = []
    count = 0
    
    with torch.no_grad():
        for inputs, labels in data_loader:
            if count >= n_samples:
                break
            
            inputs = inputs.to(device)
            features = model.extract_features(inputs)
            
            all_embeddings.append(features.cpu().numpy())
            all_labels.extend(labels.numpy())
            count += len(labels)
    
    embeddings = np.concatenate(all_embeddings, axis=0)[:n_samples]
    labels = np.array(all_labels)[:n_samples]
    
    logger.info(f"Extracted {len(embeddings)} embeddings of dim {embeddings.shape[1]}")
    return embeddings, labels


def compute_tsne(embeddings, perplexity=30, seed=42):
    """Compute t-SNE projection."""
    from sklearn.manifold import TSNE
    
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=seed,
                n_iter=1000, learning_rate='auto', init='pca')
    coords = tsne.fit_transform(embeddings)
    return coords


def compute_umap(embeddings, n_neighbors=15, seed=42):
    """Compute UMAP projection."""
    try:
        import umap
        reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors, 
                           random_state=seed, min_dist=0.1)
        coords = reducer.fit_transform(embeddings)
        return coords
    except ImportError:
        logger.warning("UMAP not installed, skipping UMAP analysis")
        return None


def save_embeddings(embeddings, labels, coords_tsne, coords_umap,
                    config, dataset_name, task, condition_name):
    """Save embeddings and projections to disk."""
    base = config['data']['base_dir']
    embed_dir = os.path.join(base, config['outputs']['embeddings_dir'])
    os.makedirs(embed_dir, exist_ok=True)
    
    task_str = f"_{task}" if task else ""
    prefix = f"{dataset_name}{task_str}_{condition_name}"
    
    np.savez(
        os.path.join(embed_dir, f"{prefix}_embeddings.npz"),
        embeddings=embeddings,
        labels=labels,
        tsne=coords_tsne,
        umap=coords_umap if coords_umap is not None else np.array([]),
    )
    
    logger.info(f"Embeddings saved for {prefix}")
