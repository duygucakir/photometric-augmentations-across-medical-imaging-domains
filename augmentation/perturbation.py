"""
Phase 3 - Photometric Perturbation Module
==========================================
Implements on-the-fly channel-wise Gaussian perturbation.
Applied ONLY to training batches. Validation/test remain clean.
"""

import torch
import numpy as np
import logging

logger = logging.getLogger(__name__)


class ChannelPerturbation:
    """
    Channel-wise Gaussian perturbation for training augmentation.
    
    ε ~ N(0, σ²) applied per-pixel, per-selected-channel.
    This archived implementation operates on ImageNet-normalized tensors.
    Sigma is scaled from [0,255] space to [0,1] space and the result is not
    clipped. This behavior is retained to reproduce the reported runs.
    """
    
    def __init__(self, channels, sigma, seed=None):
        """
        Args:
            channels: list of channels to perturb, e.g., ["R"], ["G"], ["B"], ["R","G","B"]
            sigma: noise standard deviation in [0, 255] scale
            seed: optional random seed (not used for per-call, since we want stochastic augmentation)
        """
        self.channels = channels
        self.sigma_raw = sigma
        self.sigma_normalized = sigma / 255.0
        
        self.channel_map = {"R": 0, "G": 1, "B": 2}
        self.channel_indices = [self.channel_map[c] for c in channels]
        
        logger.info(f"ChannelPerturbation initialized: channels={channels}, "
                    f"sigma={sigma:.2f} (raw), sigma_norm={self.sigma_normalized:.4f}")
    
    def __call__(self, tensor):
        """
        Apply perturbation to a single image tensor.
        
        Args:
            tensor: [C, H, W] ImageNet-normalized image tensor
        
        Returns:
            perturbed tensor
        """
        noise = torch.zeros_like(tensor)
        for ch_idx in self.channel_indices:
            noise[ch_idx] = torch.randn_like(tensor[ch_idx]) * self.sigma_normalized
        
        perturbed = tensor + noise
        return perturbed
    
    def __repr__(self):
        return (f"ChannelPerturbation(channels={self.channels}, "
                f"sigma={self.sigma_raw}, sigma_norm={self.sigma_normalized:.4f})")


class PerturbedTrainDataset(torch.utils.data.Dataset):
    """
    Wrapper dataset that applies channel-wise Gaussian perturbation 
    on-the-fly during training. Applied ONLY to train data.
    """
    
    def __init__(self, base_dataset, channels, sigma):
        """
        Args:
            base_dataset: underlying dataset returning (tensor, label) tuples
            channels: list of channel names to perturb
            sigma: noise std in [0,255] scale
        """
        self.base_dataset = base_dataset
        self.perturbation = ChannelPerturbation(channels, sigma)
        
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        img, label = self.base_dataset[idx]
        img = self.perturbation(img)
        return img, label


def resolve_sigma(config, condition_config):
    """
    Resolve sigma value from condition config.
    
    Args:
        config: base config
        condition_config: condition-specific config
    
    Returns:
        float sigma value in [0, 255] scale, or None if no perturbation
    """
    pert_cfg = condition_config.get('perturbation', {})
    if not pert_cfg.get('enabled', False):
        return None
    
    sigma_key = pert_cfg.get('sigma', config.get('perturbation', {}).get('default_sigma', 'medium'))
    sigma_levels = config.get('perturbation', {}).get('sigma_levels', {
        'low': 12.75,
        'medium': 51.0,
        'high': 102.0
    })
    
    if isinstance(sigma_key, str):
        return sigma_levels.get(sigma_key, 51.0)
    elif isinstance(sigma_key, (int, float)):
        return float(sigma_key)
    else:
        return 51.0


def log_perturbation_config(condition_config, sigma):
    """Log perturbation configuration for traceability."""
    pert_cfg = condition_config.get('perturbation', {})
    
    logger.info("Perturbation config:")
    logger.info(f"  Enabled: {pert_cfg.get('enabled', False)}")
    
    if pert_cfg.get('enabled', False):
        logger.info(f"  Channels: {pert_cfg.get('channels', [])}")
        logger.info(f"  Sigma (raw): {sigma}")
        logger.info(f"  Sigma (normalized): {sigma / 255.0:.4f}")
        logger.info(f"  Applied to: training only")
    else:
        logger.info("  No perturbation applied")
