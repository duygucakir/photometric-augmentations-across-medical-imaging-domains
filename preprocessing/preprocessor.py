"""
Phase 3 - Deterministic Preprocessing Module
=============================================
Implements specular highlight detection/inpainting,
HSV conversion, histogram equalization on V channel.
"""

import numpy as np
import cv2
from PIL import Image
import logging

logger = logging.getLogger(__name__)


def detect_specular_highlights(image_np, threshold=200):
    """
    Detect specular highlights using intensity thresholding.
    
    Args:
        image_np: numpy array (H, W, 3) in RGB, uint8
        threshold: intensity threshold for specular detection
    
    Returns:
        mask: binary mask where 1 = specular highlight
    """
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    return mask


def inpaint_specular(image_np, mask, kernel_size=15):
    """
    Inpaint specular highlight regions using Telea's method.
    
    Args:
        image_np: numpy array (H, W, 3) in RGB, uint8
        mask: binary mask from specular detection
        kernel_size: inpainting radius
    
    Returns:
        inpainted image as numpy array
    """
    # Convert to BGR for OpenCV
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    inpainted = cv2.inpaint(image_bgr, mask, kernel_size, cv2.INPAINT_TELEA)
    return cv2.cvtColor(inpainted, cv2.COLOR_BGR2RGB)


def histogram_equalize_v(image_np):
    """
    Apply histogram equalization on the V channel in HSV space.
    
    Args:
        image_np: numpy array (H, W, 3) in RGB, uint8
    
    Returns:
        equalized image as numpy array in RGB
    """
    hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)
    hsv[:, :, 2] = cv2.equalizeHist(hsv[:, :, 2])
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def apply_preprocessing(image, config):
    """
    Apply the full deterministic preprocessing pipeline.
    
    Steps:
    1. Specular highlight detection and inpainting
    2. HSV conversion + histogram equalization on V
    3. Convert back to RGB
    
    Args:
        image: PIL Image (RGB)
        config: preprocessing config dict
    
    Returns:
        preprocessed PIL Image (RGB)
    """
    pre_cfg = config.get('preprocessing', {})
    if not pre_cfg.get('enabled', False):
        return image
    
    # Convert to numpy
    img_np = np.array(image)
    
    # Step 1: Specular highlight detection and inpainting
    spec_cfg = pre_cfg.get('specular_highlight', {})
    if spec_cfg.get('enabled', True):
        threshold = spec_cfg.get('threshold', 200)
        kernel_size = spec_cfg.get('kernel_size', 15)
        mask = detect_specular_highlights(img_np, threshold)
        if np.any(mask > 0):
            img_np = inpaint_specular(img_np, mask, kernel_size)
    
    # Step 2: Histogram equalization on V channel
    heq_cfg = pre_cfg.get('histogram_equalization', {})
    if heq_cfg.get('enabled', True):
        img_np = histogram_equalize_v(img_np)
    
    return Image.fromarray(img_np)


class PreprocessorWrapper:
    def __init__(self, config):
        self.config = config
        
    def __call__(self, img):
        return apply_preprocessing(img, self.config)

def create_preprocessing_fn(config):
    """
    Create a preprocessing function to be passed to dataset.
    
    Args:
        config: full experiment config
    
    Returns:
        preprocessing function or None
    """
    if config.get('preprocessing', {}).get('enabled', False):
        return PreprocessorWrapper(config)
    return None


def log_preprocessing_config(config, dataset_name, task=None):
    """Log preprocessing configuration for traceability."""
    pre_cfg = config.get('preprocessing', {})
    task_str = f"/{task}" if task else ""
    
    logger.info(f"Preprocessing config for {dataset_name}{task_str}:")
    logger.info(f"  Enabled: {pre_cfg.get('enabled', False)}")
    
    if pre_cfg.get('enabled', False):
        spec_cfg = pre_cfg.get('specular_highlight', {})
        logger.info(f"  Specular highlight detection: {spec_cfg.get('enabled', True)}")
        logger.info(f"    Threshold: {spec_cfg.get('threshold', 200)}")
        logger.info(f"    Kernel size: {spec_cfg.get('kernel_size', 15)}")
        
        heq_cfg = pre_cfg.get('histogram_equalization', {})
        logger.info(f"  Histogram equalization: {heq_cfg.get('enabled', True)}")
        logger.info(f"    Color space: {heq_cfg.get('color_space', 'HSV')}")
        logger.info(f"    Channel: {heq_cfg.get('channel', 'V')}")
