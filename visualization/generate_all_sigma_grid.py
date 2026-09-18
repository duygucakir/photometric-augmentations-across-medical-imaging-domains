import os
import glob
import argparse
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torchvision.transforms import ToTensor
from pathlib import Path

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from preprocessing.preprocessor import apply_preprocessing
from augmentation.perturbation import ChannelPerturbation

REPO_ROOT = Path(__file__).resolve().parents[1]


def _choose_image(pattern, label):
    matches = glob.glob(str(pattern))
    if not matches:
        raise FileNotFoundError(f"No images found for {label}: {pattern}")
    return random.choice(matches)


def get_image_paths(data_root):
    base = Path(data_root)
    paths = {
        "APTOS": _choose_image(base / "APTOS/train_images/train_images/*.png", "APTOS"),
        "Skin": _choose_image(base / "skin_dataset_resized/train_set/*/*.jpg", "Skin"),
        "Kvasir: Barretts": _choose_image(base / "The Hyper Kvasir Dataset/dataset/barretts/1. original/barretts/*.jpg", "Barretts"),
        "Kvasir: Esophagitis": _choose_image(base / "The Hyper Kvasir Dataset/dataset/esophagitis/1. original/esophagitis/*.jpg", "Esophagitis"),
        "Kvasir: Polyps": _choose_image(base / "The Hyper Kvasir Dataset/dataset/polyps/1. original/polyps/*.jpg", "Polyps"),
        "Kvasir: Ulcerative-colitis": _choose_image(base / "The Hyper Kvasir Dataset/dataset/ulcerative-colitis/1. original/ulcerative-colitis/*.jpg", "Ulcerative colitis"),
    }
    return paths

def apply_noise_to_image(img_pil, channels, sigma):
    tensor = ToTensor()(img_pil)
    perturber = ChannelPerturbation(channels, sigma)
    perturbed = perturber(tensor)
    perturbed = torch.clamp(perturbed, 0, 1)
    return perturbed.permute(1, 2, 0).numpy()

def main(data_root, output_path):
    random.seed(42)
    paths = get_image_paths(data_root)
    
    datasets = list(paths.keys())
    
    columns = [
        "Original", "Preprocessed", 
        "Blue Low", "Blue Med", "Blue High",
        "Green Low", "Green Med", "Green High",
        "Red Low", "Red Med", "Red High",
        "RGB Low", "RGB Med", "RGB High"
    ]
    
    sigma_levels = {'Low': 12.75, 'Med': 51.0, 'High': 102.0}
    
    pre_config = {
        'preprocessing': {
            'enabled': True,
            'specular_highlight': {'enabled': True, 'threshold': 200, 'kernel_size': 15},
            'histogram_equalization': {'enabled': True, 'color_space': 'HSV', 'channel': 'V'}
        }
    }
    
    fig, axes = plt.subplots(len(datasets), len(columns), figsize=(30, 12))
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    
    for row_idx, ds in enumerate(datasets):
        img_path = paths[ds]
        orig_img = Image.open(img_path).convert('RGB')
        
        # Center crop to square for consistent visualization
        w, h = orig_img.size
        min_dim = min(w, h)
        left = (w - min_dim)/2
        top = (h - min_dim)/2
        right = (w + min_dim)/2
        bottom = (h + min_dim)/2
        orig_img = orig_img.crop((left, top, right, bottom)).resize((224, 224))
        
        pre_img = apply_preprocessing(orig_img, pre_config)
        
        # Original
        ax = axes[row_idx, 0]
        ax.imshow(orig_img)
        ax.set_xticks([]); ax.set_yticks([])
        if row_idx == 0: ax.set_title("Original", fontweight='bold')
        ax.set_ylabel(ds, fontweight='bold')
        
        # Preprocessed
        ax = axes[row_idx, 1]
        ax.imshow(pre_img)
        ax.set_xticks([]); ax.set_yticks([])
        if row_idx == 0: ax.set_title("Preprocessed", fontweight='bold')
        
        col_idx = 2
        for ch_label, ch_list in [('Blue', ['B']), ('Green', ['G']), ('Red', ['R']), ('RGB', ['R', 'G', 'B'])]:
            for sig_label, sig_val in sigma_levels.items():
                ax = axes[row_idx, col_idx]
                torch.manual_seed(42) # For reproducible noise
                noisy_arr = apply_noise_to_image(pre_img, ch_list, sig_val)
                ax.imshow(noisy_arr)
                ax.set_xticks([]); ax.set_yticks([])
                if row_idx == 0: ax.set_title(f"{ch_label} {sig_label}", fontweight='bold')
                col_idx += 1

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    print(f"Saved to {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate the paper's sigma grid")
    parser.add_argument(
        '--data-root',
        default=str(REPO_ROOT / 'datasets'),
        help='Directory containing APTOS, skin_dataset_resized, and The Hyper Kvasir Dataset',
    )
    parser.add_argument(
        '--output',
        default=str(REPO_ROOT / 'outputs/figures/all_sigma_grid.png'),
        help='Output image path',
    )
    args = parser.parse_args()
    main(args.data_root, args.output)
