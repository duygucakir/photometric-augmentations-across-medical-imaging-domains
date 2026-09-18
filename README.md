# Photometric Augmentations Across Medical Imaging Domains

Research code for **“On the Transferability of Photometric Augmentations Across Medical Imaging Domains.”** The repository evaluates channel-wise Gaussian photometric perturbations across retinal fundus imaging (APTOS), dermoscopic skin-lesion classification (ISIC-derived data), and gastrointestinal endoscopy (HyperKvasir).

The experimental design uses a fixed ImageNet-pretrained Xception backbone, deterministic preprocessing, stratified five-fold cross-validation, and perturbations applied only to training images.

## Experimental design

- **Modalities:** retinal fundus, dermoscopy, and gastrointestinal endoscopy.
- **Tasks:** APTOS five-class grading, skin-lesion binary classification, and four HyperKvasir binary subtasks (Barrett's, esophagitis, polyps, and ulcerative colitis).
- **Preprocessing:** specular-highlight detection and Telea inpainting, followed by HSV V-channel histogram equalization.
- **Perturbations:** additive Gaussian noise on R, G, B, or all RGB channels. In the archived implementation, `sigma / 255` noise is added to the ImageNet-normalized training tensor.
- **Intensity levels:** low (`sigma = 12.75`), medium (`sigma = 51.0`), and high (`sigma = 102.0`) in `[0, 255]` pixel space.
- **Evaluation:** accuracy, macro-F1, AUC, balanced accuracy, MCC, precision, recall, PR-AUC, and per-class F1.
- **Statistics:** paired t-tests, Holm-Bonferroni correction, Cohen's dz, and confidence intervals.
- **Class imbalance:** the archived runs use inverse-frequency weighted sampling by default.

## Repository layout

```text
.
├── run_pipeline.py              # Main experiment runner
├── configs/                     # Base settings and experiment registry
├── data/                        # Dataset loading and split logic
├── preprocessing/               # Deterministic preprocessing
├── augmentation/                # Channel-wise Gaussian perturbations
├── models/                      # Xception classifier
├── training/                    # Training, checkpointing, and resume logic
├── evaluation/                  # Metrics and result serialization
├── stats_analysis/              # Statistical comparisons
├── analysis/                    # Channel-distribution and embedding analyses
├── visualization/               # Figures and qualitative grids
├── splits/                      # Published fold assignments (portable paths)
├── results/                     # Compact reported results and statistical tables
└── datasets/                    # Local datasets; ignored by Git
```

Large datasets, checkpoints, logs, predictions, and generated figures are intentionally excluded from version control.

## Installation

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The recorded experiment environment used Python 3.13.5, PyTorch 2.7.1, NumPy 2.2.6, and Apple MPS. See [`results/environment.json`](results/environment.json).

## Dataset preparation

Download the datasets from their original distribution channels:

- [APTOS 2019 Blindness Detection](https://www.kaggle.com/c/aptos2019-blindness-detection)
- [ISIC Archive](https://www.isic-archive.com/)
- [HyperKvasir](https://doi.org/10.1038/s41597-020-00622-y)

Place them under `datasets/` using the structure documented in [`datasets/README.md`](datasets/README.md). Dataset files are not redistributed by this repository.

The committed JSON files in `splits/` preserve the exact fold indices used for the reported experiments. Their image paths are relative to the repository root. If the local dataset file ordering or contents differ from the recorded release, regenerate the split files deliberately after reviewing the implications for direct result comparability.

## Running experiments

Run commands from the repository root.

Show all options:

```bash
python run_pipeline.py --help
```

Run one baseline on APTOS:

```bash
python run_pipeline.py --dataset aptos --condition baseline_pre
```

Run one HyperKvasir task and perturbation condition:

```bash
python run_pipeline.py \
  --dataset hyperkvasir \
  --task polyps \
  --condition pre_RGB_high
```

Run the default paper grid across all tasks:

```bash
python run_pipeline.py
```

No-preprocessing ablations are available in `configs/experiment_registry.yaml` and can be run individually with `--condition`.

The runner is resume-safe: checkpoints and per-fold status files are written to `outputs/`, which is ignored by Git.

## Reproducing the qualitative perturbation grid

```bash
python visualization/generate_all_sigma_grid.py
```

Custom dataset and output locations can be supplied with `--data-root` and `--output`.

## Reported results

The complete fold-level metrics are in [`results/fold_level_results.md`](results/fold_level_results.md). Machine-readable statistical comparisons are under `results/statistical_tests/`.

Selected macro-F1 results from the manuscript:

| Dataset/task | Baseline | Best observed condition | Best macro-F1 | RGB-high macro-F1 |
|---|---:|---|---:|---:|
| APTOS | 0.6318 | B-low | 0.6366 | 0.1960 |
| Skin | 0.9259 | B-medium | 0.9400 | 0.8665 |
| Barrett's | 0.7165 | G-low | 0.7175 | 0.5071 |
| Esophagitis | 0.8631 | RGB-low | 0.8695 | 0.5401 |
| Polyps | 0.9896 | R-low (closest) | 0.9893 | 0.7165 |
| Ulcerative colitis | 0.9769 | RGB-low (closest) | 0.9762 | 0.7964 |

## Reproducibility notes

- The random seed is fixed at `42`.
- Validation and test images are never perturbed.
- The same saved folds are reused across all conditions.
- Patient identifiers were unavailable, so stratification is performed at image level.
- Pretrained Xception weights are downloaded by `timm` on first use.
- Full training is computationally expensive; run a single dataset and condition first to validate the environment.

### Archived implementation details to preserve

The included code is the implementation that produced the archived results. Two details should be kept in mind when comparing the repository with prose descriptions of the method:

1. Channel noise is added after torchvision conversion and ImageNet normalization. The tensor is not clipped after noise is added.
2. `training.use_weighted_sampler` defaults to `true`, reproducing the inverse-frequency `WeightedRandomSampler` used by the archived runner. Set it to `false` to train with ordinary shuffled batches, but doing so changes the experimental protocol and will not be directly comparable to the included results.

Changing either behavior requires rerunning the affected experiments and updating the reported results.

## Citation

If you use this code, cite the associated article:

> On the Transferability of Photometric Augmentations Across Medical Imaging Domains.

Publication metadata and a DOI can be added here when available.

## License

No software license has been selected yet. Add an appropriate `LICENSE` file before publishing the repository if you intend others to reuse or modify the code.
