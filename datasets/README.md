# Dataset layout

The datasets are not redistributed. Place downloaded files in this directory using the following layout:

```text
datasets/
├── APTOS/
│   ├── train_1.csv
│   ├── valid.csv
│   ├── test.csv
│   ├── train_images/
│   │   └── train_images/
│   │       └── *.png
│   └── val_images/
│       └── val_images/
│           └── *.png
├── skin_dataset_resized/
│   ├── train_set/
│   │   ├── benign/
│   │   └── malignant/
│   └── val_set/
│       ├── benign/
│       └── malignant/
└── The Hyper Kvasir Dataset/
    └── dataset/
        ├── barretts/1. original/{barretts,no-barretts}/
        ├── esophagitis/1. original/{esophagitis,no-esophagitis}/
        ├── polyps/1. original/{polyps,no-polyps}/
        └── ulcerative-colitis/1. original/{ulcerative-colitis,no-ulcerative-colitis}/
```

APTOS CSV files are expected to contain an image identifier in the first column and an integer class label in the second column. Paths can be changed in `configs/base_config.yaml` if your local layout differs.
