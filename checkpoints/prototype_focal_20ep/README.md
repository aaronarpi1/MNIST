# Checkpoint: prototype_focal_20ep

Generated 2026-09-15.

## How this checkpoint was produced

```bash
digit-classification train \
  --data-dir ./data \
  --output-dir ./checkpoints/prototype_focal_20ep \
  --epochs 20 \
  --class-counts 0=1200,5=300,8=3500 \
  --selection-strategy prototype \
  --loss-function 1
```

All other options were left at their CLI defaults:

| Option | Value |
| --- | --- |
| `--seed` | 42 |
| `--test-split` | 0.2 |
| `--val-split` | 0.15 |
| `--pca-components` | 30 |
| `--batch-size` | 64 |
| `--balance-augment` | off (no augmentation) |
| `--focal-alpha` | 0.2 |
| `--focal-gamma` | 2.0 |

This uses the exact class counts from the problem statement (1,200 zeros,
300 fives, 3,500 eights), the PCA + K-Means "prototype" selection strategy
(rather than random sampling), focal loss with its default alpha/gamma,
and no minority-class augmentation.

## Files

- `digit-classifier.ckpt` -- the trained Lightning checkpoint.
- `evaluation_report.txt` -- `sklearn.metrics.classification_report` from
  running `evaluate` against this checkpoint's held-out test split.
- `lightning_logs/version_0/metrics.csv` -- raw per-step/per-epoch metrics
  logged during training.

## Reproducing the test set used for evaluation

Because there's no shared config file, the exact test split is
reconstructed from the curation parameters, not stored separately. To
reproduce it (or re-run evaluation), pass the same curation options used
for training:

```bash
digit-classification evaluate \
  --checkpoint-path ./checkpoints/prototype_focal_20ep/digit-classifier.ckpt \
  --data-dir ./data \
  --class-counts 0=1200,5=300,8=3500 \
  --selection-strategy prototype
```

(`--seed`, `--test-split`, and `--pca-components` all default to the same
values used above, so they don't need to be repeated explicitly.)

## Validation loss / accuracy by epoch

| Epoch | Val loss | Val accuracy | Train loss | Train accuracy |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0.007649 | 0.9667 | 0.021534 | 0.9088 |
| 1 | 0.002661 | 0.9850 | 0.003470 | 0.9838 |
| 2 | 0.001784 | 0.9917 | 0.001369 | 0.9929 |
| 3 | 0.002014 | 0.9917 | 0.000721 | 0.9979 |
| 4 | 0.001114 | 0.9933 | 0.000512 | 0.9994 |
| 5 | 0.001091 | 0.9933 | 0.000263 | 0.9994 |
| 6 | 0.001039 | 0.9933 | 0.000158 | 1.0000 |
| 7 | 0.001068 | 0.9933 | 0.000111 | 1.0000 |
| 8 | 0.000928 | 0.9967 | 0.000083 | 1.0000 |
| 9 | 0.001168 | 0.9950 | 0.000062 | 1.0000 |
| 10 | 0.001284 | 0.9933 | 0.000051 | 1.0000 |
| 11 | 0.001271 | 0.9950 | 0.000520 | 0.9979 |
| 12 | 0.002325 | 0.9883 | 0.000599 | 0.9982 |
| 13 | 0.001662 | 0.9900 | 0.000518 | 0.9974 |
| 14 | 0.000872 | 0.9950 | 0.000219 | 0.9985 |
| 15 | 0.001155 | 0.9933 | 0.0000336 | 1.0000 |
| 16 | 0.001030 | 0.9967 | 0.0000210 | 1.0000 |
| 17 | 0.000943 | 0.9967 | 0.0000169 | 1.0000 |
| 18 | 0.000894 | 0.9967 | 0.0000129 | 1.0000 |
| 19 | 0.000882 | 0.9967 | 0.0000122 | 1.0000 |

## Test set evaluation

See `evaluation_report.txt`. Summary: 0.99 overall accuracy on the 1,000
held-out test images (240 zeros, 60 fives, 700 eights).
