# Checkpoint: random_focal_20ep

Generated 2026-09-15. Same configuration as
[`prototype_focal_20ep`](../prototype_focal_20ep/README.md), except using
random selection instead of prototype selection -- for comparing the two
selection strategies under otherwise identical settings.

## How this checkpoint was produced

```bash
digit-classification train \
  --data-dir ./data \
  --output-dir ./checkpoints/random_focal_20ep \
  --epochs 20 \
  --class-counts 0=1200,5=300,8=3500 \
  --selection-strategy random \
  --loss-function 1
```

All other options were left at their CLI defaults:

| Option | Value |
| --- | --- |
| `--seed` | 42 |
| `--test-split` | 0.2 |
| `--val-split` | 0.15 |
| `--pca-components` | 30 (unused -- only affects prototype selection) |
| `--batch-size` | 64 |
| `--balance-augment` | off (no augmentation) |
| `--focal-alpha` | 0.2 |
| `--focal-gamma` | 2.0 |

## Files

- `digit-classifier.ckpt` -- the trained Lightning checkpoint.
- `evaluation_report.txt` -- `sklearn.metrics.classification_report` from
  running `evaluate` against this checkpoint's held-out test split.
- `lightning_logs/version_0/metrics.csv` -- raw per-step/per-epoch metrics
  logged during training.

## Reproducing the test set used for evaluation

```bash
digit-classification evaluate \
  --checkpoint-path ./checkpoints/random_focal_20ep/digit-classifier.ckpt \
  --data-dir ./data \
  --class-counts 0=1200,5=300,8=3500 \
  --selection-strategy random
```

(`--seed` and `--test-split` default to the same values used above, so
they don't need to be repeated explicitly.)

## Validation loss / accuracy by epoch

| Epoch | Val loss | Val accuracy | Train loss | Train accuracy |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0.004536 | 0.9800 | 0.020187 | 0.9194 |
| 1 | 0.001999 | 0.9917 | 0.003090 | 0.9835 |
| 2 | 0.002477 | 0.9867 | 0.001392 | 0.9926 |
| 3 | 0.001468 | 0.9883 | 0.000797 | 0.9962 |
| 4 | 0.001007 | 0.9917 | 0.000387 | 0.9994 |
| 5 | 0.000924 | 0.9950 | 0.000232 | 1.0000 |
| 6 | 0.000858 | 0.9950 | 0.000149 | 1.0000 |
| 7 | 0.000771 | 0.9950 | 0.000087 | 1.0000 |
| 8 | 0.000742 | 0.9950 | 0.000064 | 1.0000 |
| 9 | 0.000759 | 0.9967 | 0.000046 | 1.0000 |
| 10 | 0.000723 | 0.9950 | 0.0000398 | 1.0000 |
| 11 | 0.000732 | 0.9950 | 0.0000297 | 1.0000 |
| 12 | 0.000730 | 0.9950 | 0.0000261 | 1.0000 |
| 13 | 0.000676 | 0.9950 | 0.0000218 | 1.0000 |
| 14 | 0.000665 | 0.9950 | 0.0000186 | 1.0000 |
| 15 | 0.000711 | 0.9950 | 0.0000165 | 1.0000 |
| 16 | 0.000663 | 0.9950 | 0.0000139 | 1.0000 |
| 17 | 0.000685 | 0.9950 | 0.0000119 | 1.0000 |
| 18 | 0.000665 | 0.9950 | 0.0000105 | 1.0000 |
| 19 | 0.000673 | 0.9950 | 0.0000092 | 1.0000 |

## Test set evaluation

See `evaluation_report.txt`. Summary: 1.00 overall accuracy on the 1,000
held-out test images (240 zeros, 60 fives, 700 eights).
