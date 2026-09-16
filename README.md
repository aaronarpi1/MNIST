# Digit Classification

A curated, deliberately imbalanced MNIST classifier (digits 0, 5, 8), built
as a Typer CLI backed by PyTorch Lightning.

## Setup

Requires Python 3.11+. Check your default interpreter first:

```bash
python3 --version
```

If it's older than 3.11, install one (e.g. `brew install python@3.11` on
macOS) and use it explicitly below in place of `python3`.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[test]"
```

## Project layout

```
src/digit_classification/
    cli.py          Typer CLI: download-data, train, evaluate, predict,
                     review-augmentations
    data.py         MNIST download, curation/selection, train/val/test
                     split, minority-class augmentation, Dataset class
    model.py        DigitClassifier (LightningModule) and FocalLoss
    evaluation.py    Checkpoint loading and classification-report
                     evaluation on the held-out test split
    visualize.py    Saves before/after augmentation example images
tests/              pytest suite, one file per src module, plus
                     test_cli.py for the CLI itself
tests/test_images/  Real MNIST JPEGs checked in for fast, reproducible
                     tests (no download needed)
scripts/            One-off dev utilities, not part of the test suite
```

There is no config file or environment variable layer -- every parameter is
a CLI option declared in `cli.py` with its default right there, validated
by Typer at parse time. Run any command with `--help` for the full,
authoritative list of options.

## CLI usage

```bash
digit-classification download-data --data-dir ./data

digit-classification train \
    --data-dir ./data --output-dir ./outputs \
    [--epochs 1-20, default 15] [--batch-size 1-512, default 64] \
    [--seed 42] [--class-counts 0=1200,5=300,8=3500] \
    [--test-split 0.2] [--val-split 0.15] \
    [--selection-strategy random|prototype] [--pca-components 30] \
    [--balance-augment] [--balance-target max] \
    [--loss-function 0=cross-entropy|1=focal, default 0] \
    [--focal-alpha 0.0-1.0, default 0.2] [--focal-gamma 0.0-10.0, default 2.0]

digit-classification evaluate \
    --checkpoint-path ./outputs/digit-classifier.ckpt --data-dir ./data \
    [--seed 42] [--class-counts 0=1200,5=300,8=3500] [--test-split 0.2] \
    [--selection-strategy random|prototype] [--pca-components 30] \
    [--batch-size 64]

digit-classification predict \
    --checkpoint-path ./outputs/digit-classifier.ckpt --input-path ./digit.png \
    [--classes 0,5,8]

digit-classification review-augmentations \
    [--digit 5] [--num-examples 5] [--seed 42] [--output-dir ./augmentation_review]
```

`evaluate` needs the same curation options (`--seed`, `--class-counts`,
`--test-split`, `--selection-strategy`, `--pca-components`) passed to
`train`, to reconstruct the identical held-out test split. `predict`'s
`--classes` should match the digits the checkpoint was trained on -- it's
only used to label the JSON output.

`review-augmentations` is a dev/debug helper: it saves one side-by-side
(original | augmented) PNG per sampled example, so augmentations can be
checked visually.

### Selection strategies

- **random**: draws `class_counts[label]` indices per class without
  replacement, seeded for reproducibility.
- **prototype**: projects each class's images onto their top principal
  components (PCA), clusters them with K-Means, and keeps the image
  closest to each cluster centroid -- biasing selection toward
  representative examples of each digit.

### Balancing

`--balance-augment` tops up under-represented classes in the *training*
split only (validation/test always reflect the true curated imbalance) by
resampling existing images and applying a small randomized augmentation
(rotation, translation, scale, shear, stroke thickness, brightness, noise).

## Example run

```bash
digit-classification train \
    --data-dir ./data --output-dir ./checkpoints/prototype_focal_20ep \
    --epochs 20 --class-counts 0=1200,5=300,8=3500 \
    --selection-strategy prototype --loss-function 1
```

20 epochs, PCA/K-Means prototype selection, focal loss (default
alpha/gamma), no augmentation. Final validation accuracy 0.997, validation
loss 0.00088. Test-set evaluation:

```
              precision    recall  f1-score   support

           0       0.99      0.99      0.99       240
           5       1.00      0.93      0.97        60
           8       0.99      1.00      1.00       700

    accuracy                           0.99      1000
```

Full per-epoch metrics, the checkpoint, and the evaluation report are
saved under `checkpoints/prototype_focal_20ep/` (see its `README.md`).

## Tests

```bash
pytest
```

Tests use real MNIST images checked into `tests/test_images/` rather than
a live download, so the suite runs fully offline and deterministically.
