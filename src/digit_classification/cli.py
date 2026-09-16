"""Typer command line interface for the digit-classification project.

Every parameter is declared here as a CLI option with a default -- there
is no config file or environment variable layer anywhere in this project.
Run any command with --help to see its full set of options and defaults.

    digit-classification download-data --data-dir <dir>
    digit-classification train --data-dir <dir> --output-dir <dir>
    digit-classification evaluate --checkpoint-path <ckpt> --data-dir <dir>
    digit-classification predict --checkpoint-path <ckpt> --input-path <image>
    digit-classification review-augmentations --digit 5 --output-dir ./review
"""

import json
import os

import numpy as np
import torch
import typer
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from PIL import Image
from torch.utils.data import DataLoader

from digit_classification.data import (
    BalanceTarget,
    CuratedDigitsDataset,
    SelectionStrategy,
    balance_with_augmentation,
    download_mnist,
    parse_class_counts,
    parse_classes,
    prepare_datasets,
)
from digit_classification.evaluation import evaluate_checkpoint
from digit_classification.model import DigitClassifier
from digit_classification.visualize import save_augmentation_examples

app = typer.Typer(
    help=(
        "Curate, train, evaluate, and run inference on a curated "
        "MNIST classifier."
    )
)


def _validate_class_counts(value: str) -> str:
    """Validate a `--class-counts` value at CLI parse time.

    Must be non-empty, comma-separated `label=count` pairs of integers,
    with no duplicate labels, each label a valid MNIST digit (0-9), and
    each count a positive integer.
    """
    try:
        class_counts = parse_class_counts(value)
    except ValueError:
        raise typer.BadParameter(
            "must be comma-separated label=count pairs of integers, "
            "e.g. '0=1200,5=300,8=3500'"
        )

    pairs = [pair for pair in value.split(",") if pair.strip()]
    if not class_counts or len(pairs) != len(class_counts):
        raise typer.BadParameter(
            "must specify at least one label=count pair, with no "
            "duplicate labels."
        )
    for label, count in class_counts.items():
        if not 0 <= label <= 9:
            raise typer.BadParameter(
                f"digit label {label} is not a valid MNIST digit (0-9)."
            )
        if count <= 0:
            raise typer.BadParameter(
                f"count for digit {label} must be a positive integer, "
                f"got {count}."
            )

    return value


def _validate_classes(value: str) -> str:
    """Validate a `--classes` value at CLI parse time.

    Must be non-empty, comma-separated integers, with no duplicates, each
    a valid MNIST digit (0-9).
    """
    try:
        classes = parse_classes(value)
    except ValueError:
        raise typer.BadParameter(
            "must be comma-separated integers, e.g. '0,5,8'"
        )

    entries = [entry for entry in value.split(",") if entry.strip()]
    if not classes or len(entries) != len(classes):
        raise typer.BadParameter(
            "must specify at least one digit label, with no duplicates."
        )
    for label in classes:
        if not 0 <= label <= 9:
            raise typer.BadParameter(
                f"digit label {label} is not a valid MNIST digit (0-9)."
            )

    return value


def _data_dir_option() -> str:
    """Shared `--data-dir` option: identical everywhere it's used to
    read (rather than create) the downloaded MNIST data."""
    return typer.Option(
        "./data",
        "--data-dir",
        help="Directory MNIST is downloaded to / read from.",
    )


def _checkpoint_path_option() -> str:
    """Shared `--checkpoint-path` option: identical everywhere it's
    used to load a trained checkpoint."""
    return typer.Option(
        ...,
        "--checkpoint-path",
        help="Path to a trained Lightning checkpoint.",
    )


def _class_counts_option(help: str) -> str:
    """Shared `--class-counts` default value and validation callback.
    `help` is still passed in per call site, since its wording differs
    between `train` and `evaluate`."""
    return typer.Option(
        "0=1200,5=300,8=3500",
        "--class-counts",
        callback=_validate_class_counts,
        help=help,
    )


def _seed_option(help: str) -> int:
    """Shared `--seed` default value. `help` is still passed in per
    call site, since its wording differs across commands."""
    return typer.Option(42, "--seed", help=help)


@app.command("download-data")
def download_data(
    data_dir: str = typer.Option(
        "./data", "--data-dir", help="Directory to download MNIST into."
    ),
):
    """Download the MNIST training dataset to the specified data directory."""
    download_mnist(data_dir)
    typer.echo(f"MNIST training data ready at {data_dir}")


@app.command()
def train(
    data_dir: str = _data_dir_option(),
    output_dir: str = typer.Option(
        "./outputs",
        "--output-dir",
        help="Directory the trained checkpoint and logs are written to.",
    ),
    epochs: int = typer.Option(
        15,
        "--epochs",
        min=1,
        max=20,
        help="Training epochs (1-20, matching the problem statement's cap).",
    ),
    batch_size: int = typer.Option(
        64,
        "--batch-size",
        min=1,
        max=512,
        help=(
            "Training batch size (1-512), sized for a curated set of "
            "a few thousand images."
        ),
    ),
    seed: int = _seed_option(
        help=(
            "Seed controlling image selection, the train/val/test "
            "split, augmentation, and model init/batching. Pass the "
            "same value to `evaluate` to reproduce the same held-out "
            "test split."
        ),
    ),
    class_counts: str = _class_counts_option(
        help=(
            "Images to curate per digit label, as label=count pairs. "
            "Pass the same value to `evaluate` to reproduce the same "
            "held-out test split."
        ),
    ),
    test_split: float = typer.Option(
        0.2,
        "--test-split",
        min=0.0,
        max=1.0,
        help=(
            "Fraction of the curated set held out for evaluation. "
            "Pass the same value to `evaluate` to reproduce the same "
            "held-out test split."
        ),
    ),
    val_split: float = typer.Option(
        0.15,
        "--val-split",
        min=0.0,
        max=1.0,
        help=(
            "Fraction of the remaining (non-test) curated set held "
            "out for validation. Higher means a larger, less "
            "sample-noisy validation set at the cost of less "
            "training data."
        ),
    ),
    selection_strategy: SelectionStrategy = typer.Option(
        SelectionStrategy.RANDOM,
        "--selection-strategy",
        help=(
            "How to choose which images represent each class. Pass "
            "the same value to `evaluate` to reproduce the same "
            "held-out test split."
        ),
    ),
    pca_components: int = typer.Option(
        30,
        "--pca-components",
        min=1,
        help=(
            "PCA dimensions used by the 'prototype' selection "
            "strategy. Pass the same value to `evaluate` to "
            "reproduce the same held-out test split."
        ),
    ),
    balance_augment: bool = typer.Option(
        False,
        "--balance-augment/--no-balance-augment",
        help=(
            "Top up minority classes in the training split with "
            "randomized augmentations so all classes are balanced "
            "going into training."
        ),
    ),
    balance_target: str = typer.Option(
        BalanceTarget.MAX.value,
        "--balance-target",
        help=(
            "Target count per class when balancing: 'max' matches "
            "the largest class, or set an explicit integer."
        ),
    ),
    loss_function: int = typer.Option(
        0,
        "--loss-function",
        min=0,
        max=1,
        help="Loss function: 0 = cross-entropy, 1 = focal loss.",
    ),
    focal_alpha: float = typer.Option(
        0.2,
        "--focal-alpha",
        min=0.0,
        max=1.0,
        help=(
            "Alpha parameter for focal loss. Only used when "
            "--loss-function is 1."
        ),
    ),
    focal_gamma: float = typer.Option(
        2.0,
        "--focal-gamma",
        min=0.0,
        max=10.0,
        help=(
            "Gamma (focusing) parameter for focal loss -- higher "
            "down-weights easy examples more, focusing training on "
            "hard ones. Only used when --loss-function is 1."
        ),
    ),
):
    """Train the model on the curated dataset and save a checkpoint."""
    seed_everything(seed, workers=True)
    parsed_class_counts = parse_class_counts(class_counts)
    classes = tuple(sorted(parsed_class_counts))

    dataset = download_mnist(data_dir)
    splits = prepare_datasets(
        data_dir=data_dir,
        class_counts=parsed_class_counts,
        seed=seed,
        selection_strategy=selection_strategy,
        pca_components=pca_components,
        test_split=test_split,
        val_split=val_split,
        dataset=dataset,
    )

    train_images, train_labels = splits.train_images, splits.train_labels
    if balance_augment:
        target = (
            BalanceTarget.MAX
            if balance_target == BalanceTarget.MAX
            else int(balance_target)
        )
        train_images, train_labels, _ = balance_with_augmentation(
            train_images, train_labels, seed=seed, target=target
        )

    train_dataset = CuratedDigitsDataset(train_images, train_labels, classes)
    val_dataset = CuratedDigitsDataset(
        splits.val_images, splits.val_labels, classes
    )
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = DigitClassifier(
        num_classes=len(classes),
        loss_function=loss_function,
        focal_alpha=focal_alpha,
        focal_gamma=focal_gamma,
    )

    os.makedirs(output_dir, exist_ok=True)
    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir, filename="digit-classifier"
    )
    trainer = Trainer(
        default_root_dir=output_dir,
        accelerator="cpu",
        max_epochs=epochs,
        callbacks=[checkpoint_callback],
    )
    trainer.fit(model, train_loader, val_loader)
    typer.echo(f"Training complete. Checkpoint saved under {output_dir}")


@app.command()
def evaluate(
    checkpoint_path: str = _checkpoint_path_option(),
    data_dir: str = _data_dir_option(),
    seed: int = _seed_option(
        help=(
            "Must match the value passed to `train` to reproduce "
            "the same test split."
        ),
    ),
    class_counts: str = _class_counts_option(
        help=(
            "Images curated per digit label, as label=count pairs. "
            "Must match the value passed to `train`."
        ),
    ),
    test_split: float = typer.Option(
        0.2,
        "--test-split",
        min=0.0,
        max=1.0,
        help=(
            "Fraction of the curated set held out for evaluation. "
            "Must match the value passed to `train`."
        ),
    ),
    selection_strategy: SelectionStrategy = typer.Option(
        SelectionStrategy.RANDOM,
        "--selection-strategy",
        help=(
            "How images were chosen to represent each class. Must "
            "match `train`."
        ),
    ),
    pca_components: int = typer.Option(
        30,
        "--pca-components",
        min=1,
        help=(
            "PCA dimensions used by the 'prototype' selection "
            "strategy. Must match `train`."
        ),
    ),
    batch_size: int = typer.Option(
        64,
        "--batch-size",
        min=1,
        max=512,
        help="Evaluation batch size (1-512).",
    ),
):
    """Evaluate a trained checkpoint on the held-out test split.

    The curation options (--seed, --class-counts, --test-split,
    --selection-strategy, --pca-components) must match the values passed
    to `train` -- they're what reproduces the exact same test split the
    model never saw during training.
    """
    report = evaluate_checkpoint(
        checkpoint_path=checkpoint_path,
        data_dir=data_dir,
        class_counts=parse_class_counts(class_counts),
        seed=seed,
        selection_strategy=selection_strategy,
        pca_components=pca_components,
        test_split=test_split,
        batch_size=batch_size,
    )
    typer.echo(report)


@app.command()
def predict(
    checkpoint_path: str = _checkpoint_path_option(),
    input_path: str = typer.Option(
        ...,
        "--input-path",
        help="Path to a grayscale digit image (resized to 28x28).",
    ),
    classes: str = typer.Option(
        "0,5,8",
        "--classes",
        callback=_validate_classes,
        help=(
            "Comma-separated digit labels the model predicts, in "
            "the same set used for `train`."
        ),
    ),
):
    """Predict the digit label of a single image, printing per-class
    probabilities."""
    if not os.path.isfile(input_path):
        typer.echo(f"No such image file: {input_path}", err=True)
        raise typer.Exit(code=1)

    parsed_classes = parse_classes(classes)

    model = DigitClassifier.load_from_checkpoint(checkpoint_path)
    model.eval()

    image = Image.open(input_path)
    image = image.convert("L")
    image = image.resize((28, 28))

    pixel_values = np.array(image, dtype=np.float32)
    normalized_pixel_values = pixel_values / 255.0
    image_tensor = torch.from_numpy(normalized_pixel_values)
    # forward() expects a batch of images shaped (batch_size, channels,
    # height, width); this is a single 1-channel image, so add both of
    # those dimensions.
    image_tensor = image_tensor.unsqueeze(0)  # add the channel dimension
    image_tensor = image_tensor.unsqueeze(0)  # add the batch dimension

    dummy_label = torch.zeros(1, dtype=torch.long)
    with torch.no_grad():
        probabilities = model.predict_step((image_tensor, dummy_label))

    probability_values = probabilities[0].tolist()
    result = {
        str(digit): float(probability)
        for digit, probability in zip(parsed_classes, probability_values)
    }
    typer.echo(json.dumps(result, indent=2))


@app.command("review-augmentations")
def review_augmentations(
    data_dir: str = _data_dir_option(),
    digit: int = typer.Option(
        5,
        "--digit",
        min=0,
        max=9,
        help="Which digit's images to sample originals from.",
    ),
    num_examples: int = typer.Option(
        5,
        "--num-examples",
        min=1,
        max=10,
        help="How many before/after example images to save.",
    ),
    seed: int = _seed_option(
        help=(
            "Seed controlling which images are sampled and how "
            "they're augmented."
        ),
    ),
    output_dir: str = typer.Option(
        "./augmentation_review",
        "--output-dir",
        help=(
            "Directory to save one side-by-side (original | "
            "augmented) image per example into."
        ),
    ),
):
    """Save side-by-side before/after comparison images, one per example,
    for visually sanity-checking the augmentation workflow."""
    dataset = download_mnist(data_dir)
    saved_paths = save_augmentation_examples(
        dataset,
        digit=digit,
        num_examples=num_examples,
        seed=seed,
        output_dir=output_dir,
    )
    typer.echo(
        f"Saved {len(saved_paths)} before/after example(s) to {output_dir}"
    )


if __name__ == "__main__":
    app()
