"""Evaluation utilities: run a trained checkpoint over the held-out test
split and report standard classification metrics.

This module only handles plumbing (loading a checkpoint, running
inference, formatting a report) -- it depends on `DigitClassifier` being
implemented (see `digit_classification/model.py`). Every parameter needed
to reproduce the curated test split is a plain argument here, supplied by
`cli.evaluate`; there is no shared config object.
"""

import numpy as np
import torch
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader

from digit_classification.data import (
    CuratedDigitsDataset,
    SelectionStrategy,
    download_mnist,
    prepare_datasets,
)
from digit_classification.model import DigitClassifier


def load_model(checkpoint_path: str) -> DigitClassifier:
    """Load a trained `DigitClassifier` from a Lightning checkpoint."""
    return DigitClassifier.load_from_checkpoint(checkpoint_path)


def predict_labels(
    model, dataloader: DataLoader
) -> tuple[np.ndarray, np.ndarray]:
    """Run inference over `dataloader` and return (true_labels,
    predicted_labels)."""
    model.eval()
    all_true = []
    all_pred = []
    with torch.no_grad():
        for images, labels in dataloader:
            probabilities = model.predict_step((images, labels))
            predicted = torch.argmax(probabilities, dim=1)
            all_true.append(labels.numpy())
            all_pred.append(predicted.numpy())
    true_labels = np.concatenate(all_true)
    predicted_labels = np.concatenate(all_pred)
    return true_labels, predicted_labels


def evaluate_checkpoint(
    checkpoint_path: str,
    data_dir: str,
    class_counts: dict[int, int],
    seed: int,
    selection_strategy: SelectionStrategy,
    pca_components: int,
    test_split: float,
    batch_size: int,
) -> str:
    """Evaluate a checkpoint on the curated test split and return a
    report string.

    `data_dir`, `class_counts`, `seed`, `selection_strategy`, and
    `pca_components`/`test_split` must match the values passed to
    `cli.train` in order to reproduce the exact same held-out test split
    the model never saw during training.
    """
    model = load_model(checkpoint_path)
    dataset = download_mnist(data_dir)
    splits = prepare_datasets(
        data_dir=data_dir,
        class_counts=class_counts,
        seed=seed,
        selection_strategy=selection_strategy,
        pca_components=pca_components,
        test_split=test_split,
        val_split=0.15,
        dataset=dataset,
    )
    classes = tuple(sorted(class_counts))
    test_dataset = CuratedDigitsDataset(
        splits.test_images, splits.test_labels, classes
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )

    true_labels, predicted_labels = predict_labels(model, test_loader)
    target_names = [str(digit) for digit in classes]
    class_indices = list(range(len(classes)))
    return classification_report(
        true_labels,
        predicted_labels,
        labels=class_indices,
        target_names=target_names,
    )
