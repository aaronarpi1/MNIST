"""Data set creation for a subset of MNIST data.

Pipeline:
  1. Download the full MNIST training set.
  2. Curate the requested number of images per class (`curate_indices`),
     using either uniform random sampling or PCA + K-Means "best
     prototype" selection (`SelectionStrategy`).
  3. Split the curated set into train/val/test, stratified by class, with
     a fixed seed so the split is reproducible.
  4. Callers that want to balance minority classes (e.g. `cli.train`)
     call `balance_with_augmentation` themselves on the returned training
     split -- validation and test are never balanced here, so they always
     reflect the true, curated imbalance.
"""

from enum import Enum

import numpy as np
import torch
from PIL import Image, ImageFilter
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision.datasets import MNIST
from torchvision.transforms import functional as TF


class SelectionStrategy(str, Enum):
    """How images are chosen to represent each curated digit class.

    Validated at the CLI boundary (Typer/Click enforces this is one of
    the two members), so `curate_indices` can dispatch on it directly
    with no "unknown strategy" fallback needed.
    """

    RANDOM = "random"
    PROTOTYPE = "prototype"


class BalanceTarget(str, Enum):
    """Sentinel for the `--balance-target` CLI option.

    `MAX` means "match the largest class's count"; any other value passed
    on the CLI is treated as an explicit integer target count instead (see
    `balance_with_augmentation`'s `target` parameter).
    """

    MAX = "max"


def parse_class_counts(raw: str) -> dict[int, int]:
    """Parse a `--class-counts` value like "0=1200,5=300,8=3500" into a
    dict."""
    counts: dict[int, int] = {}
    for pair in raw.split(","):
        pair = pair.strip()
        if not pair:
            continue
        label_str, count_str = pair.split("=")
        counts[int(label_str)] = int(count_str)
    return counts


def parse_classes(raw: str) -> tuple[int, ...]:
    """Parse a `--classes` value like "0,5,8" into sorted digit labels."""
    return tuple(
        sorted(int(label.strip()) for label in raw.split(",") if label.strip())
    )


def download_mnist(data_dir: str) -> MNIST:
    """Download (if needed) and return the full MNIST training split."""
    return MNIST(root=data_dir, train=True, transform=None, download=True)


def _random_indices_per_class(
    labels: np.ndarray, class_counts: dict[int, int], seed: int
) -> np.ndarray:
    """Uniform random selection of `class_counts[label]` indices per class."""
    rng = np.random.default_rng(seed)
    selected = []
    ordered_class_counts = sorted(class_counts.items())
    for label, count in ordered_class_counts:
        class_indices = np.flatnonzero(labels == label)
        if count > len(class_indices):
            raise ValueError(
                f"Requested {count} images for digit {label} but only "
                f"{len(class_indices)} are available."
            )
        chosen_indices = rng.choice(class_indices, size=count, replace=False)
        selected.append(chosen_indices)
    return np.concatenate(selected)


def _prototype_indices_per_class(
    images: np.ndarray,
    labels: np.ndarray,
    class_counts: dict[int, int],
    seed: int,
    pca_components: int,
) -> np.ndarray:
    """Select representative "prototype" images per class via PCA + K-Means.

    For each class, its images are projected onto their top
    `pca_components` principal components, then clustered into `count`
    K-Means clusters (one cluster per image we need to keep). The single
    image closest to each cluster centroid is kept. This biases selection
    toward images that are representative of a distinct mode of
    handwriting for that digit, rather than a uniform random draw, which
    can just as easily include noisy or atypical outliers.
    """
    selected = []
    for label, count in sorted(class_counts.items()):
        class_indices = np.flatnonzero(labels == label)
        if count > len(class_indices):
            raise ValueError(
                f"Requested {count} images for digit {label} but only "
                f"{len(class_indices)} are available."
            )

        class_images = images[class_indices]
        # Flatten each 28x28 image into a single row of 784 pixels, since
        # PCA operates on a 2D (num_images, num_features) array.
        flattened = class_images.reshape(len(class_indices), -1)
        flattened = flattened.astype(np.float64)

        n_components = min(
            pca_components, flattened.shape[0], flattened.shape[1]
        )
        pca = PCA(n_components=n_components, random_state=seed)
        reduced = pca.fit_transform(flattened)

        kmeans = KMeans(n_clusters=count, random_state=seed, n_init=10)
        cluster_assignments = kmeans.fit_predict(reduced)

        chosen_local = np.empty(count, dtype=int)
        for cluster_id in range(count):
            members = np.flatnonzero(cluster_assignments == cluster_id)
            centroid = kmeans.cluster_centers_[cluster_id]
            member_positions = reduced[members]
            distances = np.linalg.norm(member_positions - centroid, axis=1)
            closest_member = np.argmin(distances)
            chosen_local[cluster_id] = members[closest_member]

        selected.append(class_indices[chosen_local])
    return np.concatenate(selected)


def curate_indices(
    dataset: MNIST,
    class_counts: dict[int, int],
    seed: int,
    selection_strategy: SelectionStrategy,
    pca_components: int,
) -> np.ndarray:
    """Select the requested number of images per class using the given
    strategy."""
    labels = dataset.targets.numpy()
    if selection_strategy is SelectionStrategy.PROTOTYPE:
        images = dataset.data.numpy()
        return _prototype_indices_per_class(
            images, labels, class_counts, seed, pca_components
        )
    return _random_indices_per_class(labels, class_counts, seed)


# chance each optional perturbation below is applied at all
_AUGMENTATION_TOGGLE_PROBABILITY = 0.5


def augment_image(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Apply a small, randomized set of perturbations to a single 28x28 digit.

    Rotation, translation, and scale are always applied, each within a
    small range. Shear, stroke thickness, brightness, and pixel noise are
    each independently applied about half the time
    (`_AUGMENTATION_TOGGLE_PROBABILITY`), so augmented images vary in
    *which* extra perturbations show up, not just their magnitude. Every
    perturbation is small enough to keep the digit's label unchanged.
    """
    angle = float(rng.uniform(-12, 12))
    translate = (float(rng.integers(-2, 3)), float(rng.integers(-2, 3)))
    scale = float(rng.uniform(0.92, 1.08))
    shear = (
        float(rng.uniform(-8, 8))
        if rng.random() < _AUGMENTATION_TOGGLE_PROBABILITY
        else 0.0
    )

    pil_image = Image.fromarray(image)
    pil_image = TF.affine(
        pil_image,
        angle=angle,
        translate=translate,
        scale=scale,
        shear=shear,
        fill=0,
    )

    if rng.random() < _AUGMENTATION_TOGGLE_PROBABILITY:
        thicken = rng.random() < 0.5
        filter_to_apply = (
            ImageFilter.MaxFilter(3) if thicken else ImageFilter.MinFilter(3)
        )
        pil_image = pil_image.filter(filter_to_apply)

    augmented = np.array(pil_image)
    augmented = augmented.astype(np.float32)

    if rng.random() < _AUGMENTATION_TOGGLE_PROBABILITY:
        brightness_factor = float(rng.uniform(0.85, 1.15))
        augmented = augmented * brightness_factor

    if rng.random() < _AUGMENTATION_TOGGLE_PROBABILITY:
        noise_std = float(rng.uniform(5, 15))
        noise = rng.normal(scale=noise_std, size=augmented.shape)
        augmented = augmented + noise

    return np.clip(augmented, 0, 255).astype(np.uint8)


def balance_with_augmentation(
    images: np.ndarray,
    labels: np.ndarray,
    seed: int,
    target: BalanceTarget | int = BalanceTarget.MAX,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Oversample under-represented classes with randomized augmentations.

    Every class below the target count is topped up by repeatedly
    sampling (with replacement) one of its existing images and applying
    `augment_image` to it, until it reaches the target. Classes already at
    or above the target are left untouched.

    Returns (images, labels, is_synthetic) where `is_synthetic` is a
    boolean mask marking which rows were generated by this function.
    """
    rng = np.random.default_rng(seed)
    unique_labels, counts = np.unique(labels, return_counts=True)
    target_count = (
        int(counts.max()) if target == BalanceTarget.MAX else int(target)
    )

    extra_images = []
    extra_labels = []
    for label, count in zip(unique_labels, counts):
        deficit = target_count - int(count)
        if deficit <= 0:
            continue
        class_images = images[labels == label]
        source_indices = rng.integers(0, len(class_images), size=deficit)
        for source_index in source_indices:
            extra_images.append(augment_image(class_images[source_index], rng))
            extra_labels.append(label)

    if not extra_images:
        is_synthetic = np.zeros(len(images), dtype=bool)
        return images, labels, is_synthetic

    extra_images_array = np.stack(extra_images)
    combined_images = np.concatenate([images, extra_images_array])

    extra_labels_array = np.array(extra_labels, dtype=labels.dtype)
    combined_labels = np.concatenate([labels, extra_labels_array])

    original_flags = np.zeros(len(images), dtype=bool)
    extra_flags = np.ones(len(extra_images), dtype=bool)
    is_synthetic = np.concatenate([original_flags, extra_flags])

    return combined_images, combined_labels, is_synthetic


class DatasetSplits:
    """Container for the curated train/val/test image and label arrays."""

    def __init__(
        self,
        train_images: np.ndarray,
        train_labels: np.ndarray,
        val_images: np.ndarray,
        val_labels: np.ndarray,
        test_images: np.ndarray,
        test_labels: np.ndarray,
    ):
        self.train_images = train_images
        self.train_labels = train_labels
        self.val_images = val_images
        self.val_labels = val_labels
        self.test_images = test_images
        self.test_labels = test_labels


def prepare_datasets(
    data_dir: str,
    class_counts: dict[int, int],
    seed: int,
    selection_strategy: SelectionStrategy,
    pca_components: int,
    test_split: float,
    val_split: float,
    dataset: MNIST | None = None,
) -> DatasetSplits:
    """Run the reproducible curation -> split pipeline (no balancing).

    1. Select `class_counts` images per class from raw MNIST using
       `selection_strategy`.
    2. Hold out `test_split` of the curated set for evaluation, stratified
       so the held-out set keeps the same class imbalance.
    3. Split the remainder into train/val (`val_split` fraction going to
       val), stratified.

    Every step is seeded from `seed`, so calling this twice with the same
    arguments and dataset yields identical splits. Balancing minority
    classes is the caller's responsibility (see `balance_with_augmentation`)
    and should only ever be applied to the returned training split.
    """
    if dataset is None:
        dataset = download_mnist(data_dir)

    indices = curate_indices(
        dataset, class_counts, seed, selection_strategy, pca_components
    )
    images = dataset.data.numpy()[indices]
    labels = dataset.targets.numpy()[indices]

    train_val_images, test_images, train_val_labels, test_labels = (
        train_test_split(
            images,
            labels,
            test_size=test_split,
            random_state=seed,
            stratify=labels,
        )
    )
    train_images, val_images, train_labels, val_labels = train_test_split(
        train_val_images,
        train_val_labels,
        test_size=val_split,
        random_state=seed,
        stratify=train_val_labels,
    )

    return DatasetSplits(
        train_images=train_images,
        train_labels=train_labels,
        val_images=val_images,
        val_labels=val_labels,
        test_images=test_images,
        test_labels=test_labels,
    )


class CuratedDigitsDataset(Dataset):
    """A torch Dataset over curated digit images, mapped to class indices.

    Labels are remapped from raw digit values (e.g. 0/5/8) to contiguous
    class indices (0/1/2, in sorted digit order) for use with
    `nn.CrossEntropyLoss`.
    """

    def __init__(
        self, images: np.ndarray, labels: np.ndarray, classes: tuple[int, ...]
    ):
        self.images = images
        self.label_to_index = {
            label: i for i, label in enumerate(sorted(classes))
        }
        self.labels = np.array(
            [self.label_to_index[int(label)] for label in labels],
            dtype=np.int64,
        )

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image = self.images[index]
        image = image.astype(np.float32)
        image = image / 255.0  # pixel values from 0-255 to 0.0-1.0

        image_tensor = torch.from_numpy(image)
        # add the channel dimension: (1, 28, 28)
        image_tensor = image_tensor.unsqueeze(0)

        label = self.labels[index]
        return image_tensor, int(label)
