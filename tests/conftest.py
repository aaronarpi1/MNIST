"""Shared pytest fixtures and helpers for the digit_classification test suite.

Test images live under `tests/test_images/mnist/<digit>/*.jpg` -- 60 real
MNIST training images each of digits 0, 5, and 8, checked into the repo.
They were selected once, reproducibly, with a fixed seed (see
`scripts/generate_test_images.py`) via `np.random.default_rng(seed).choice`
over the real downloaded dataset, then saved as JPEGs. Loading them here
is just reading small local files -- no network access, and no repeated
downloads across test runs or machines.
"""

import os

import numpy as np
import pytest
import torch
from PIL import Image

from digit_classification.data import SelectionStrategy

_TEST_IMAGES_DIR = os.path.join(
    os.path.dirname(__file__), "test_images", "mnist"
)


class RealMNISTSample:
    """A small, real subset of MNIST loaded from `tests/test_images/`.

    Exposes the same `.data` / `.targets` attributes the real torchvision
    dataset exposes, so `digit_classification.data` functions (which only
    touch those two attributes) can be exercised against real digit
    images without downloading the full dataset.
    """

    def __init__(self, data: torch.Tensor, targets: torch.Tensor):
        self.data = data
        self.targets = targets


def load_real_mnist_sample(counts_per_class: dict) -> RealMNISTSample:
    """Load real, checked-in sample images for the requested digit classes.

    `counts_per_class` maps digit label to how many of that digit's 60
    stored images to use (in filename order -- the images themselves were
    already chosen randomly at fixture-creation time, so no further random
    selection is needed here for this to be "random real images").
    """
    images = []
    labels = []
    for label, count in counts_per_class.items():
        class_dir = os.path.join(_TEST_IMAGES_DIR, str(label))
        filenames = sorted(os.listdir(class_dir))[:count]
        if len(filenames) < count:
            raise ValueError(
                f"Requested {count} test images for digit {label} but only "
                f"{len(filenames)} are stored in {class_dir}."
            )
        for filename in filenames:
            image = Image.open(os.path.join(class_dir, filename))
            images.append(np.array(image, dtype=np.uint8))
            labels.append(label)

    data = torch.from_numpy(np.stack(images))
    targets = torch.tensor(labels, dtype=torch.int64)
    return RealMNISTSample(data, targets)


def _real_mnist_sample() -> RealMNISTSample:
    """60 real images each of digits 0, 5, and 8 (all of the stored set)."""
    return load_real_mnist_sample({0: 60, 5: 60, 8: 60})


real_mnist_sample = pytest.fixture(_real_mnist_sample)


def _small_curation_kwargs() -> dict:
    """kwargs for `curate_indices`/`prepare_datasets` that fit within
    `real_mnist_sample`."""
    return dict(
        class_counts={0: 20, 5: 10, 8: 30},
        seed=123,
        selection_strategy=SelectionStrategy.RANDOM,
        pca_components=5,
    )


small_curation_kwargs = pytest.fixture(_small_curation_kwargs)
