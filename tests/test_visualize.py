"""Tests for saving before/after augmentation example images."""

import os

import numpy as np
import pytest
from PIL import Image

from digit_classification.visualize import save_augmentation_examples


def test_saves_one_file_per_example(real_mnist_sample, tmp_path) -> None:
    """save_augmentation_examples writes exactly one file per requested
    example."""
    paths = save_augmentation_examples(
        real_mnist_sample,
        digit=0,
        num_examples=4,
        seed=1,
        output_dir=str(tmp_path),
    )

    assert len(paths) == 4
    for path in paths:
        assert os.path.exists(path)
        assert os.path.dirname(path) == str(tmp_path)


def test_each_file_is_a_side_by_side_before_after_pair(
    real_mnist_sample, tmp_path
) -> None:
    """Each saved file is a single image with the original and
    augmented digit side by side."""
    paths = save_augmentation_examples(
        real_mnist_sample,
        digit=0,
        num_examples=1,
        seed=1,
        output_dir=str(tmp_path),
    )

    cell = 28 * 4  # matches visualize._CELL_SCALE
    padding = 4  # matches visualize._PADDING
    image = Image.open(paths[0])
    assert image.size == (2 * cell + 3 * padding, cell + 2 * padding)
    assert image.mode == "L"


def test_reproducible_with_same_seed(real_mnist_sample, tmp_path) -> None:
    """The same seed produces byte-identical saved images across
    separate calls."""
    first_dir = os.path.join(str(tmp_path), "first")
    second_dir = os.path.join(str(tmp_path), "second")
    first_paths = save_augmentation_examples(
        real_mnist_sample,
        digit=5,
        num_examples=3,
        seed=7,
        output_dir=str(first_dir),
    )
    second_paths = save_augmentation_examples(
        real_mnist_sample,
        digit=5,
        num_examples=3,
        seed=7,
        output_dir=str(second_dir),
    )

    for first_path, second_path in zip(first_paths, second_paths):
        np.testing.assert_array_equal(
            np.array(Image.open(first_path)),
            np.array(Image.open(second_path)),
        )


def test_raises_when_requesting_more_than_available(
    real_mnist_sample, tmp_path
) -> None:
    """Requesting more examples than are available for a digit raises
    ValueError."""
    with pytest.raises(ValueError, match="only"):
        save_augmentation_examples(
            real_mnist_sample,
            digit=0,
            num_examples=1000,
            seed=1,
            output_dir=str(tmp_path),
        )
