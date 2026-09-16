"""Tests for data curation, PCA/K-Means prototype selection, reproducible
splitting, and the randomized minority-class augmentation workflow.
"""

import numpy as np
import pytest
import typer

from digit_classification.cli import _validate_classes
from digit_classification.data import (
    CuratedDigitsDataset,
    SelectionStrategy,
    augment_image,
    balance_with_augmentation,
    curate_indices,
    parse_class_counts,
    parse_classes,
    prepare_datasets,
)
from tests.conftest import load_real_mnist_sample


class TestRandomSelection:
    def test_selects_exact_count_per_class(
        self, real_mnist_sample, small_curation_kwargs
    ) -> None:
        """Random selection returns exactly the requested count per class."""
        indices = curate_indices(real_mnist_sample, **small_curation_kwargs)
        labels = real_mnist_sample.targets.numpy()[indices]
        counts = dict(zip(*np.unique(labels, return_counts=True)))
        assert counts == small_curation_kwargs["class_counts"]

    def test_indices_are_unique(
        self, real_mnist_sample, small_curation_kwargs
    ) -> None:
        """Random selection never returns the same index twice."""
        indices = curate_indices(real_mnist_sample, **small_curation_kwargs)
        assert len(indices) == len(set(indices.tolist()))

    def test_reproducible_with_same_seed(
        self, real_mnist_sample, small_curation_kwargs
    ) -> None:
        """Random selection is reproducible given the same seed."""
        first = curate_indices(real_mnist_sample, **small_curation_kwargs)
        second = curate_indices(real_mnist_sample, **small_curation_kwargs)
        np.testing.assert_array_equal(first, second)

    def test_different_seed_changes_selection(
        self, real_mnist_sample, small_curation_kwargs
    ) -> None:
        """A different seed changes which indices get selected."""
        first = curate_indices(real_mnist_sample, **small_curation_kwargs)
        other_kwargs = {
            **small_curation_kwargs,
            "seed": small_curation_kwargs["seed"] + 1,
        }
        second = curate_indices(real_mnist_sample, **other_kwargs)
        assert set(first.tolist()) != set(second.tolist())

    def test_raises_when_requesting_more_than_available(
        self, real_mnist_sample, small_curation_kwargs
    ) -> None:
        """Requesting more images than exist for a class raises ValueError."""
        kwargs = {**small_curation_kwargs, "class_counts": {0: 1000}}
        with pytest.raises(ValueError, match="only"):
            curate_indices(real_mnist_sample, **kwargs)


class TestPrototypeSelection:
    def test_selects_exact_count_per_class(
        self, real_mnist_sample, small_curation_kwargs
    ) -> None:
        """Prototype (PCA + K-Means) selection returns exactly the
        requested count per class."""
        kwargs = {
            **small_curation_kwargs,
            "selection_strategy": SelectionStrategy.PROTOTYPE,
        }
        indices = curate_indices(real_mnist_sample, **kwargs)
        labels = real_mnist_sample.targets.numpy()[indices]
        counts = dict(zip(*np.unique(labels, return_counts=True)))
        assert counts == kwargs["class_counts"]

    def test_indices_are_unique(
        self, real_mnist_sample, small_curation_kwargs
    ) -> None:
        """Prototype selection never returns the same index twice."""
        kwargs = {
            **small_curation_kwargs,
            "selection_strategy": SelectionStrategy.PROTOTYPE,
        }
        indices = curate_indices(real_mnist_sample, **kwargs)
        assert len(indices) == len(set(indices.tolist()))

    def test_reproducible_with_same_seed(
        self, real_mnist_sample, small_curation_kwargs
    ) -> None:
        """Prototype selection is reproducible given the same seed."""
        kwargs = {
            **small_curation_kwargs,
            "selection_strategy": SelectionStrategy.PROTOTYPE,
        }
        first = curate_indices(real_mnist_sample, **kwargs)
        second = curate_indices(real_mnist_sample, **kwargs)
        np.testing.assert_array_equal(first, second)

    def test_raises_when_requesting_more_than_available(
        self, real_mnist_sample, small_curation_kwargs
    ) -> None:
        """Requesting more prototype images than exist for a class
        raises ValueError."""
        kwargs = {
            **small_curation_kwargs,
            "selection_strategy": SelectionStrategy.PROTOTYPE,
            "class_counts": {0: 1000},
        }
        with pytest.raises(ValueError, match="only"):
            curate_indices(real_mnist_sample, **kwargs)


class TestPrepareDatasets:
    def test_split_sizes_match_request(
        self, real_mnist_sample, small_curation_kwargs
    ) -> None:
        """The curated images are fully accounted for across the
        train/val/test splits."""
        splits = prepare_datasets(
            data_dir="unused",
            test_split=0.2,
            val_split=0.1,
            dataset=real_mnist_sample,
            **small_curation_kwargs
        )
        total_curated = sum(small_curation_kwargs["class_counts"].values())
        total_returned = (
            len(splits.train_labels)
            + len(splits.val_labels)
            + len(splits.test_labels)
        )
        assert total_returned == total_curated

        expected_test = round(total_curated * 0.2)
        assert abs(len(splits.test_labels) - expected_test) <= 1

    def test_test_split_is_stratified(
        self, real_mnist_sample, small_curation_kwargs
    ) -> None:
        """Verifies that the test split contains at least one example
        of each curated class."""
        splits = prepare_datasets(
            data_dir="unused",
            test_split=0.2,
            val_split=0.1,
            dataset=real_mnist_sample,
            **small_curation_kwargs
        )
        test_counts = dict(
            zip(*np.unique(splits.test_labels, return_counts=True))
        )
        # Every curated class should appear at least once in a
        # stratified split.
        assert set(test_counts) == set(small_curation_kwargs["class_counts"])

    def test_reproducible_with_same_seed(
        self, real_mnist_sample, small_curation_kwargs
    ) -> None:
        """Verifies reproducibility: two calls to `prepare_datasets`
        with the same seed should produce identical results."""

        first = prepare_datasets(
            data_dir="unused",
            test_split=0.2,
            val_split=0.1,
            dataset=real_mnist_sample,
            **small_curation_kwargs
        )
        second = prepare_datasets(
            data_dir="unused",
            test_split=0.2,
            val_split=0.1,
            dataset=real_mnist_sample,
            **small_curation_kwargs
        )
        np.testing.assert_array_equal(first.train_labels, second.train_labels)
        np.testing.assert_array_equal(first.test_labels, second.test_labels)
        np.testing.assert_array_equal(first.train_images, second.train_images)

    def test_no_overlap_between_splits(
        self, real_mnist_sample, small_curation_kwargs
    ) -> None:
        """Verifies that the train/val/test splits are disjoint sets."""
        splits = prepare_datasets(
            data_dir="unused",
            test_split=0.2,
            val_split=0.1,
            dataset=real_mnist_sample,
            **small_curation_kwargs
        )
        train_set = {tuple(img.flatten()) for img in splits.train_images}
        val_set = {tuple(img.flatten()) for img in splits.val_images}
        test_set = {tuple(img.flatten()) for img in splits.test_images}
        assert train_set.isdisjoint(val_set)
        assert train_set.isdisjoint(test_set)
        assert val_set.isdisjoint(test_set)


class TestParsers:
    def test_parse_class_counts(self) -> None:
        """parse_class_counts turns a label=count,... string into a dict."""
        assert parse_class_counts("0=1200,5=300,8=3500") == {
            0: 1200,
            5: 300,
            8: 3500,
        }

    def test_parse_class_counts_raises_for_invalid_input(self) -> None:
        """A malformed label=count pair raises ValueError."""
        # `parse_class_counts` itself only rejects malformed label=count
        # pairs -- a trailing comma is intentionally tolerated (see
        # test_parse_class_counts_ignores_blank_segments below), and digit
        # range / positive-count checks live in the CLI's
        # `_validate_class_counts`, not here (see test_cli.py).
        with pytest.raises(ValueError):
            # no "=" in the last pair
            parse_class_counts("0=1200,5=300,8>3500")

    def test_parse_class_counts_ignores_blank_segments(self) -> None:
        """Blank/trailing comma segments are silently skipped."""
        assert parse_class_counts("0=1,,5=2,") == {0: 1, 5: 2}

    def test_parse_classes_returns_sorted_tuple(self) -> None:
        """parse_classes returns the parsed digits sorted, regardless
        of input order."""
        assert parse_classes("8,0,5") == (0, 5, 8)

    def test_parse_classes_ignores_blank_segments(self) -> None:
        """Blank/trailing comma segments are silently skipped."""
        assert parse_classes("1,,2,") == (1, 2)

    def test_parse_classes_raises_for_non_integer(self) -> None:
        """A non-integer entry raises ValueError."""
        with pytest.raises(ValueError):
            parse_classes("1,2,a")

    def test_classes_validation_rejects_negative_integer(self) -> None:
        """_validate_classes rejects a negative digit label."""
        # `parse_classes` itself has no range check (it happily parses
        # negative numbers) -- rejecting an out-of-digit-range value is a
        # CLI-boundary concern, enforced by `_validate_classes` instead.
        with pytest.raises(typer.BadParameter):
            _validate_classes("1,-2,3")


class TestAugmentImage:
    def test_preserves_shape_and_dtype(self) -> None:
        """augment_image always returns a 28x28 uint8 array."""
        rng = np.random.default_rng(0)
        original = np.zeros((28, 28), dtype=np.uint8)
        original[10:18, 10:18] = 255
        augmented = augment_image(original, rng)
        assert augmented.shape == (28, 28)
        assert augmented.dtype == np.uint8

    def test_randomized_across_calls(self) -> None:
        """Two calls with the same rng produce different augmented images."""
        rng = np.random.default_rng(0)
        original = np.zeros((28, 28), dtype=np.uint8)
        original[10:18, 10:18] = 255
        first = augment_image(original, rng)
        second = augment_image(original, rng)
        assert not np.array_equal(first, second)

    def test_stays_in_valid_pixel_range_across_many_seeds(self) -> None:
        """Every combination of the optional perturbations stays
        within [0, 255]."""
        # Exercises every combination of the independently-toggled
        # perturbations (shear, stroke thickness, brightness, noise)
        # without crashing or producing out-of-range pixel values.
        original = np.zeros((28, 28), dtype=np.uint8)
        original[10:18, 10:18] = 255
        for seed in range(50):
            augmented = augment_image(original, np.random.default_rng(seed))
            assert augmented.shape == (28, 28)
            assert augmented.dtype == np.uint8
            assert augmented.min() >= 0 and augmented.max() <= 255

    def test_produces_a_variety_of_outputs_across_seeds(self) -> None:
        """Many seeds produce genuinely varied output, not a handful
        of repeats."""
        # With four independent 50/50 toggles plus continuous rotation/
        # translation/scale, 50 seeds should not collapse to a handful of
        # identical images.
        original = np.zeros((28, 28), dtype=np.uint8)
        original[10:18, 10:18] = 255
        outputs = {
            augment_image(original, np.random.default_rng(seed)).tobytes()
            for seed in range(50)
        }
        assert len(outputs) > 40

    def test_augmented_five_stays_closer_to_its_own_original_than_to_eights(
        self,
    ) -> None:
        """Label-preservation sanity check: after augmenting a '5', its L1
        (pixel-wise absolute difference) distance to its own un-augmented
        original should still be smaller than its distance to any '8' --
        i.e. augmentation shouldn't distort a digit into resembling a
        different one.
        """
        rng = np.random.default_rng(0)

        def make_digit_image(block_row, block_col):
            # A bright 10x10 block plus per-image noise, standing in for a
            # digit's ink -- block position is what distinguishes "5" from
            # "8" here, kept far apart so the two never overlap.
            base = np.zeros((28, 28), dtype=np.float64)
            base[
                block_row: block_row + 10, block_col: block_col + 10
            ] = 200.0
            noisy = base + rng.normal(scale=15.0, size=(28, 28))
            return np.clip(noisy, 0, 255).astype(np.uint8)

        def l1_distance(image_a, image_b):
            return np.abs(
                image_a.astype(np.int32) - image_b.astype(np.int32)
            ).sum()

        fives = [make_digit_image(2, 2) for _ in range(10)]
        eights = [make_digit_image(18, 18) for _ in range(5)]

        for five in fives:
            augmented_five = augment_image(five, rng)

            distance_to_own_original = l1_distance(augmented_five, five)
            distances_to_eights = [
                l1_distance(augmented_five, eight) for eight in eights
            ]

            assert distance_to_own_original < min(distances_to_eights)


class TestBalanceWithAugmentation:
    def test_matches_majority_class_by_default(self) -> None:
        """Balancing with the default target tops every class up to
        match the largest one."""
        images = np.zeros((30, 28, 28), dtype=np.uint8)
        labels = np.array([0] * 20 + [5] * 5 + [8] * 5)
        combined_images, combined_labels, is_synthetic = (
            balance_with_augmentation(images, labels, seed=1)
        )
        counts = dict(zip(*np.unique(combined_labels, return_counts=True)))
        assert counts == {0: 20, 5: 20, 8: 20}
        assert (
            combined_images.shape[0]
            == len(combined_labels)
            == is_synthetic.shape[0]
        )
        assert is_synthetic.sum() == (20 - 5) + (20 - 5)
        assert not is_synthetic[: len(images)].any()

    def test_explicit_integer_target(self) -> None:
        """An explicit integer target overrides matching the largest class."""
        images = np.zeros((15, 28, 28), dtype=np.uint8)
        labels = np.array([0] * 10 + [5] * 5)
        _, combined_labels, is_synthetic = balance_with_augmentation(
            images, labels, seed=1, target=12
        )
        counts = dict(zip(*np.unique(combined_labels, return_counts=True)))
        assert counts == {0: 12, 5: 12}
        assert is_synthetic.sum() == (12 - 10) + (12 - 5)

    def test_noop_when_already_balanced(self) -> None:
        """Balancing a set where every class is already equal in size
        changes nothing."""
        images = np.zeros((9, 28, 28), dtype=np.uint8)
        labels = np.array([0, 0, 0, 5, 5, 5, 8, 8, 8])
        combined_images, combined_labels, is_synthetic = (
            balance_with_augmentation(images, labels, seed=1)
        )
        assert combined_images.shape[0] == 9
        assert not is_synthetic.any()

    def test_reproducible_with_same_seed(self) -> None:
        """Balancing is reproducible given the same seed."""
        images = np.array(
            [np.full((28, 28), i, dtype=np.uint8) for i in range(10)]
            + [np.zeros((28, 28), dtype=np.uint8)] * 2
        )
        labels = np.array([0] * 10 + [5] * 2)
        first = balance_with_augmentation(images, labels, seed=7)
        second = balance_with_augmentation(images, labels, seed=7)
        np.testing.assert_array_equal(first[0], second[0])
        np.testing.assert_array_equal(first[1], second[1])


class TestCuratedDigitsDataset:
    def test_length_and_item_shape(self) -> None:
        """Each item is a normalized (1, 28, 28) tensor paired with an
        int label."""
        images = np.zeros((5, 28, 28), dtype=np.uint8)
        labels = np.array([0, 5, 8, 0, 5])
        dataset = CuratedDigitsDataset(images, labels, classes=(0, 5, 8))
        assert len(dataset) == 5
        image_tensor, label_index = dataset[0]
        assert image_tensor.shape == (1, 28, 28)
        assert image_tensor.dtype.is_floating_point
        assert image_tensor.max() <= 1.0 and image_tensor.min() >= 0.0
        assert isinstance(label_index, int)

    def test_label_mapping_is_sorted_digit_order(self) -> None:
        """Raw digit labels map to class indices in sorted digit
        order, regardless of input order."""
        images = np.zeros((3, 28, 28), dtype=np.uint8)
        labels = np.array([8, 0, 5])
        dataset = CuratedDigitsDataset(images, labels, classes=(8, 0, 5))
        # Regardless of the order `classes` is given in, mapping
        # follows sorted order.
        assert dataset.label_to_index == {0: 0, 5: 1, 8: 2}
        _, label_for_digit_8 = dataset[0]
        _, label_for_digit_0 = dataset[1]
        _, label_for_digit_5 = dataset[2]
        assert (label_for_digit_8, label_for_digit_0, label_for_digit_5) == (
            2,
            0,
            1,
        )


def test_load_real_mnist_sample_helper_produces_requested_counts() -> None:
    """load_real_mnist_sample returns exactly the requested number of
    images per class."""
    sample = load_real_mnist_sample({0: 3, 5: 2})
    counts = dict(zip(*np.unique(sample.targets.numpy(), return_counts=True)))
    assert counts == {0: 3, 5: 2}
    assert sample.data.shape == (5, 28, 28)
