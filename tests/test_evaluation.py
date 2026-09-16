"""Tests for the evaluation pipeline plumbing.

`_DummyModel` and the local stub in
`test_load_model_delegates_to_checkpoint_loading` aren't stand-ins for
a missing model -- they're deliberate test doubles
with known, deterministic behavior (e.g. "always predicts class 0"), used
to test `predict_labels`'s aggregation logic and `load_model`'s checkpoint
delegation precisely, independent of what the real (now-implemented)
`DigitClassifier` actually predicts.
"""

import torch
from torch.utils.data import DataLoader

from digit_classification import evaluation
from digit_classification.data import CuratedDigitsDataset, SelectionStrategy


class _DummyModel:
    """Always "predicts" class index 0 with full confidence."""

    def eval(self):
        return self

    def predict_step(self, batch, batch_idx=0, dataloader_idx=0):
        images, _ = batch
        batch_size = images.shape[0]
        probabilities = torch.zeros(batch_size, 3)
        probabilities[:, 0] = 1.0
        return probabilities


def test_predict_labels_matches_dummy_model_output(real_mnist_sample) -> None:
    """predict_labels aggregates a dummy model's per-batch predictions
    correctly."""
    dataset = CuratedDigitsDataset(
        real_mnist_sample.data.numpy()[:9],
        real_mnist_sample.targets.numpy()[:9],
        classes=(0, 5, 8),
    )
    loader = DataLoader(dataset, batch_size=4, shuffle=False)

    true_labels, predicted_labels = evaluation.predict_labels(
        _DummyModel(), loader
    )

    assert predicted_labels.shape == true_labels.shape
    assert (predicted_labels == 0).all()


def test_evaluate_checkpoint_produces_classification_report(
    monkeypatch, real_mnist_sample
) -> None:
    """evaluate_checkpoint returns a classification_report covering
    every curated class."""
    monkeypatch.setattr(
        evaluation, "load_model", lambda checkpoint_path: _DummyModel()
    )
    monkeypatch.setattr(
        evaluation, "download_mnist", lambda data_dir: real_mnist_sample
    )

    class_counts = {0: 20, 5: 10, 8: 30}
    report = evaluation.evaluate_checkpoint(
        checkpoint_path="unused-checkpoint.ckpt",
        data_dir="unused",
        class_counts=class_counts,
        seed=123,
        selection_strategy=SelectionStrategy.RANDOM,
        pca_components=5,
        test_split=0.2,
        batch_size=64,
    )

    assert "precision" in report
    assert "recall" in report
    # All three curated digit labels should be represented in the report.
    for digit in class_counts:
        assert str(digit) in report


def test_load_model_delegates_to_checkpoint_loading(monkeypatch) -> None:
    """load_model delegates to DigitClassifier.load_from_checkpoint
    with the given path."""
    captured = {}

    class _StubDigitClassifier:
        def load_from_checkpoint(cls, checkpoint_path):
            captured["checkpoint_path"] = checkpoint_path
            return "loaded-model"

        load_from_checkpoint = classmethod(load_from_checkpoint)

    monkeypatch.setattr(evaluation, "DigitClassifier", _StubDigitClassifier)

    result = evaluation.load_model("path/to/checkpoint.ckpt")

    assert result == "loaded-model"
    assert captured["checkpoint_path"] == "path/to/checkpoint.ckpt"
