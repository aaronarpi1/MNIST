"""Tests for `DigitClassifier`."""

import pytest
import torch
import torch.nn.functional as F
from digit_classification.model import DigitClassifier, FocalLoss


def test_forward_returns_logits_for_each_class() -> None:
    """forward() returns one logit per class for each image in the batch."""
    model = DigitClassifier(num_classes=3)
    logits = model(torch.randn(4, 1, 28, 28))
    assert logits.shape == (4, 3)


def test_predict_step_returns_a_valid_probability_distribution() -> None:
    """predict_step returns per-class probabilities that sum to 1."""
    model = DigitClassifier(num_classes=3)
    probabilities = model.predict_step(
        (torch.randn(4, 1, 28, 28), torch.zeros(4, dtype=torch.long))
    )
    prob_sum = probabilities.sum(dim=1)
    assert torch.allclose(prob_sum, torch.ones(4), atol=1e-5)


def test_training_step_returns_a_scalar_loss() -> None:
    """training_step returns a single scalar loss value."""
    model = DigitClassifier(num_classes=3)
    batch = (torch.randn(4, 1, 28, 28), torch.tensor([0, 1, 2, 0]))
    loss = model.training_step(batch)
    assert loss.ndim == 0


def test_untrained_model_predictions_average_to_uniform_across_seeds() -> None:
    """A freshly initialized (untrained) model shouldn't systematically
    favor any one class over the others. Averaging predicted probabilities
    over many independent random initializations should land close to
    1/3 per class for this 3-class problem -- a large, persistent skew
    here would point to a bug in the architecture or its initialization,
    not just normal per-model randomness.
    """
    num_seeds = 50
    images_per_seed = 16
    num_classes = 3

    # fixed inputs, independent of the seeds under test
    torch.manual_seed(12345)
    fixed_images = torch.rand(images_per_seed, 1, 28, 28)

    probability_totals = torch.zeros(num_classes)
    total_predictions = 0
    for seed in range(num_seeds):
        torch.manual_seed(seed)
        model = DigitClassifier(num_classes=num_classes)
        model.eval()

        with torch.no_grad():
            logits = model.forward(fixed_images)
            probabilities = torch.softmax(logits, dim=1)

        probability_totals += probabilities.sum(dim=0)
        total_predictions += probabilities.shape[0]

    average_probabilities = probability_totals / total_predictions
    expected = torch.full((num_classes,), 1 / num_classes)
    assert torch.allclose(average_probabilities, expected, atol=0.05)


def test_focal_loss_down_weights_confident_predictions_more() -> None:
    """Focal loss should shrink the loss for a confidently-correct
    ("easy") prediction more, relative to plain cross-entropy, than it
    shrinks the loss for a less confident one -- that relative
    down-weighting of easy examples is the entire point of focal loss.
    """
    gamma = 2.0
    focal_loss = FocalLoss(alpha=None, gamma=gamma)
    target = torch.tensor([0])

    confident_logits = torch.tensor([[10.0, 0.0, 0.0]])
    unsure_logits = torch.tensor([[1.0, 0.0, 0.0]])

    confident_cross_entropy = F.cross_entropy(confident_logits, target)
    unsure_cross_entropy = F.cross_entropy(unsure_logits, target)

    confident_focal = focal_loss(confident_logits, target)
    unsure_focal = focal_loss(unsure_logits, target)

    # Fraction of the cross-entropy loss left after focal loss's
    # down-weighting -- smaller means a bigger reduction.
    confident_fraction_kept = confident_focal / confident_cross_entropy
    unsure_fraction_kept = unsure_focal / unsure_cross_entropy
    assert confident_fraction_kept < unsure_fraction_kept

    # The down-weighting factor is exactly (1 - pt) ** gamma, where pt
    # is the model's predicted probability of the true class.
    confident_pt = torch.exp(-confident_cross_entropy)
    unsure_pt = torch.exp(-unsure_cross_entropy)
    assert torch.isclose(
        confident_fraction_kept, (1 - confident_pt) ** gamma, atol=1e-5
    )
    assert torch.isclose(
        unsure_fraction_kept, (1 - unsure_pt) ** gamma, atol=1e-5
    )
