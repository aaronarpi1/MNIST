"""Digit classifier model definition.

This file implements a PyTorch Lightning `LightningModule` that:
  - Takes a (1, 28, 28) grayscale image as input.
  - Classifies it as one of the curated digits.


`digit_classification.data.CuratedDigitsDataset` maps raw digit labels to
contiguous class indices in sorted order (e.g. for classes (0, 5, 8):
index 0 -> digit 0, index 1 -> digit 5, index 2 -> digit 8).

"""

from __future__ import annotations
import functools
import time

import torch
import torchmetrics
from torch import nn
from lightning.pytorch import LightningModule


def log_execution_time(func):
    """Decorator: time a method call and print total + per-image time.

    Infers batch size from the method's first argument after `self`: a
    tensor's leading dimension, or (for a batch like (inputs, targets))
    the leading dimension of its first element.
    """

    @functools.wraps(func)
    def wrapper(self, batch, *args, **kwargs):
        start = time.time()
        result = func(self, batch, *args, **kwargs)
        elapsed = time.time() - start

        data = batch[0] if isinstance(batch, (tuple, list)) else batch
        batch_size = data.size(0)
        print(
            f"{func.__name__} time on batch of {batch_size}: "
            f"{elapsed:.4f} seconds, "
            f"time per image: {elapsed / batch_size:.6f} seconds"
        )
        return result

    return wrapper


class FocalLoss(nn.Module):
    def __init__(
        self,
        alpha: float | list | torch.Tensor | None = None,
        gamma: float = 2.0,
    ) -> None:
        """
        Custom focal loss implementation for multi-class classification.
        Focal loss is designed to address class imbalance by down-weighting
        easy examples and focusing training on hard negatives. It makes the
        gradient effectively smaller for well-classified examples, allowing
        the model to focus on misclassified or lower-confidence examples.

        It is not a good loss function for balanced datasets.

        Args:
            alpha (float, list, or Tensor): Class weights for imbalance.
            gamma (float): Focusing parameter for hard examples.
        """
        super().__init__()
        self.gamma = gamma

        if isinstance(alpha, (float, int)):
            self.alpha = float(alpha)
        elif isinstance(alpha, list):
            self.alpha = torch.tensor(alpha)
        else:
            self.alpha = alpha

    def forward(
        self, inputs: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        ce_loss = torch.nn.functional.cross_entropy(
            inputs, targets, reduction="none"
        )
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.alpha is not None:
            if isinstance(self.alpha, torch.Tensor):
                if self.alpha.device != inputs.device:
                    self.alpha = self.alpha.to(inputs.device)
                alpha_t = self.alpha[targets]
            else:
                alpha_t = self.alpha
            focal_loss = alpha_t * focal_loss
        return focal_loss.mean()


class DigitClassifier(LightningModule):
    """LightningModule for classifying curated MNIST digits."""

    def __init__(
        self,
        num_classes: int = 3,
        learning_rate: float = 1e-3,
        loss_function: int = 0,
        focal_alpha: float = 0.2,
        focal_gamma: float = 2.0,
    ):
        """Initialize the digit classifier model.
        args:
            num_classes: Number of classes to classify.
            learning_rate: Learning rate for the optimizer.
            loss_function: 0 for CrossEntropyLoss, 1 for FocalLoss.
            focal_alpha: Alpha parameter for FocalLoss.
            focal_gamma: Gamma parameter for FocalLoss.
        """
        super().__init__()
        self.save_hyperparameters()
        self.lr = learning_rate
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(7 * 7 * 32, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )
        if loss_function == 0:
            self.loss_function = nn.CrossEntropyLoss()
        else:
            self.loss_function = FocalLoss(
                alpha=focal_alpha, gamma=focal_gamma
            )
        self.train_acc = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes
        )
        self.val_acc = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes
        )

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    @log_execution_time
    def forward(self, input_batch: torch.Tensor) -> torch.Tensor:
        conv_features = self.features(input_batch)
        class_logits = self.classifier(conv_features)
        return class_logits

    @log_execution_time
    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        inputs, truth = batch
        outputs = self.forward(inputs)
        loss = self.loss_function(outputs, truth)
        predictions = torch.argmax(outputs, dim=1)
        self.train_acc(predictions, truth)
        self.log(
            "Training loss", loss, on_step=True, on_epoch=True, prog_bar=True
        )
        self.log(
            "Training accuracy",
            self.train_acc,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return loss

    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        inputs, truth = batch
        outputs = self.forward(inputs)
        loss = self.loss_function(outputs, truth)
        predictions = torch.argmax(outputs, dim=1)
        self.val_acc(predictions, truth)
        self.log(
            "Validation loss", loss, on_step=True, on_epoch=True, prog_bar=True
        )
        self.log(
            "Validation accuracy",
            self.val_acc,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

    def predict_step(
        self, batch: tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        """Predict the class probabilities (softmax, summing to 1.0)
        for a batch of images."""
        inputs, _ = batch
        outputs = self.forward(inputs)
        probabilities = torch.softmax(outputs, dim=1)
        return probabilities
