"""A small utility for visually sanity-checking the augmentation workflow:
save individual before/after comparison images (original alongside its
randomly augmented counterpart) into a directory, so you can eyeball that
augmentations stay label-preserving.
"""

import os

import numpy as np
from PIL import Image
from torchvision.datasets import MNIST

from digit_classification.data import augment_image

_CELL_SCALE = 4  # upscale each 28x28 digit for easier viewing
_PADDING = 4


def save_augmentation_examples(
    dataset: MNIST,
    digit: int,
    num_examples: int,
    seed: int,
    output_dir: str,
) -> list[str]:
    """Sample `num_examples` images of `digit`, augment each once, and save
    each as its own side-by-side (original | augmented) comparison image
    under `output_dir`. Returns the saved file paths.
    """
    rng = np.random.default_rng(seed)
    labels = dataset.targets.numpy()
    class_indices = np.flatnonzero(labels == digit)
    if num_examples > len(class_indices):
        raise ValueError(
            f"Requested {num_examples} examples for digit {digit} but only "
            f"{len(class_indices)} are available."
        )

    chosen = rng.choice(class_indices, size=num_examples, replace=False)
    all_images = dataset.data.numpy()
    originals = all_images[chosen]

    os.makedirs(output_dir, exist_ok=True)

    cell = 28 * _CELL_SCALE
    pair_width = 2 * cell + 3 * _PADDING
    pair_height = cell + 2 * _PADDING

    saved_paths = []
    for index, original in enumerate(originals):
        augmented = augment_image(original, rng)

        pair = Image.new("L", (pair_width, pair_height), color=255)

        original_image = Image.fromarray(original)
        original_image = original_image.resize((cell, cell), Image.NEAREST)
        pair.paste(original_image, (_PADDING, _PADDING))

        augmented_image = Image.fromarray(augmented)
        augmented_image = augmented_image.resize((cell, cell), Image.NEAREST)
        pair.paste(augmented_image, (2 * _PADDING + cell, _PADDING))

        path = os.path.join(output_dir, f"digit{digit}_example{index:02d}.png")
        pair.save(path)
        saved_paths.append(path)

    return saved_paths
