"""One-off script that generated `tests/test_images/mnist/`.

Downloads the real MNIST training set, reproducibly picks a fixed random
sample of images per digit class (via a seeded `np.random.default_rng`),
and saves them as JPEGs checked into the repo -- so the test suite can use
real digit images without any test run ever downloading MNIST itself.

Not part of the test suite or the package; run manually only if the
checked-in test images ever need to be regenerated (e.g. to cover more
digit classes, or more images per class):

    python scripts/generate_test_images.py
"""
import os

import numpy as np
from PIL import Image

from digit_classification.data import download_mnist

FIXTURE_SEED = 20240115  # fixed, so regenerating these files picks the same images
IMAGES_PER_CLASS = 60
CLASSES = (0, 5, 8)
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "tests", "test_images", "mnist")


def main() -> None:
    dataset = download_mnist("/tmp/mnist_fixture_source")
    labels = dataset.targets.numpy()
    images = dataset.data.numpy()

    rng = np.random.default_rng(FIXTURE_SEED)
    for label in CLASSES:
        class_dir = os.path.join(OUTPUT_DIR, str(label))
        os.makedirs(class_dir, exist_ok=True)

        class_indices = np.flatnonzero(labels == label)
        chosen = rng.choice(class_indices, size=IMAGES_PER_CLASS, replace=False)

        for position, index in enumerate(chosen):
            image = Image.fromarray(images[index])
            path = os.path.join(class_dir, f"{position:02d}.jpg")
            image.save(path, quality=95)

        print(f"digit {label}: saved {len(chosen)} images to {class_dir}")


if __name__ == "__main__":
    main()
