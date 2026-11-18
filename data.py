from pathlib import Path
from typing import Tuple

import numpy as np
from scipy.io import loadmat


def load_dataset(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")

    contents = loadmat(path)
    if "X" not in contents or "y" not in contents:
        raise ValueError(f"{path} must contain MATLAB variables 'X' and 'y'.")

    images = np.asarray(contents["X"])
    targets = np.asarray(contents["y"], dtype=np.float32)
    if images.ndim != 4 or images.shape[2] != 1:
        raise ValueError(
            "Expected X with shape (height, width, 1, samples); "
            f"got {images.shape}."
        )
    if targets.ndim != 2 or targets.shape[1] != 2:
        raise ValueError(f"Expected y with two continuous columns; got {targets.shape}.")
    if images.shape[3] != targets.shape[0]:
        raise ValueError(
            f"Image/target count mismatch: {images.shape[3]} images, "
            f"{targets.shape[0]} targets."
        )
    if images.shape[3] == 0:
        raise ValueError("The dataset contains no samples.")
    if not np.isfinite(targets).all():
        raise ValueError("Targets contain NaN or infinite values.")

    images = np.transpose(images, (3, 2, 0, 1)).astype(np.float32)
    if not np.isfinite(images).all():
        raise ValueError("Images contain NaN or infinite values.")
    if images.max() > 1.0:
        images /= 255.0
    if images.min() < 0.0 or images.max() > 1.0:
        raise ValueError("Expected image values in the range [0, 255] or [0, 1].")

    return images, targets


def split_samples(
    images: np.ndarray, targets: np.ndarray, validation_fraction: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("Validation fraction must be between 0 and 1.")
    split_index = int(len(images) * (1.0 - validation_fraction))
    if split_index < 1 or split_index >= len(images):
        raise ValueError("Not enough samples for the requested train/validation split.")
    return (
        images[:split_index],
        targets[:split_index],
        images[split_index:],
        targets[split_index:],
    )
