"""Helpers for loading processed QuickDraw subsets."""

from __future__ import annotations

import pathlib
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class QuickDrawSplit:
    images: np.ndarray
    labels: np.ndarray
    class_names: tuple[str, ...]

    @property
    def normalized_images(self) -> np.ndarray:
        """Return images as float32 values in [-1, 1]."""
        return self.images.astype(np.float32) / 127.5 - 1.0


def load_quickdraw_npz(path: str | pathlib.Path, split: str = "train") -> QuickDrawSplit:
    """Load a processed QuickDraw `.npz` file.

    Parameters
    ----------
    path:
        Path to a processed QuickDraw subset included with the course.
    split:
        Either `"train"` or `"val"`.
    """
    if split not in {"train", "val"}:
        raise ValueError("split must be 'train' or 'val'")

    data = np.load(path)
    images = data[f"x_{split}"]
    labels = data[f"y_{split}"]
    class_names = tuple(str(name) for name in data["class_names"])
    return QuickDrawSplit(images=images, labels=labels, class_names=class_names)
