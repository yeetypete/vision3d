"""Shared type contracts for vision3d metrics."""

from typing import TYPE_CHECKING, TypedDict

if TYPE_CHECKING:
    from torch import Tensor

    from vision3d.tensors import BoundingBoxes3D


class Prediction3D[N](TypedDict):
    """Per-frame detection output.

    Attributes:
        boxes: ``[N, K]`` predicted 3D bounding boxes; ``K`` depends on
            the box format.
        scores: ``[N]`` confidence scores.
        labels: ``[N]`` integer class labels.
    """

    boxes: BoundingBoxes3D
    scores: Tensor[N]
    labels: Tensor[N]


class Target3D[M](TypedDict):
    """Per-frame ground-truth annotations.

    Attributes:
        boxes: ``[M, K]`` ground-truth 3D bounding boxes.
        labels: ``[M]`` integer class labels.
    """

    boxes: BoundingBoxes3D
    labels: Tensor[M]
