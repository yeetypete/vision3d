"""Shared types for :mod:`vision3d.metrics`."""

from typing import NotRequired, TypedDict

from torch import Tensor

from vision3d.tensors import BoundingBoxes3D


class Prediction3D(TypedDict):
    """Per-frame detection output.

    Attributes:
        boxes: ``[N, K]`` predicted 3D bounding boxes; ``K`` depends on
            the box format.
        scores: ``[N]`` confidence scores.
        labels: ``[N]`` integer class labels.
        velocities: Optional ``[N, 2]`` ground-plane (xy) velocities in
            m/s. Required by :class:`~vision3d.metrics.NuScenesDetectionScore`
            to compute the mean Average Velocity Error (AVE). Defaults to
            zeros when omitted.
        attributes: Optional ``[N]`` integer attribute labels. Used by
            :class:`~vision3d.metrics.NuScenesDetectionScore` to compute the
            mean Average Attribute Error (AAE). A negative value marks
            "no attribute". Defaults to ``-1`` when omitted.
    """

    boxes: BoundingBoxes3D
    scores: Tensor
    labels: Tensor
    velocities: NotRequired[Tensor]
    attributes: NotRequired[Tensor]


class Target3D(TypedDict):
    """Per-frame ground-truth annotations.

    Attributes:
        boxes: ``[M, K]`` ground-truth 3D bounding boxes.
        labels: ``[M]`` integer class labels.
        velocities: Optional ``[M, 2]`` ground-plane (xy) velocities in
            m/s. See :class:`Prediction3D`.
        attributes: Optional ``[M]`` integer attribute labels. A negative
            value marks "no attribute", in which case the box is ignored
            when computing the attribute error. See :class:`Prediction3D`.
    """

    boxes: BoundingBoxes3D
    labels: Tensor
    velocities: NotRequired[Tensor]
    attributes: NotRequired[Tensor]
