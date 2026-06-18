"""
Visualizing predictions with vision3d
=====================================

This example demonstrates overlaying detector predictions on ground truth
with :func:`vision3d.viz.log_sample` and :func:`vision3d.viz.log_boxes_3d`.
Ground truth and predictions are logged to separate `Rerun
<https://rerun.io/>`_ entities so they can be toggled independently: both
keep per-class colors, ground truth is drawn as translucent colored boxes
and predictions as a wireframe with their confidence score in the label.

The visualization is dataset-agnostic, so any vision3d dataset works here;
we use :class:`~vision3d.datasets.NuScenes3D` to source real frames.
"""

# %%
# Load a frame
# ------------
# Prediction visualization only needs a
# :class:`~vision3d.datasets.SampleInputs` /
# :class:`~vision3d.datasets.SampleTargets` pair to overlay on. See the
# :ref:`dataset examples <sphx_glr_auto_examples_datasets>` for the full
# loading and batching pipeline.

from pathlib import Path

from vision3d.datasets import NuScenes3D

NUSCENES_ROOT = Path("~/.cache/vision3d/nuscenes-mini").expanduser()

dataset = NuScenes3D(NUSCENES_ROOT, version="v1.0-mini", split="train", download=True)

# %%
# Synthesize predictions
# ----------------------
# A real detector would produce these, but to keep the example
# self-contained we synthesize a :class:`~vision3d.metrics.Prediction3D`
# from each frame's targets: drop a few objects, jitter the boxes, and
# assign random confidence scores.

import torch

from vision3d.datasets import SampleTargets
from vision3d.metrics import Prediction3D
from vision3d.tensors import BoundingBoxes3D


def fake_predictions(
    targets: SampleTargets, *, generator: torch.Generator
) -> Prediction3D:
    """Perturb ground-truth boxes into plausible predictions.

    Returns:
        A :class:`~vision3d.metrics.Prediction3D` derived from ``targets``.
    """
    boxes = targets["boxes"]
    labels = targets["labels"]
    n = boxes.shape[0]

    # Keep ~80% of the objects as detections.
    keep = torch.rand(n, generator=generator) < 0.8
    raw = boxes.as_subclass(torch.Tensor)[keep].clone()
    labels = labels[keep]

    # Jitter centers (+/- 0.5 m) and sizes (+/- 10%).
    raw[:, :3] += torch.empty_like(raw[:, :3]).uniform_(-0.5, 0.5, generator=generator)
    raw[:, 3:6] *= 1 + torch.empty_like(raw[:, 3:6]).uniform_(
        -0.1, 0.1, generator=generator
    )

    scores = torch.empty(len(raw)).uniform_(0.3, 0.99, generator=generator)
    return Prediction3D(
        boxes=BoundingBoxes3D(raw, format=boxes.format),
        scores=scores,
        labels=labels,
    )


# %%
# Overlay predictions with :func:`vision3d.viz.log_sample`
# --------------------------------------------------------
# Passing ``predictions`` overlays detections alongside the ground truth.
# ``label_to_id`` keeps per-class colors consistent across frames, and
# ``score_threshold`` drops low-confidence detections before logging.

import rerun as rr
import rerun.blueprint as rrb

from vision3d.viz import fusion_layout, log_sample

rr.init("vision3d_predictions", spawn=True)
rr.send_blueprint(
    rrb.Blueprint(
        fusion_layout(NuScenes3D.camera_names, NuScenes3D.camera_grid),
        rrb.TimePanel(state="collapsed"),
    )
)
rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

generator = torch.Generator().manual_seed(0)
for frame_idx in range(10):
    f_inputs, f_targets = dataset[frame_idx]
    f_preds = fake_predictions(f_targets, generator=generator)
    rr.set_time("frame", sequence=frame_idx)
    log_sample(
        f_inputs,
        f_targets,
        predictions=f_preds,
        label_to_id=dataset.class_to_idx,
        score_threshold=0.4,
        jpeg_quality=75,
    )

# %%
# Logging boxes directly with :func:`vision3d.viz.log_boxes_3d`
# -------------------------------------------------------------
# :func:`~vision3d.viz.log_sample` is a convenience wrapper; for finer
# control you can drive :func:`vision3d.viz.log_boxes_3d` yourself. It
# accepts ``scores``, a ``score_threshold``, and a ``fill_mode`` so you can
# style ground truth and predictions however you like. Here we log the same
# predictions to a standalone entity as a dense wireframe with a stricter
# threshold.

from vision3d.viz import log_boxes_3d

f_inputs, f_targets = dataset[0]
f_preds = fake_predictions(f_targets, generator=generator)

rr.set_time("frame", sequence=0)
log_boxes_3d(
    "world/pred_strict/boxes",
    f_preds["boxes"],
    class_ids=f_preds["labels"].tolist(),
    label_to_id=dataset.class_to_idx,
    scores=f_preds["scores"],
    score_threshold=0.7,
    fill_mode="densewireframe",
    show_labels=True,
)
