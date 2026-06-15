"""
Using nuScenes with vision3d
==============================

This example demonstrates using the nuScenes dataset (mini-split) with
:class:`vision3d.datasets.NuScenes3D`. It covers inspecting the
:class:`~vision3d.datasets.SampleInputs`,
:class:`~vision3d.datasets.SampleTargets` tuple returned by the dataset,
batching with :func:`vision3d.datasets.collate_fn` for training, and
visualizing a frame with :func:`vision3d.viz.log_sample`.
"""

# %%
# Construct the dataset
# ---------------------
# :class:`~vision3d.datasets.NuScenes3D` yields sample frames describing
# the 3D scene. Each sample carries lidar points, all six camera images,
# their intrinsics and extrinsics, and 3D bounding-box annotations of the
# objects in the scene.

from pathlib import Path

from vision3d.datasets import NuScenes3D

NUSCENES_ROOT = Path("~/.cache/vision3d/nuscenes-mini").expanduser()

dataset = NuScenes3D(NUSCENES_ROOT, version="v1.0-mini", split="train", download=True)
print(f"len(dataset) = {len(dataset)}")
print(f"classes ({len(dataset.classes)}): {dataset.classes}")

# %%
# Inspect a sample
# ----------------
# A single index returns a ``(inputs, targets)`` tuple where ``inputs``
# is a :class:`~vision3d.datasets.FusionInputs` dict and ``targets``
# is a :class:`~vision3d.datasets.SampleTargets` dict. Most values are
# semantic tensor types from :mod:`vision3d.tensors`
# (:class:`~vision3d.tensors.PointCloud3D`,
# :class:`~vision3d.tensors.CameraImages`,
# :class:`~vision3d.tensors.BoundingBoxes3D`, ...) so
# :mod:`vision3d.transforms` can dispatch to the right operation per
# input.

inputs, targets = dataset[0]

print("inputs:")
print(
    f"  points: type={type(inputs['points']).__name__} "
    f"shape={tuple(inputs['points'].shape)} dtype={inputs['points'].dtype}"
)
print(
    f"  images: type={type(inputs['images']).__name__} "
    f"shape={tuple(inputs['images'].shape)} dtype={inputs['images'].dtype}"
)
print(
    f"  intrinsics: type={type(inputs['intrinsics']).__name__} "
    f"shape={tuple(inputs['intrinsics'].shape)} dtype={inputs['intrinsics'].dtype}"
)
print(
    f"  extrinsics: type={type(inputs['extrinsics']).__name__} "
    f"shape={tuple(inputs['extrinsics'].shape)} dtype={inputs['extrinsics'].dtype}"
)

print("targets:")
print(
    f"  boxes: type={type(targets['boxes']).__name__} "
    f"shape={tuple(targets['boxes'].shape)} dtype={targets['boxes'].dtype} "
    f"format={targets['boxes'].format.name}"
)
print(
    f"  labels: type={type(targets['labels']).__name__} "
    f"shape={tuple(targets['labels'].shape)} dtype={targets['labels'].dtype}"
)

# %%
# Batch with :func:`vision3d.datasets.collate_fn`
# -----------------------------------------------
# Variable-size tensors (point clouds, per-frame box counts) cannot be stacked
# along a batch dimension, so :func:`vision3d.datasets.collate_fn` returns
# tuples-of-tensors keyed the same as the per-sample dicts. Pass it as the
# ``collate_fn`` argument to :class:`~torch.utils.data.DataLoader` whenever
# you train or evaluate on a vision3d dataset.

from torch.utils.data import DataLoader

from vision3d.datasets import collate_fn

loader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)
batch_inputs, batch_targets = next(iter(loader))

print(f"batch size: {len(batch_inputs)}")
for i, (inp, tgt) in enumerate(zip(batch_inputs, batch_targets)):
    print(
        f"  sample {i}: "
        f"points={tuple(inp['points'].shape)} "
        f"boxes={tuple(tgt['boxes'].shape)}"
    )

# %%
# Fake some predictions
# ---------------------
# nuScenes only ships ground-truth annotations, so to demonstrate the
# ground-truth-vs-prediction overlay we synthesize a
# :class:`~vision3d.metrics.Prediction3D` from each frame's targets:
# jitter the boxes, drop a few, and assign random confidence scores. A
# real detector's output would slot in here unchanged.

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
# Visualize the dataset
# ---------------------
# :func:`vision3d.viz.log_sample` logs a
# :class:`~vision3d.datasets.SampleInputs` /
# :class:`~vision3d.datasets.SampleTargets` pair to `Rerun
# <https://rerun.io/>`_ for interactive visualization. Passing
# ``predictions`` overlays detections alongside the ground truth: both keep
# per-class colors, ground truth is drawn as translucent colored boxes and
# predictions as a wireframe with their confidence score in the label.
# ``score_threshold`` drops low-confidence detections.

import rerun as rr
import rerun.blueprint as rrb

from vision3d.viz import fusion_layout, log_sample

rr.init("vision3d_nuscenes", spawn=True)
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
