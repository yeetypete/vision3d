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
# Densify with multiple lidar sweeps
# ----------------------------------
# A single lidar key-frame is sparse, which limits 3D detection accuracy.
# Multi-sweep accumulation increases point density by combining the sweeps
# captured between key-frames into the current frame. Each sweep is
# motion-compensated into the key-frame lidar frame, and every point is
# annotated with its time offset relative to the key-frame. The denser point
# cloud retains temporal information and improves model performance.
#
# The technique was popularized by the nuScenes dataset (Caesar et al.,
# "nuScenes: A Multimodal Dataset for Autonomous Driving", CVPR 2020,
# `arXiv:1903.11027 <https://arxiv.org/abs/1903.11027>`_), which accumulates
# 10 sweeps, roughly 0.5 seconds at the 2 Hz key-frame rate.
#
# In vision3d, set ``num_sweeps`` on :class:`~vision3d.datasets.NuScenes3D`.
# Each point gains a trailing time-offset column, so the point cloud grows from
# ``[N, 5]`` to ``[N, 6]`` channels (x, y, z, intensity, ring number, time).

dense = NuScenes3D(NUSCENES_ROOT, version="v1.0-mini", split="train", num_sweeps=10)

# A mid-scene frame. The first frames of a scene have fewer prior sweeps to
# fold in, so aggregation falls back to whatever is available.
frame = 10
single_points = dataset[frame][0]["points"]
dense_points = dense[frame][0]["points"]

print(f"single sweep: {tuple(single_points.shape)} (x,y,z,intensity,ring)")
print(f"10 sweeps:    {tuple(dense_points.shape)} (x,y,z,intensity,ring,time)")
time = dense_points[:, -1]
print(f"time column:  {time.min():.3f} to {time.max():.3f} s before key-frame")

import rerun as rr
import rerun.blueprint as rrb

from vision3d.viz import fusion_layout, log_sample

dense_inputs, dense_targets = dense[frame]

rr.init("vision3d_nuscenes_sweeps", spawn=True)
rr.send_blueprint(
    rrb.Blueprint(
        fusion_layout(NuScenes3D.camera_names, NuScenes3D.camera_grid),
        rrb.TimePanel(state="collapsed"),
    )
)
rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
log_sample(dense_inputs, dense_targets, label_to_id=dense.class_to_idx, jpeg_quality=75)

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
# Visualize the dataset
# ---------------------
# :func:`vision3d.viz.log_sample` logs a
# :class:`~vision3d.datasets.SampleInputs` /
# :class:`~vision3d.datasets.SampleTargets` pair to `Rerun
# <https://rerun.io/>`_ for interactive visualization. See the
# visualization examples for overlaying detector predictions on the
# ground truth.

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

for frame_idx in range(10):
    f_inputs, f_targets = dataset[frame_idx]
    rr.set_time("frame", sequence=frame_idx)
    log_sample(f_inputs, f_targets, label_to_id=dataset.class_to_idx, jpeg_quality=75)
