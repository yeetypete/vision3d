"""
Augmenting samples with vision3d transforms
===========================================

This example showcases :mod:`vision3d.transforms` on the nuScenes dataset.

Transforms automatically dispatch on the tensor types from
:mod:`vision3d.tensors` carried by the sample, so a geometric transform
like :class:`~vision3d.transforms.RandomRotate3D` updates points, boxes,
and extrinsics together without requiring any special handling.

Every transform in :mod:`vision3d.transforms` is input and dataset
agnostic by design. They support every
:class:`~vision3d.tensors.BoundingBox3DFormat`, run on lidar-only,
camera-only, and fusion samples, and make no assumptions about scene
composition such as camera count, sensor layout, or axis convention.
"""

# %%
# Construct the dataset
# ---------------------
# Grab a single ``(inputs, targets)`` sample to act as the baseline that
# every transform is applied to.

from pathlib import Path

import torch

from vision3d.datasets import NuScenes3D

NUSCENES_ROOT = Path("~/.cache/vision3d/nuscenes-mini").expanduser()
FRAME_INDEX = 100

torch.manual_seed(42)

dataset = NuScenes3D(NUSCENES_ROOT, version="v1.0-mini", split="train", download=True)
inputs, targets = dataset[FRAME_INDEX]
print(f"num boxes: {targets['boxes'].shape[0]}")

# %%
# Visualize the baseline sample
# -----------------------------
# Render the original sample in an embedded Rerun viewer to use as
# a reference for the transformed scenes shown later in this example.

import rerun as rr
import rerun.blueprint as rrb

from vision3d.viz import fusion_layout, log_sample

label_to_id = dataset.class_to_idx

rr.init("vision3d_original", spawn=True)
rr.send_blueprint(
    rrb.Blueprint(
        fusion_layout(
            NuScenes3D.camera_names,
            NuScenes3D.camera_grid,
            entity_prefix="original",
            name="Original",
        ),
        rrb.TimePanel(state="hidden"),
    )
)
rr.log("original", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
log_sample(
    inputs,
    targets,
    entity_prefix="original",
    label_to_id=label_to_id,
    jpeg_quality=75,
)

# %%
# Apply a single transform
# ------------------------
# Every transform is a :class:`torchvision.transforms.v2.Transform` that
# accepts ``(inputs, targets)`` and returns the transformed pair. The
# :class:`~vision3d.tensors.PointCloud3D`,
# :class:`~vision3d.tensors.CameraImages`, and
# :class:`~vision3d.tensors.BoundingBoxes3D` semantic tensor types
# steer dispatch, so a single call updates geometry, imagery, and box
# annotations with geometric and photometric consistency.

import math

from vision3d.transforms import RandomRotate3D

rotate = RandomRotate3D(angle_range=math.pi / 4, p=1.0)
r_inputs, r_targets = rotate(inputs, targets)
print(f"rotated boxes shape: {tuple(r_targets['boxes'].shape)}")

# %%
# Geometric safety
# ----------------
# vision3d transforms mirror the torchvision v2 dispatch model: each
# transform declares the input types it operates on via the class-level
# ``_transformed_types`` tuple, and any input whose type is not listed
# passes through unchanged. Transforms whose operation would only be
# correct on a subset of scene types additionally override
# :meth:`~vision3d.transforms.Transform.check_inputs` to raise
# :class:`TypeError` for input combinations they cannot handle. Together
# these guard against silently producing geometrically inconsistent
# scenes (e.g. flipping the lidar but not the camera image alongside
# it).
#
# For example, :class:`~vision3d.transforms.RandomFlip3D` operates on
# :class:`~vision3d.tensors.PointCloud3D` and
# :class:`~vision3d.tensors.BoundingBoxes3D`, and its ``check_inputs``
# refuses samples that also carry camera tensors (images, extrinsics,
# intrinsics): flipping the 3D scene without coordinated changes to the
# camera side would break geometric consistency. Running it on a fusion
# dataset sample therefore raises a :class:`TypeError`.
#
# The error signals that the transform is not compatible with a fusion
# pipeline. :class:`~vision3d.transforms.RandomFlip3D` is intended for
# lidar-only training pipelines, where there are no camera tensors to
# fall out of correspondence.

from vision3d.transforms import RandomFlip3D

flip = RandomFlip3D(axis="x", p=1.0)
try:
    flip(inputs, targets)
except TypeError as e:
    print(e)

# %%
# Camera-coordinated flips
# ------------------------
# Flipping a fusion sample *is* well defined when the flip is expressed in
# image space rather than world space. vision3d registers
# ``horizontal_flip`` / ``vertical_flip`` kernels for every camera tensor,
# so torchvision's
# :class:`~torchvision.transforms.v2.RandomHorizontalFlip` and
# :class:`~torchvision.transforms.v2.RandomVerticalFlip` update the images,
# intrinsics, and extrinsics together with the points and boxes.
# For an upright camera rig a horizontal image flip maps to
# a world **Y** reflection and a vertical flip to a world **Z** reflection.
# These two transforms appear in the showcase below. The remaining
# ``RandomFlip3D`` world **X** flip has no image-space equivalent and stays
# lidar-only.

from torchvision.transforms import v2

hflip_inputs, hflip_targets = v2.RandomHorizontalFlip(p=1.0)(inputs, targets)
print(f"image-flipped boxes shape: {tuple(hflip_targets['boxes'].shape)}")

# %%
# Composing transforms
# --------------------
# vision3d transforms are designed to be chained together to build
# advanced data-augmentation pipelines to be used during training.
# The standard :class:`torchvision.transforms.v2.Compose` can run 3D
# transforms alongside any tensor-aware torchvision image transform.
# torchvision image transforms see only the
# :class:`~vision3d.tensors.CameraImages` tensor and leave 3D geometry
# untouched, so mixing them is safe.

from torchvision.transforms import v2

from vision3d.transforms import (
    PointJitter,
    RandomScale3D,
    RandomTranslate3D,
    RangeFilter3D,
)

compose = v2.Compose(
    [
        RandomRotate3D(angle_range=math.pi / 4, p=1.0),
        RandomScale3D(scale_range=(0.7, 1.3), p=1.0),
        RandomTranslate3D(translation_range=5.0, p=1.0),
        PointJitter(sigma=0.1, p=1.0),
        RangeFilter3D(point_cloud_range=(-50, -50, -5, 50, 50, 3)),
        v2.Resize(size=[450, 800]),
        v2.CenterCrop(size=[400, 700]),
        v2.ColorJitter(brightness=0.6, contrast=0.6, saturation=0.6, hue=0.3),
    ]
)

c_inputs, c_targets = compose(inputs, targets)
print(f"composed points: {tuple(c_inputs['points'].shape)}")
print(f"composed images: {tuple(c_inputs['images'].shape)}")
print(f"composed boxes:  {tuple(c_targets['boxes'].shape)}")

# %%
# Cross-sample augmentation with CopyPaste3D
# ------------------------------------------
# :class:`~vision3d.transforms.CopyPaste3D` is an advanced augmentation
# method based on the ground-truth sampling technique first introduced
# in `SECOND <https://www.mdpi.com/1424-8220/18/10/3337>`_. It improves
# scene diversity by injecting instances from other scenes into the
# current one. Unlike single-sample transforms
# :class:`~vision3d.transforms.CopyPaste3D`
# operates on collated batches and reads from an internal object
# database that grows lazily with each seen batch.

from torch.utils.data import DataLoader, Subset

from vision3d.datasets import collate_fn
from vision3d.transforms import CopyPaste3D

target_counts = {
    dataset.class_to_idx["car"]: 30,
    dataset.class_to_idx["pedestrian"]: 20,
    dataset.class_to_idx["traffic_cone"]: 15,
}
copy_paste = CopyPaste3D(target_counts=target_counts, min_points=5)

dataset_range = list(range(max(0, FRAME_INDEX - 10), FRAME_INDEX))
dataset_loader = DataLoader(
    Subset(dataset, dataset_range),
    batch_size=2,
    collate_fn=collate_fn,
)
for epoch in range(2):
    for batch_inputs, batch_targets in dataset_loader:
        copy_paste(batch_inputs, batch_targets)

cp_inputs, cp_targets = copy_paste((inputs,), (targets,))
print(f"boxes before: {targets['boxes'].shape[0]}")
print(f"boxes after CopyPaste3D:  {cp_targets[0]['boxes'].shape[0]}")

rr.init("vision3d_copy_paste", spawn=True)
rr.send_blueprint(
    rrb.Blueprint(
        fusion_layout(
            NuScenes3D.camera_names,
            NuScenes3D.camera_grid,
            entity_prefix="copy_paste",
            name="CopyPaste3D",
        ),
        rrb.TimePanel(state="hidden"),
    )
)
rr.log("copy_paste", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
log_sample(
    cp_inputs[0],
    cp_targets[0],
    entity_prefix="copy_paste",
    label_to_id=label_to_id,
    jpeg_quality=75,
)

# %%
# Transforms showcase
# ------------------------
# View every transform side by side in the embedded Rerun viewer, each
# on its own tab. Compare each tab against the baseline viewer at the
# top of the page.

from vision3d.transforms import PointSample, PointShuffle

transforms = [
    (
        "translate",
        "RandomTranslate3D(5.0)",
        RandomTranslate3D(translation_range=5.0, p=1.0),
    ),
    ("rotate", "RandomRotate3D(pi/4)", RandomRotate3D(angle_range=math.pi / 4, p=1.0)),
    (
        "scale",
        "RandomScale3D(0.25, 4.0)",
        RandomScale3D(scale_range=(0.25, 4.0), p=1.0),
    ),
    ("hflip", "RandomHorizontalFlip", v2.RandomHorizontalFlip(p=1.0)),
    ("vflip", "RandomVerticalFlip", v2.RandomVerticalFlip(p=1.0)),
    (
        "color_jitter",
        "ColorJitter",
        v2.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.4),
    ),
    ("gaussian_blur", "GaussianBlur", v2.GaussianBlur(kernel_size=31, sigma=10.0)),
    ("solarize", "Solarize", v2.RandomSolarize(threshold=0.5, p=1.0)),
    ("resize_half", "Resize(half)", v2.Resize(size=[450, 800])),
    ("center_crop", "CenterCrop()", v2.CenterCrop(size=[600, 800])),
    ("pad", "Pad(100)", v2.Pad(padding=100)),
    ("point_shuffle", "PointShuffle", PointShuffle(p=1.0)),
    ("point_sample", "PointSample(4096)", PointSample(n=4096)),
    ("point_jitter", "PointJitter(sigma=0.1)", PointJitter(sigma=0.1, p=1.0)),
    (
        "range_filter",
        "RangeFilter3D()",
        RangeFilter3D(point_cloud_range=(-30, -30, -5, 30, 30, 3)),
    ),
    ("compose", "Compose", compose),
]

rr.init("vision3d_transforms", spawn=True)
rr.send_blueprint(
    rrb.Blueprint(
        rrb.Tabs(
            *(
                fusion_layout(
                    NuScenes3D.camera_names,
                    NuScenes3D.camera_grid,
                    entity_prefix=prefix,
                    name=name,
                )
                for prefix, name, _ in transforms
            )
        ),
        rrb.TimePanel(state="hidden"),
    )
)

for prefix, name, pipeline in transforms:
    rr.log(prefix, rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    t_inputs, t_targets = pipeline(inputs, targets)
    log_sample(
        t_inputs,
        t_targets,
        entity_prefix=prefix,
        label_to_id=label_to_id,
        jpeg_quality=75,
    )
