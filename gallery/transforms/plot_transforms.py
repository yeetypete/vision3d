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
print(f"num boxes: {targets['boxes'].shape[0]} boxes")

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
# current frame. Unlike single-sample transforms
# :class:`~vision3d.transforms.CopyPaste3D`
# operates on collated batches and reads from an internal object
# database that grows lazily as batches pass through.

from torch.utils.data import DataLoader, Subset

from vision3d.datasets import SampleInputs, SampleTargets, collate_fn
from vision3d.transforms import CopyPaste3D

target_counts = {
    dataset.class_to_idx["car"]: 30,
    dataset.class_to_idx["pedestrian"]: 20,
}
copy_paste = CopyPaste3D(target_counts=target_counts, min_points=5)

warmup_indices = list(range(max(0, FRAME_INDEX - 10), FRAME_INDEX))
dataset_loader = DataLoader(
    Subset(dataset, warmup_indices),
    batch_size=2,
    collate_fn=collate_fn,
)
for epoch in range(2):
    for batch_inputs, batch_targets in dataset_loader:
        copy_paste(batch_inputs, batch_targets)


# %%
# Wrap the batched signature with a small adapter so it fits the
# single-sample ``(inputs, targets)`` interface used by the
# visualization loop below.


def copy_paste_one(
    inp: SampleInputs, tgt: SampleTargets
) -> tuple[SampleInputs, SampleTargets]:
    out_inp, out_tgt = copy_paste((inp,), (tgt,))
    return out_inp[0], out_tgt[0]


# %%
# Transform showcase
# ------------------------
# View every transform side by side in the embedded Rerun viewer,
# each on its own tab. The first tab shows the original, untransformed
# sample as a reference.

import rerun as rr
import rerun.blueprint as rrb

from vision3d.transforms import PointSample, PointShuffle, RandomFlip3D
from vision3d.viz import fusion_layout, log_sample

transforms = [
    ("original", "Original", lambda i, t: (i, t)),
    ("flip_z", "RandomFlip3D(axis='z')", RandomFlip3D(axis="z", p=1.0)),
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
    ("copy_paste", "CopyPaste3D", copy_paste_one),
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

label_to_id = dataset.class_to_idx
annotation_context = [(i, label) for label, i in label_to_id.items()]

for prefix, name, pipeline in transforms:
    rr.log(prefix, rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    rr.log(f"{prefix}/boxes", rr.AnnotationContext(annotation_context), static=True)
    t_inputs, t_targets = pipeline(inputs, targets)
    log_sample(
        t_inputs,
        t_targets,
        entity_prefix=prefix,
        label_to_id=label_to_id,
        jpeg_quality=75,
    )
