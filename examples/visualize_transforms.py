"""Visualize vision3d transforms on nuScenes dataset with Rerun.

Shows the original and each transform result in tabs. Each tab has a 3D view
and all 6 camera projections.

Usage::

    uv run python examples/visualize_transforms.py --root /path/to/nuscenes-mini
    uv run python examples/visualize_transforms.py --root /path/to/nuscenes-mini --frame 10
"""

import argparse
import math

import rerun as rr
import rerun.blueprint as rrb
from torchvision.transforms import v2

from vision3d.datasets import NuScenes3D
from vision3d.datasets.nuscenes import CAMERA_NAMES
from vision3d.transforms import (
    CopyPaste3D,
    RandomFlip3D,
    RandomRotate3D,
    RandomScale3D,
    RandomTranslate3D,
)
from vision3d.viz import log_sample


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize 3D transforms on nuScenes data with Rerun."
    )
    parser.add_argument(
        "--root", type=str, required=True, help="Path to nuScenes root directory."
    )
    parser.add_argument("--frame", type=int, default=10, help="Frame index to use.")
    parser.add_argument(
        "--version", type=str, default="v1.0-mini", help="Dataset version."
    )
    parser.add_argument(
        "--split", type=str, default="train", help="Dataset split (train/val)."
    )
    rr.script_add_args(parser)
    args = parser.parse_args()

    ds = NuScenes3D(args.root, version=args.version, split=args.split)
    inputs, targets = ds[args.frame]

    # Label mapping from dataset (class index -> name)
    label_to_id = ds.class_to_idx

    # Target all classes for copy-paste
    all_class_ids = list(range(len(ds.classes)))
    copy_paste = CopyPaste3D(target_counts={c: 30 for c in all_class_ids}, min_points=5)

    db_range = range(max(0, args.frame - 15), args.frame)
    for i in db_range:
        inp_i, tgt_i = ds[i]
        copy_paste((inp_i,), (tgt_i,))
    print(
        f"  Database populated from {len(db_range)} frames, {len(ds.classes)} classes"
    )

    transforms = [
        ("original", "Original", None),
        # 3D spatial
        ("flip_z", "RandomFlip3D(axis='z')", RandomFlip3D(axis="z", p=1.0)),
        (
            "translate",
            "RandomTranslate3D(5.0)",
            RandomTranslate3D(translation_range=5.0, p=1.0),
        ),
        (
            "rotate",
            "RandomRotate3D(pi/4)",
            RandomRotate3D(angle_range=math.pi / 4, p=1.0),
        ),
        (
            "scale",
            "RandomScale3D(0.5, 1.5)",
            RandomScale3D(scale_range=(0.5, 1.5), p=1.0),
        ),
        (
            "color_jitter",
            "ColorJitter",
            v2.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1),
        ),
        ("gaussian_blur", "GaussianBlur", v2.GaussianBlur(kernel_size=15, sigma=5.0)),
        ("grayscale", "Grayscale", v2.Grayscale()),
        ("copy_paste", "CopyPaste3D", copy_paste),
    ]

    # Build blueprint: one tab per transform, each with 3D + 6 camera views
    tabs = []
    for prefix, name, _ in transforms:
        cam_views = [
            rrb.Spatial2DView(
                name=cam_name,
                origin=f"/{prefix}/cam_{i}",
                contents=[
                    "+ $origin/**",
                    f"+ /{prefix}/boxes/**",
                ],
                overrides={
                    f"/{prefix}/boxes": rr.Boxes3D.from_fields(
                        fill_mode="majorwireframe"
                    ),
                },
            )
            for i, cam_name in enumerate(CAMERA_NAMES)
        ]
        tabs.append(
            rrb.Vertical(
                rrb.Spatial3DView(origin=f"/{prefix}", name="3D"),
                rrb.Grid(*cam_views),
                row_shares=[3, 2],
                name=name,
            )
        )

    blueprint = rrb.Blueprint(rrb.Tabs(*tabs))

    rr.script_setup(args, "vision3d_transforms")
    rr.send_blueprint(blueprint)

    for prefix, name, transform in transforms:
        rr.log(prefix, rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

        if transform is None:
            t_inputs, t_targets = inputs, targets
        elif isinstance(transform, CopyPaste3D):
            out_inputs, out_targets = transform((inputs,), (targets,))
            t_inputs, t_targets = out_inputs[0], out_targets[0]
        else:
            t_inputs, t_targets = transform(inputs, targets)

        if label_to_id:
            annotation_context = [(i, label) for label, i in label_to_id.items()]
            rr.log(
                f"{prefix}/boxes",
                rr.AnnotationContext(annotation_context),
                static=True,
            )

        log_sample(t_inputs, t_targets, entity_prefix=prefix, label_to_id=label_to_id)
        n_boxes = t_targets["boxes"].shape[0] if t_targets else 0
        print(f"  Logged: {name} ({n_boxes} boxes)")

    rr.script_teardown(args)


if __name__ == "__main__":
    main()
