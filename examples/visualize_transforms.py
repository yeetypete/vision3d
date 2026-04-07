"""Visualize vision3d transforms on nuScenes dataset with Rerun.

Shows the original and each transform result in tabs. Each tab has a 3D view
and all 6 camera projections.

Usage::

    uv run python examples/visualize_transforms.py --root /path/to/nuscenes-mini
    uv run python examples/visualize_transforms.py --root /path/to/nuscenes-mini --frame 10
"""

import argparse

import rerun as rr
import rerun.blueprint as rrb

from vision3d.datasets import NuScenes3D
from vision3d.datasets.nuscenes import CAMERA_NAMES
from vision3d.transforms import RandomFlip3D, RandomTranslate3D
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

    transforms = [
        ("original", "Original", None),
        ("flip_z", "RandomFlip3D(axis='z')", RandomFlip3D(axis="z", p=1.0)),
        (
            "translate",
            "RandomTranslate3D(5.0)",
            RandomTranslate3D(translation_range=5.0, p=1.0),
        ),
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

    ds = NuScenes3D(args.root, version=args.version, split=args.split)
    inputs, targets = ds[args.frame]

    # Build class label mapping
    label_to_id: dict[str, int] = {}
    if targets and "class_names" in targets:
        for name in targets["class_names"]:
            if name not in label_to_id:
                label_to_id[name] = len(label_to_id)

    for prefix, name, transform in transforms:
        rr.log(prefix, rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

        if label_to_id:
            annotation_context = [(i, label) for label, i in label_to_id.items()]
            rr.log(
                f"{prefix}/boxes",
                rr.AnnotationContext(annotation_context),
                static=True,
            )

        if transform is None:
            t_inputs, t_targets = inputs, targets
        else:
            t_inputs, t_targets = transform(inputs, targets)

        log_sample(t_inputs, t_targets, entity_prefix=prefix, label_to_id=label_to_id)
        print(f"  Logged: {name}")

    rr.script_teardown(args)


if __name__ == "__main__":
    main()
