"""Visualize vision3d transforms on KITTI dataset with Rerun.

Usage::

    uv run python examples/visualize_transforms.py --root /path/to/kitti
    uv run python examples/visualize_transforms.py --root /path/to/kitti --frame 10
"""

import argparse

import rerun as rr
import rerun.blueprint as rrb

from vision3d.datasets import Kitti3D
from vision3d.transforms import RandomFlip3D
from vision3d.viz import log_sample


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize 3D transforms on KITTI data with Rerun."
    )
    parser.add_argument(
        "--root", type=str, required=True, help="Path to KITTI root directory."
    )
    parser.add_argument("--frame", type=int, default=10, help="Frame index to use.")
    rr.script_add_args(parser)
    args = parser.parse_args()

    # Each transform gets its own entity prefix
    transforms = [
        ("original", "Original", None),
        ("flip_x", "Flip X", RandomFlip3D(axis="x", p=1.0)),
        ("flip_y", "Flip Y", RandomFlip3D(axis="y", p=1.0)),
        ("flip_z", "Flip Z", RandomFlip3D(axis="z", p=1.0)),
    ]

    # Build blueprint: tabs for each transform, each tab has 3D + camera side by side
    tabs = []
    for prefix, name, _ in transforms:
        tabs.append(
            rrb.Horizontal(
                rrb.Spatial3DView(origin=f"/{prefix}", name="3D"),
                rrb.Spatial2DView(
                    origin=f"/{prefix}/cam",
                    name="Camera",
                    contents=[
                        "+ $origin/**",
                        f"+ /{prefix}/boxes/**",
                    ],
                    overrides={
                        f"/{prefix}/boxes": rr.Boxes3D.from_fields(
                            fill_mode="majorwireframe"
                        ),
                    },
                ),
                name=name,
            )
        )

    blueprint = rrb.Blueprint(rrb.Tabs(*tabs))

    rr.script_setup(args, "vision3d_transforms")
    rr.send_blueprint(blueprint)

    ds = Kitti3D(args.root, train=True)
    inputs, targets = ds[args.frame]

    # Build class label mapping
    label_to_id: dict[str, int] = {}
    if targets and "class_names" in targets:
        for name in targets["class_names"]:
            if name not in label_to_id:
                label_to_id[name] = len(label_to_id)

    for prefix, name, transform in transforms:
        # Log coordinate system for each view
        rr.log(prefix, rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

        if label_to_id:
            annotation_context = [(i, label) for label, i in label_to_id.items()]
            rr.log(
                f"{prefix}/boxes", rr.AnnotationContext(annotation_context), static=True
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
