"""Visualize nuScenes 3D dataset using Rerun.

Usage::

    uv run python examples/visualize_nuscenes.py --root /path/to/nuscenes-mini
    uv run python examples/visualize_nuscenes.py --root /path/to/nuscenes-mini --frame 10
    uv run python examples/visualize_nuscenes.py --root /path/to/nuscenes-mini --num-frames 5
"""

import argparse

import rerun as rr
import rerun.blueprint as rrb

from vision3d.datasets import NuScenes3D
from vision3d.datasets.nuscenes import CAMERA_NAMES
from vision3d.viz import log_sample


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize nuScenes 3D dataset with Rerun."
    )
    parser.add_argument(
        "--root", type=str, required=True, help="Path to nuScenes root directory."
    )
    parser.add_argument("--frame", type=int, default=0, help="Starting frame index.")
    parser.add_argument(
        "--num-frames", type=int, default=1, help="Number of frames to visualize."
    )
    parser.add_argument(
        "--version", type=str, default="v1.0-mini", help="Dataset version."
    )
    parser.add_argument(
        "--split", type=str, default="train", help="Dataset split (train/val)."
    )
    rr.script_add_args(parser)
    args = parser.parse_args()

    # Build camera views: one per camera with box projection
    cam_views = [
        rrb.Spatial2DView(
            name=cam_name,
            origin=f"/world/cam_{i}",
            contents=[
                "+ $origin/**",
                "+ /world/boxes/**",
            ],
            overrides={
                "/world/boxes": rr.Boxes3D.from_fields(fill_mode="majorwireframe"),
            },
        )
        for i, cam_name in enumerate(CAMERA_NAMES)
    ]

    blueprint = rrb.Blueprint(
        rrb.Vertical(
            rrb.Horizontal(
                rrb.Spatial3DView(origin="/world", name="3D"),
                column_shares=[1],
            ),
            rrb.Grid(*cam_views),
            row_shares=[3, 2],
        ),
    )

    rr.script_setup(args, "vision3d_nuscenes")
    rr.send_blueprint(blueprint)

    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

    ds = NuScenes3D(args.root, version=args.version, split=args.split)
    print(f"Dataset: {len(ds)} frames")

    end = min(args.frame + args.num_frames, len(ds))

    # Build class label mapping from dataset
    label_to_id = ds.class_to_idx
    frames = []
    for i in range(args.frame, end):
        inputs, targets = ds[i]
        frames.append((inputs, targets))

    # Log annotation context once
    annotation_context = [(i, label) for label, i in label_to_id.items()]
    rr.log("world/boxes", rr.AnnotationContext(annotation_context), static=True)

    # Log frames
    for i, (inputs, targets) in enumerate(frames, start=args.frame):
        rr.set_time("frame", sequence=i)
        log_sample(inputs, targets, label_to_id=label_to_id)
        print(f"  Logged frame {i}")

    rr.script_teardown(args)


if __name__ == "__main__":
    main()
