"""Visualize KITTI 3D dataset using Rerun.

Usage::

    uv run python examples/visualize_kitti.py --root /path/to/kitti
    uv run python examples/visualize_kitti.py --root /path/to/kitti --frame 100
    uv run python examples/visualize_kitti.py --root /path/to/kitti --num-frames 10
"""

import argparse

import rerun as rr
import rerun.blueprint as rrb

from vision3d.datasets import Kitti3D
from vision3d.viz import log_sample


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize KITTI 3D dataset with Rerun."
    )
    parser.add_argument(
        "--root", type=str, required=True, help="Path to KITTI root directory."
    )
    parser.add_argument("--frame", type=int, default=0, help="Starting frame index.")
    parser.add_argument(
        "--num-frames", type=int, default=1, help="Number of frames to visualize."
    )
    rr.script_add_args(parser)
    args = parser.parse_args()

    blueprint = rrb.Horizontal(
        rrb.Spatial3DView(origin="/world", name="3D"),
        rrb.Spatial2DView(
            origin="/world/cam",
            name="Camera",
            contents=[
                "+ $origin/**",
                "+ /world/boxes/**",
            ],
            overrides={
                "/world/boxes": rr.Boxes3D.from_fields(fill_mode="majorwireframe"),
            },
        ),
    )

    rr.script_setup(args, "vision3d_kitti")
    rr.send_blueprint(blueprint)

    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

    ds = Kitti3D(args.root, train=True)
    print(f"Dataset: {len(ds)} frames")

    end = min(args.frame + args.num_frames, len(ds))

    # Build class label mapping from dataset
    label_to_id = ds.class_to_idx
    frames = []
    for i in range(args.frame, end):
        inputs, targets = ds[i]
        frames.append((inputs, targets))

    # Log frames
    for i, (inputs, targets) in enumerate(frames, start=args.frame):
        rr.set_time("frame", sequence=i)
        log_sample(inputs, targets, label_to_id=label_to_id)
        print(f"  Logged frame {i}")

    rr.script_teardown(args)


if __name__ == "__main__":
    main()
