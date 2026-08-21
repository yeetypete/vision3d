"""Feed nuScenes mini into the vision3d annotator.

Logs each frame as an RGB-colored lidar cloud, the six camera images with their
pinholes, and one editable entity per ground-truth box, then sends the
annotator layout.

Boxes are keyed by their nuScenes *instance token*, so a given object keeps the
same entity path for as long as it is tracked. That is what makes scrubbing the
timeline coherent: correcting a box at one frame stays attached to that object
rather than to whatever happened to be the n-th annotation in the frame.

Run the annotator first (it listens on the standard Rerun gRPC port)::

    cargo run --release --manifest-path tools/annotator/Cargo.toml

then feed it::

    python tools/annotator/feed_nuscenes.py --all

Or write a recording to open later::

    python tools/annotator/feed_nuscenes.py --all --save /tmp/annotate.rrd
"""

import argparse
from pathlib import Path
from typing import ClassVar

import rerun as rr

from vision3d.datasets import NuScenes3D
from vision3d.viz import (
    annotator_layout,
    boxes_to_map,
    lidar_to_map,
    log_boxes_3d_editable,
    log_cameras,
    log_ego,
    log_point_cloud_rgb_by_sweep,
    reserve_box_slots,
)

NUSCENES_MINI_ROOT = Path("~/.cache/vision3d/nuscenes-mini").expanduser()
TRUCK_BAG_ROOT = Path("/home/sschlaepfer/git/core/learning/ml3d/out/bag_0802")


class TruckBagDataset(NuScenes3D):
    """A nuScenes-layout bag extraction with a five-camera mast rig.

    The tables follow the nuScenes layout but the rig and ontology do not: five
    cameras with their own channel names, and no ``category.json`` entries at
    all, since the bag is unlabelled. That is the point -- boxes get created in
    the annotator rather than loaded.

    ``category_map`` is therefore empty: there are no source categories to map.
    ``classes`` still matters, because it is the ontology the annotator offers
    in its label dropdown.
    """

    camera_names: ClassVar[tuple[str, ...]] = (
        "Main",
        "MastLeftSide",
        "MastRightSide",
        "MastLeftRear",
        "MastRightRear",
    )
    # Row-major, indices into `camera_names`: side-front-side, then the rears.
    camera_grid: ClassVar[tuple[tuple[int, ...], ...] | None] = ((1, 0, 2), (3, 4))

    classes: ClassVar[tuple[str, ...]] = ("truck", "truck_cabin", "truck_bed")
    category_map: ClassVar[dict[str, str]] = {}


DATASETS = {
    "truck-bag": (TruckBagDataset, TRUCK_BAG_ROOT, "v1.0-mini", "all"),
    "nuscenes-mini": (NuScenes3D, NUSCENES_MINI_ROOT, "v1.0-mini", "train"),
}

# Entity layout. The annotator rewrites entities under BOX_ENTITY, so keep
# ground truth there and nothing else.
EGO_ENTITY = "world/ego"
# Sensor data is parented under the ego node, so the 3D view can be rooted there
# and render it in raw coordinates -- the view then rides with the machine.
LIDAR_ENTITY = f"{EGO_ENTITY}/lidar"
CAMERA_PREFIX = f"{EGO_ENTITY}/cam"
BOX_ENTITY = "world/annotations"


def stable_box_ids(dataset: NuScenes3D, index: int, expected: int) -> list[str] | None:
    """Derive per-object entity names from nuScenes instance tokens.

    Mirrors the dataset's own annotation filtering (``category_map`` drops
    categories outside the detection classes) so the returned ids line up
    one-to-one with ``targets["boxes"]``.

    Reaches into the dataset's private devkit handle, since
    :class:`~vision3d.datasets.SampleTargets` carries no instance identity.

    Args:
        dataset: The nuScenes dataset being fed.
        index: Sample index.
        expected: Number of boxes the dataset returned for this sample.

    Returns:
        One id per box, or ``None`` if identity could not be derived or did not
        line up, in which case the caller should fall back to positional ids.
    """
    try:
        sample = dataset._nusc.get("sample", dataset._sample_tokens[index])
        ids = []
        for ann_token in sample["anns"]:
            ann = dataset._nusc.get("sample_annotation", ann_token)
            if dataset.category_map.get(ann["category_name"]) is None:
                continue
            ids.append(ann["instance_token"])
    except (AttributeError, KeyError):
        return None

    return ids if len(ids) == expected else None


def main() -> None:
    """Parse arguments and stream nuScenes frames to the annotator."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=sorted(DATASETS),
        default="truck-bag",
        help="Which dataset layout and ontology to use.",
    )
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--version", default=None)
    parser.add_argument(
        "--split",
        default=None,
        help='Scene split. Custom datasets need "all", their scenes not being nuScenes ones.',
    )
    parser.add_argument("--start", type=int, default=0, help="First sample index.")
    parser.add_argument("--frames", type=int, default=1, help="Number of samples.")
    parser.add_argument("--all", action="store_true", help="Feed every sample.")
    parser.add_argument(
        "--save", type=Path, default=None, help="Write an .rrd instead of connecting."
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=80,
        help="Camera JPEG quality. Lower keeps viewer memory down over long timelines.",
    )
    parser.add_argument(
        "--frame",
        choices=("map", "lidar"),
        default="map",
        help=(
            "Frame to log geometry in. 'map' keeps the world fixed and the robot "
            "moving, which is what makes an annotation stay put across frames. "
            "'lidar' keeps the robot fixed, as raw sensor data."
        ),
    )
    parser.add_argument(
        "--num-sweeps",
        type=int,
        default=10,
        help=(
            "Lidar sweeps to load. Each sweep is logged as its own entity, and "
            'the annotator\'s "sweeps" slider chooses how many are shown.'
        ),
    )
    parser.add_argument(
        "--box-slots",
        type=int,
        default=64,
        help="How many box slots to reserve for in-viewer creation.",
    )
    parser.add_argument(
        "--point-radii",
        type=float,
        default=0.04,
        help="Point radius in metres. Larger points are easier to annotate against.",
    )
    args = parser.parse_args()

    cls, default_root, default_version, default_split = DATASETS[args.dataset]
    dataset = cls(
        args.root or default_root,
        version=args.version or default_version,
        split=args.split or default_split,
        num_sweeps=args.num_sweeps,
        # Only the public mini split is downloadable; anything else is local.
        download=args.dataset == "nuscenes-mini",
    )
    label_to_id = {name: i for i, name in enumerate(dataset.classes)}
    print(f"{args.dataset}: {len(dataset)} samples, classes={dataset.classes}")

    rr.init("vision3d_annotator")
    if args.save is not None:
        rr.save(str(args.save))
    else:
        rr.connect_grpc()

    rr.send_blueprint(
        annotator_layout(
            cls.camera_names,
            cls.camera_grid,
            entity_prefix=CAMERA_PREFIX,
            box_entity=BOX_ENTITY,
        )
    )

    # Reserve creation slots before any real frame, so the clears that register
    # them cannot mask boxes the annotator writes later.
    rr.set_time("frame", sequence=args.start - 1)
    reserve_box_slots(BOX_ENTITY, args.box_slots)

    end = len(dataset) if args.all else min(args.start + args.frames, len(dataset))
    live: set[str] = set()
    positional_frames = 0
    anchor = None
    frameless = 0

    for index in range(args.start, end):
        inputs, targets = dataset[index]
        rr.set_time("frame", sequence=index)

        points = inputs["points"]
        boxes = targets["boxes"]
        extrinsics = inputs["extrinsics"]

        if args.frame == "map":
            if anchor is None:
                first = lidar_to_map(dataset, args.start)
                # Anchor on the first sample so map coordinates stay small.
                anchor = None if first is None else first[:3, 3].clone()
            lidar_from_map = lidar_to_map(dataset, index, anchor)
            if lidar_from_map is None:
                frameless += 1
            else:
                # The ego node carries the pose; everything under it stays in
                # sensor coordinates and is placed in the map by this transform.
                log_ego(EGO_ENTITY, lidar_from_map)
                boxes = boxes_to_map(boxes, lidar_from_map)

        sweeps = log_point_cloud_rgb_by_sweep(
            LIDAR_ENTITY,
            points,
            inputs["images"],
            inputs["intrinsics"],
            extrinsics,
            radii=args.point_radii,
        )

        log_cameras(
            CAMERA_PREFIX,
            inputs["images"],
            inputs["intrinsics"],
            extrinsics,
            jpeg_quality=args.jpeg_quality,
        )

        n = len(boxes)
        box_ids = stable_box_ids(dataset, index, n)
        if box_ids is None:
            box_ids = [f"box_{i}" for i in range(n)]
            positional_frames += 1

        log_boxes_3d_editable(
            BOX_ENTITY,
            boxes,
            labels=[dataset.classes[int(i)] for i in targets["labels"]],
            class_ids=[int(i) for i in targets["labels"]],
            label_to_id=label_to_id,
            box_ids=box_ids,
        )

        # Objects that left the scene must be cleared, or latest-at semantics
        # would keep showing their last known pose for the rest of the timeline.
        for gone in live - set(box_ids):
            rr.log(f"{BOX_ENTITY}/{gone}", rr.Clear(recursive=True))
        live = set(box_ids)

        if index % 25 == 0 or index == end - 1:
            print(
                f"frame {index}/{end - 1}: {n} boxes, "
                f"{len(inputs['points'])} points in {sweeps} sweep(s)"
            )

    if frameless:
        print(
            f"warning: {frameless} frame(s) had no readable ego pose and stayed in "
            "the lidar frame; annotations on those will move with the robot"
        )
    if positional_frames:
        print(
            f"warning: {positional_frames} frame(s) fell back to positional box ids; "
            "object identity is not stable across those frames"
        )
    if args.save is not None:
        print(f"wrote {args.save}")


if __name__ == "__main__":
    main()
