"""Feed a ROS 2 MCAP rosbag straight into the annotator.

Reads the bag directly -- no extracted dataset in between:

* ``/tf`` + ``/tf_static`` give the geometry. The 3D view is rooted at the
  ``CABIN`` frame, which is where the sensors are mounted, so the view rides with
  the machine as it slews. Annotations are stored in the ``map`` frame, so an
  object that never moved keeps one pose for the whole sequence.
* the three Livox lidars are merged into a single cloud in the ``CABIN`` frame.
  Captures are kept in the **map** frame and logged one entity per sweep, newest
  as ``sweep_0``, which the annotator's sweeps slider selects between. Map
  coordinates are what make accumulation line up: the cabin slews around 15
  degrees between keyframes, so sweeps held in cabin coordinates would fan out.

  Accumulating via the view's visible time range was tried and reverted: a time
  range applies to everything in the view, so past *annotation* edits rendered as
  duplicate boxes, hover highlighted one instance across every sample, and point
  radius only took on the newest one.
* the five cameras contribute intrinsics from ``CameraInfo`` and h265 samples
  from ``foxglove_msgs/CompressedVideo``. The samples are passed to the viewer
  undecoded, *and* decoded here with PyAV so points can be coloured by
  reprojection into the images.

Run the annotator, then::

    python tools/annotator/feed_mcap.py --seconds 20

Requires ``mcap``, ``mcap-ros2-support`` and ``av``.
"""

from __future__ import annotations

import argparse
import sys
from collections import deque
from pathlib import Path

import numpy as np
import rerun as rr
from mcap_ros2.reader import read_ros2_messages

sys.path.insert(0, str(Path(__file__).parent))
from mcap_labels import (
    MANUAL_TOPIC,
    first_message_time,
    read_from_bag,
    sidecar_for,
)
from mcap_source import (
    CameraDecoder,
    TransformTree,
    colorize_from_cameras,
    pointcloud_xyz_reflectivity,
    reflectivity_colors,
    transform_matrix,
)

from vision3d.viz import (
    annotator_layout,
    log_labels,
    log_source,
    reserve_box_slots,
)

DEFAULT_BAG = Path(
    "/home/sschlaepfer/docker-data-exchange/rosbags/rosbag2_2026_08_02-19_29_27_0.mcap"
)

#: The sensors are mounted on the cabin, which slews relative to the tracks.
EGO_FRAME = "CABIN"
MAP_FRAME = "map"

EGO_ENTITY = "world/ego"
# Sweeps live in the map frame, not under the ego node: each capture is stored in
# the coordinates it occupies, so accumulating them needs no re-logging.
LIDAR_ENTITY = "world/sweeps"
CAMERA_PREFIX = f"{EGO_ENTITY}/cam"
BOX_ENTITY = "world/annotations"
#: The recording's own boxes -- predictions, or a previous pass's labels. Shown
#: for reference and never written back to: the recording stays immutable.
SOURCE_ENTITY = f"{BOX_ENTITY}/source"
#: Boxes this session owns. These are what the exporter saves to the sidecar.
MANUAL_ENTITY = f"{BOX_ENTITY}/manual"

LIDAR_TOPICS = {
    "/livox/lidar_front_left/self_filtered": "livox_front_left",
    "/livox/lidar_rear_left/self_filtered": "livox_rear_left",
    "/livox/lidar_rear_right/self_filtered": "livox_rear_right",
}
#: Order fixes the camera indices, and so the grid layout below.
CAMERAS = ("Main", "MastLeftSide", "MastRightSide", "MastLeftRear", "MastRightRear")
CAMERA_GRID = ((1, 0, 2), (3, 4))
#: The lidar that paces keyframes; the other two are taken as of that moment.
KEYFRAME_LIDAR = "/livox/lidar_front_left/self_filtered"

ANNOTATION_CLASSES = ("truck", "truck_cabin", "truck_bed")


def merged_ontology(
    defaults: tuple[str, ...], loaded: dict[str, int]
) -> tuple[dict[str, int], dict[int, int]]:
    """Combine the session's classes with those a bag's annotations carry.

    The annotation context is logged statically, so logging a second one
    replaces the first outright. Handing the loader only the classes found in
    the bag therefore shrank the ontology to those, and the panel's dropdown
    with it.

    The session's own classes are never displaced -- they are what the operator
    expects to pick from, whatever the bag happens to contain. Two things can
    collide with that, and both are resolved by renumbering the bag's boxes
    rather than dropping a class:

    * the bag calls one of our classes by a different id, so its boxes are
      renumbered onto ours;
    * the bag has a different class occupying one of our ids, so that class
      moves to a free id and its boxes follow.

    Args:
        defaults: The classes this session offers, in id order.
        loaded: Class name to id, as read from a bag's annotations.

    Returns:
        ``(class name to id, old id to new id)``. The remap is empty unless a
        bag's numbering collided; apply it to the records before logging them,
        or their boxes will come back wearing the wrong label.
    """
    classes = {name: i for i, name in enumerate(defaults)}
    taken = set(classes.values())
    remap: dict[int, int] = {}

    # By id, so the outcome does not depend on dict ordering.
    for name, class_id in sorted(loaded.items(), key=lambda kv: kv[1]):
        if name in classes:
            if classes[name] != class_id:
                print(
                    f"note: the bag numbers '{name}' as {class_id}, this session "
                    f"as {classes[name]}; renumbering its boxes"
                )
                remap[class_id] = classes[name]
            continue

        if class_id in taken:
            free = max(taken) + 1
            print(
                f"note: class id {class_id} is already "
                f"'{next(n for n, i in classes.items() if i == class_id)}' here, so "
                f"the bag's '{name}' moves to {free}"
            )
            remap[class_id] = free
            class_id = free

        classes[name] = class_id
        taken.add(class_id)

    return classes, remap


def stamp_ns(header) -> int:
    """Nanoseconds from a ROS message header stamp.

    Args:
        header: A ROS message header with a ``stamp``.

    Returns:
        The stamp in nanoseconds.
    """
    return header.stamp.sec * 10**9 + header.stamp.nanosec


def log_camera(entity: str, info, tree: TransformTree, time_ns: int) -> bool:
    """Log one camera's intrinsics and its pose relative to the ego frame.

    Args:
        entity: Camera entity path.
        info: A ``sensor_msgs/CameraInfo`` message.
        tree: The transform graph.
        time_ns: Time to resolve the mount transform at.

    Returns:
        Whether the camera could be placed.
    """
    optical = info.header.frame_id
    ego_from_optical = tree.lookup(EGO_FRAME, optical, time_ns)
    if ego_from_optical is None:
        return False

    optical_from_ego = np.linalg.inv(ego_from_optical)
    rr.log(
        entity,
        rr.Transform3D(
            translation=optical_from_ego[:3, 3],
            mat3x3=optical_from_ego[:3, :3],
            relation=rr.TransformRelation.ChildFromParent,
        ),
        static=True,
    )
    rr.log(
        entity,
        rr.Pinhole(
            image_from_camera=np.asarray(info.k, dtype=np.float64).reshape(3, 3),
            width=info.width,
            height=info.height,
            camera_xyz=rr.ViewCoordinates.RDF,
        ),
        static=True,
    )
    return True


#: Read this much before the requested window so the first keyframe has a pose.
#: ``/tf`` runs at a few hundred hertz, so this is one chunk's worth of margin.
TF_PREROLL_NS = 1_000_000_000
#: ``/tf_static`` is latched -- published once, near the start -- so a windowed
#: read has to fetch it separately or the transform graph has no fixed edges.
TF_STATIC_WINDOW_NS = 5_000_000_000


def load_static_transforms(bag: Path, tree: TransformTree, start_ns: int) -> int:
    """Fill ``tree`` with the bag's latched transforms.

    Seeking straight to a window would skip these: the recording carries exactly
    one ``/tf_static`` message, a fraction of a second in, and without it nothing
    resolves to the map frame.

    Args:
        bag: Recording to read.
        tree: Transform graph to add to.
        start_ns: The bag's first message time.

    Returns:
        How many transforms were added.
    """
    added = 0
    for msg in read_ros2_messages(
        str(bag),
        topics=["/tf_static"],
        start_time=start_ns,
        end_time=start_ns + TF_STATIC_WINDOW_NS,
    ):
        for tr in msg.ros_msg.transforms:
            tree.add(
                tr.header.frame_id,
                tr.child_frame_id,
                stamp_ns(tr.header),
                transform_matrix(tr.transform.translation, tr.transform.rotation),
            )
            added += 1
    return added


def main() -> None:
    """Stream a bag into the annotator."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bag", type=Path, default=DEFAULT_BAG)
    parser.add_argument(
        "--seconds",
        type=float,
        default=20.0,
        help="Length of bag to read, from --start-at. Ignored with --all.",
    )
    parser.add_argument(
        "--all", action="store_true", help="Read to the end of the bag."
    )
    parser.add_argument(
        "--start-at", type=float, default=0.0, help="Offset into the bag, in seconds."
    )
    parser.add_argument(
        "--hz", type=float, default=10.0, help="Keyframe rate for lidar and boxes."
    )
    parser.add_argument(
        "--num-sweeps",
        type=int,
        default=5,
        help=(
            "Past captures to keep. Each is logged as its own entity and the "
            "annotator's sweeps slider chooses how many are shown."
        ),
    )
    parser.add_argument("--box-slots", type=int, default=64)
    parser.add_argument(
        "--labels",
        type=Path,
        default=None,
        help=(
            "Sidecar holding this tool's own annotations. Defaults to "
            "<bag>.labels.mcap when one exists. The recording is never written "
            "to; labels live here."
        ),
    )
    parser.add_argument(
        "--manual-topic",
        default=MANUAL_TOPIC,
        help="Topic the sidecar publishes its SceneUpdates on.",
    )
    parser.add_argument(
        "--no-labels",
        action="store_true",
        help="Ignore annotations already in the bag or in a sidecar.",
    )
    parser.add_argument(
        "--annotation-topic",
        default="/annotations/boxes",
        help=(
            "SceneUpdate topic in the recording itself, read as reference "
            "boxes. Never written to."
        ),
    )
    parser.add_argument("--point-radii", type=float, default=0.04)
    parser.add_argument("--save", type=Path, default=None)
    parser.add_argument(
        "--no-video", action="store_true", help="Skip camera video samples."
    )
    parser.add_argument(
        "--color",
        choices=("camera", "reflectivity"),
        default="camera",
        help=(
            "Point colour. 'camera' reprojects into the decoded images; "
            "'reflectivity' avoids decoding and is much cheaper."
        ),
    )
    args = parser.parse_args()

    rr.init("vision3d_annotator")
    if args.save is not None:
        rr.save(str(args.save))
    else:
        rr.connect_grpc()

    rr.send_blueprint(
        annotator_layout(
            CAMERAS,
            CAMERA_GRID,
            entity_prefix=CAMERA_PREFIX,
            box_entity=BOX_ENTITY,
            ego=EGO_ENTITY,
        )
    )
    # So the annotator knows which recording it is labelling, and where its
    # sidecar belongs.
    log_source(str(args.bag))

    source_records: list[dict] = []
    manual_records: list[dict] = []
    loaded_classes: dict[str, int] = {}

    if not args.no_labels:
        # Two independent sets. The recording's own annotations are reference
        # material -- a model's predictions, or an earlier pass -- and the
        # sidecar holds what this tool has produced. Neither is written into the
        # recording.
        source_records, source_classes = read_from_bag(args.bag, args.annotation_topic)
        loaded_classes.update(source_classes)

        sidecar = args.labels or sidecar_for(args.bag)
        if sidecar.exists():
            manual_records, manual_classes = read_from_bag(sidecar, args.manual_topic)
            loaded_classes.update(manual_classes)

    # One ontology for both sections, logged on the shared parent so each
    # inherits it. Logging it per section would have them shadow one another.
    classes, remap = merged_ontology(ANNOTATION_CLASSES, loaded_classes)
    renumbered = 0
    for record in (*source_records, *manual_records):
        if record.get("class_id") in remap:
            record["class_id"] = remap[record["class_id"]]
            renumbered += 1
    if renumbered:
        print(f"renumbered {renumbered} box(es) onto this session's classes")

    rr.log(
        BOX_ENTITY,
        rr.AnnotationContext([(i, name) for name, i in classes.items()]),
        static=True,
    )

    # `classes={}` so neither call re-logs a context of its own.
    if source_records:
        # Wireframe: the recording's boxes are reference material, so they are
        # drawn without a face. That reads as transparent against the points,
        # tells them apart from your own boxes at a glance, and leaves nothing
        # for the 3D view's drag to pick -- which is what makes them immutable
        # in practice rather than only by convention.
        logged = log_labels(
            SOURCE_ENTITY, source_records, {}, fill_mode="majorwireframe"
        )
        tracks = len({r["track"] for r in source_records})
        print(
            f"{logged} box(es) across {tracks} track(s) from the recording "
            f"({args.annotation_topic})"
        )
    if manual_records:
        logged = log_labels(MANUAL_ENTITY, manual_records, {})
        tracks = len({r["track"] for r in manual_records})
        print(f"{logged} box(es) across {tracks} track(s) from {sidecar.name}")

    tree = TransformTree()
    video_topics = {
        f"/hal/perception/{name}/compressed_video": i for i, name in enumerate(CAMERAS)
    }
    info_topics = {
        f"/hal/perception/{name}/camera_info": i for i, name in enumerate(CAMERAS)
    }
    topics = ["/tf", "/tf_static", *LIDAR_TOPICS, *info_topics, *video_topics]

    clouds: dict[str, object] = {}
    # Newest first, in map coordinates.
    history: deque[tuple[np.ndarray, np.ndarray]] = deque(
        maxlen=max(1, args.num_sweeps)
    )
    placed: set[int] = set()
    intrinsics: dict[int, np.ndarray] = {}
    cam_from_ego: dict[int, np.ndarray] = {}
    # One decoder thread per camera. Inline decoding put ~130 fps of HEVC in
    # series with the bag reader and dominated the runtime.
    decoders = (
        {index: CameraDecoder() for index in range(len(CAMERAS))}
        if args.color == "camera"
        else {}
    )
    # The bag's own start, not the first message read: with a window the read
    # begins mid-recording, and progress should still be reported as an offset
    # into the bag. The annotation topic is excluded because a bag saved by an
    # older version has a bogus timestamp there.
    origin_ns = first_message_time(args.bag, exclude=args.annotation_topic)
    last_keyframe_ns = 0
    keyframes = 0
    video_samples = 0
    dropped_frames = 0

    def elapsed(time_ns: int) -> float:
        return (time_ns - origin_ns) / 1e9

    static_edges = load_static_transforms(args.bag, tree, origin_ns)
    if not static_edges:
        # Every lookup to the map frame goes through these. Failing here is
        # silent otherwise -- keyframes just get dropped for want of a transform.
        print(
            f"warning: no /tf_static in the first "
            f"{TF_STATIC_WINDOW_NS / 1e9:.0f}s of {args.bag.name}; "
            "transforms to the map frame will not resolve"
        )

    # Seek to the window instead of reading up to it. MCAP keeps a chunk index,
    # so this is the difference between decompressing the whole recording and
    # decompressing the part that is wanted.
    window_start = origin_ns + int(args.start_at * 1e9)
    read_from = max(origin_ns, window_start - TF_PREROLL_NS)
    read_to = None if args.all else window_start + int(args.seconds * 1e9)
    print(
        f"reading {'to the end' if read_to is None else f'{args.seconds:.0f}s'} "
        f"from t={args.start_at:.1f}s, {static_edges} static transform(s)"
    )

    for msg in read_ros2_messages(
        str(args.bag), topics=topics, start_time=read_from, end_time=read_to
    ):
        topic = msg.channel.topic
        m = msg.ros_msg
        now = msg.log_time_ns

        seconds = elapsed(now)
        if seconds < args.start_at:
            # Still fill the transform graph, or the first keyframe has no pose.
            if topic in ("/tf", "/tf_static"):
                for tr in m.transforms:
                    tree.add(
                        tr.header.frame_id,
                        tr.child_frame_id,
                        stamp_ns(tr.header),
                        transform_matrix(
                            tr.transform.translation, tr.transform.rotation
                        ),
                    )
            continue

        if topic in ("/tf", "/tf_static"):
            for tr in m.transforms:
                tree.add(
                    tr.header.frame_id,
                    tr.child_frame_id,
                    stamp_ns(tr.header),
                    transform_matrix(tr.transform.translation, tr.transform.rotation),
                )
            continue

        if topic in info_topics:
            index = info_topics[topic]
            if index not in placed:
                stamp = stamp_ns(m.header)
                ego_from_cam = tree.lookup(EGO_FRAME, m.header.frame_id, stamp)
                if ego_from_cam is not None and log_camera(
                    f"{CAMERA_PREFIX}_{index}", m, tree, stamp
                ):
                    placed.add(index)
                    intrinsics[index] = np.asarray(m.k, dtype=np.float64).reshape(3, 3)
                    cam_from_ego[index] = np.linalg.inv(ego_from_cam)
            continue

        if topic in video_topics:
            index = video_topics[topic]
            sample = bytes(m.data)

            if not args.no_video:
                # Every sample is logged, not just those at keyframes: an h265
                # stream is only decodable in order.
                rr.set_time("time", timestamp=np.datetime64(now, "ns"))
                rr.log(
                    f"{CAMERA_PREFIX}_{index}",
                    rr.VideoStream(codec=rr.VideoCodec.H265, sample=sample),
                )
                video_samples += 1

            if index in decoders:
                decoders[index].submit(sample)
            continue

        # A lidar cloud.
        clouds[LIDAR_TOPICS[topic]] = m
        if topic != KEYFRAME_LIDAR or len(clouds) < len(LIDAR_TOPICS):
            continue
        if now - last_keyframe_ns < 1e9 / args.hz:
            continue
        last_keyframe_ns = now

        stamp = stamp_ns(m.header)
        map_from_ego = tree.lookup(MAP_FRAME, EGO_FRAME, stamp)
        if map_from_ego is None:
            dropped_frames += 1
            continue

        merged_xyz, merged_rgb = [], []
        for frame, cloud in clouds.items():
            ego_from_lidar = tree.lookup(EGO_FRAME, frame, stamp_ns(cloud.header))
            if ego_from_lidar is None:
                continue
            xyz, reflectivity = pointcloud_xyz_reflectivity(cloud)
            rotation = ego_from_lidar[:3, :3].T.astype(np.float32)
            merged_xyz.append(xyz @ rotation + ego_from_lidar[:3, 3].astype(np.float32))
            merged_rgb.append(reflectivity_colors(reflectivity, len(xyz)))

        if not merged_xyz:
            dropped_frames += 1
            continue

        points = np.concatenate(merged_xyz)
        colors = np.concatenate(merged_rgb)
        coverage = None
        if decoders:
            views = []
            for index, decoder in sorted(decoders.items()):
                if index not in intrinsics:
                    continue
                rgb = decoder.latest_rgb()
                if rgb is not None:
                    views.append((rgb, intrinsics[index], cam_from_ego[index]))
            if views:
                colors, coverage = colorize_from_cameras(points, views)

        # Absolute, not relative: the timeline value is what the exporter writes
        # as a SceneUpdate log time, and a bag start of 1970 makes the recording
        # unreadable everywhere else.
        rr.set_time("time", timestamp=np.datetime64(now, "ns"))
        rr.log(
            EGO_ENTITY,
            rr.Transform3D(
                translation=map_from_ego[:3, 3], mat3x3=map_from_ego[:3, :3]
            ),
        )
        # Map coordinates, so an accumulated sweep sits where its surfaces are
        # rather than where the cabin was pointing at capture time.
        in_map = points @ map_from_ego[:3, :3].T.astype(np.float32) + map_from_ego[
            :3, 3
        ].astype(np.float32)
        history.appendleft((in_map, colors))

        # The slot a capture occupies changes as it ages, so the ring is
        # rewritten each keyframe -- the coordinates never are.
        for age, (past_points, past_colors) in enumerate(history):
            rr.log(
                f"{LIDAR_ENTITY}/sweep_{age}",
                rr.Points3D(past_points, colors=past_colors, radii=args.point_radii),
            )

        if keyframes == 0:
            # Reserve creation slots once the timeline has started, so the
            # clears that register them cannot mask a later annotation.
            reserve_box_slots(MANUAL_ENTITY, args.box_slots)

        keyframes += 1
        if keyframes % 5 == 1:
            covered = (
                "" if coverage is None else f", {coverage * 100:.0f}% camera-coloured"
            )
            print(
                f"t={seconds:6.2f}s  keyframe {keyframes}: "
                f"{len(points)} points from {len(merged_xyz)} lidars, "
                f"{len(history)} sweep(s){covered}"
            )

    dropped = sum(decoder.close()[0] for decoder in decoders.values())
    print(
        f"\n{keyframes} keyframes, {video_samples} video samples, "
        f"{len(placed)}/{len(CAMERAS)} cameras placed"
    )
    if dropped:
        print(
            f"note: {dropped} video sample(s) skipped by a busy decoder; "
            "colour on those keyframes came from a slightly older frame"
        )
    if dropped_frames:
        print(f"warning: {dropped_frames} keyframe(s) dropped for want of a transform")
    if args.save is not None:
        print(f"wrote {args.save}")


if __name__ == "__main__":
    main()
