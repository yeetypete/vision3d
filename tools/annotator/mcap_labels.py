"""Write annotations into a bag as ``foxglove_msgs/SceneUpdate``.

The point is that labels and model predictions share one schema: whatever the
model publishes in production is what the annotator edits, and Foxglove renders
either without extra work.

The bag is rewritten rather than appended to, because MCAP keeps its index in a
trailer and the Python library has no append mode. Message payloads are passed
through untouched -- only chunking and compression are redone -- which puts a
900 MB bag at roughly five seconds. Appending in place is possible (truncate at
``DataEnd``, then regenerate the summary, since chunk offsets before it stay
valid) but means hand-writing MCAP records, and the failure mode is a corrupted
recording. Not worth it for a few seconds.

The write goes to a temporary file beside the bag and is renamed over it only
after the message count is verified, so an interrupted save cannot truncate the
original.
"""

from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path

from mcap.reader import make_reader
from mcap.well_known import MessageEncoding, SchemaEncoding
from mcap.writer import CompressionType, Writer

SCHEMA_PATH = Path(__file__).parent / "schemas" / "foxglove_SceneUpdate.msg"
SCENE_UPDATE_TYPE = "foxglove_msgs/msg/SceneUpdate"

#: Per-class colours, so Foxglove distinguishes them without configuration.
CLASS_COLORS = (
    (0.95, 0.35, 0.35),
    (0.35, 0.85, 0.45),
    (0.35, 0.55, 0.95),
    (0.95, 0.75, 0.30),
    (0.75, 0.45, 0.95),
)


def _color(class_id: int | None) -> dict:
    r, g, b = CLASS_COLORS[(class_id or 0) % len(CLASS_COLORS)]
    return {"r": r, "g": g, "b": b, "a": 0.35}


def _entity(record: dict, lifetime_ns: int, frame: str, stamp_ns: int) -> dict:
    """Build a SceneEntity for one annotation record.

    Args:
        record: A row as written by the annotator's exporter.
        lifetime_ns: How long the entity lives; 0 means forever, which is how a
            box marked static is expressed.
        frame: Frame the pose is in.
        stamp_ns: The entity's timestamp. Passed in rather than read from the
            record because a static box has no time of its own, and a timestamp
            of zero puts it in 1970 -- where no transform to ``frame`` exists,
            so the box cannot be placed at all.

    Returns:
        A SceneEntity dict ready for CDR encoding.
    """
    t = stamp_ns
    half = record["half_size"]
    quat = record["quat"]

    metadata = [{"key": "source", "value": "human"}]
    if record.get("class") is not None:
        metadata.append({"key": "class", "value": str(record["class"])})
    if record.get("class_id") is not None:
        metadata.append({"key": "class_id", "value": str(record["class_id"])})
    if record.get("static"):
        metadata.append({"key": "static", "value": "true"})

    return {
        "timestamp": {"sec": t // 10**9, "nanosec": t % 10**9},
        "frame_id": frame,
        # The track id: this is what makes a reload edit the same object rather
        # than create a second one beside it.
        "id": str(record["track"]),
        "lifetime": {"sec": lifetime_ns // 10**9, "nanosec": lifetime_ns % 10**9},
        "frame_locked": True,
        "metadata": metadata,
        "arrows": [],
        "cubes": [
            {
                "pose": {
                    "position": dict(zip("xyz", record["center"], strict=True)),
                    "orientation": dict(zip("xyzw", quat, strict=True)),
                },
                # SceneUpdate cubes take full extents, the exporter stores halves.
                "size": {"x": half[0] * 2, "y": half[1] * 2, "z": half[2] * 2},
                "color": _color(record.get("class_id")),
            }
        ],
        "spheres": [],
        "cylinders": [],
        "lines": [],
        "triangles": [],
        "texts": [],
        "models": [],
    }


def scene_updates(
    records: list[dict],
    frame: str,
    keyframe_interval_ns: int,
    static_time_ns: int | None = None,
) -> list[tuple[int, dict]]:
    """Group annotation records into one SceneUpdate per timestamp.

    A track marked static is emitted once with an infinite lifetime, and its
    per-frame rows are dropped: an entity id identifies one object, so writing
    both left two entities fighting over the same id -- and the static one had
    no class, since the class is only ever written per frame.

    Args:
        records: Rows from the annotator's exporter.
        frame: Frame the poses are in.
        keyframe_interval_ns: Lifetime for per-frame entities. Taken from the
            actual keyframe spacing rather than hard-coded, or boxes blink when
            the bag is re-read at a different rate.
        static_time_ns: Timestamp for static entities. Should be the bag's start
            time; falls back to the first annotated keyframe.

    Returns:
        ``(log_time_ns, SceneUpdate)`` pairs, sorted by time.
    """
    by_track: dict[str, list[dict]] = {}
    for record in records:
        by_track.setdefault(record["track"], []).append(record)

    per_time: dict[int, list[dict]] = {}
    static_rows: list[dict] = []

    for rows in by_track.values():
        fixed = [r for r in rows if r.get("static") or r.get("t") is None]
        if not fixed:
            for row in rows:
                per_time.setdefault(int(row["t"]), []).append(row)
            continue

        # The last static write is the current pose; the class comes from
        # whichever row has one.
        row = dict(fixed[-1])
        if row.get("class") is None:
            named = next((r for r in rows if r.get("class") is not None), None)
            if named is not None:
                row["class"] = named["class"]
                row["class_id"] = named.get("class_id")
        static_rows.append(row)

    first = min(per_time) if per_time else 0
    stamp = static_time_ns if static_time_ns is not None else first

    out: list[tuple[int, dict]] = []
    if static_rows:
        entities = [_entity(r, 0, frame, stamp) for r in static_rows]
        out.append((stamp, {"deletions": [], "entities": entities}))
    for t in sorted(per_time):
        entities = [_entity(r, keyframe_interval_ns, frame, t) for r in per_time[t]]
        out.append((t, {"deletions": [], "entities": entities}))
    return sorted(out, key=lambda pair: pair[0])


_COMPRESSION = {
    "": CompressionType.NONE,
    "none": CompressionType.NONE,
    "lz4": CompressionType.LZ4,
    "zstd": CompressionType.ZSTD,
}


def first_message_time(bag: Path, exclude: str | None = None) -> int:
    """The log time of the bag's first message, optionally ignoring one topic.

    The summary's ``message_start_time`` is not usable when the annotation topic
    is the thing being repaired: a save that wrote a bad timestamp becomes the
    bag's reported start, so the fault would be measured against itself.

    Args:
        bag: Recording to read.
        exclude: Topic to skip.

    Returns:
        Nanoseconds, or 0 if the bag has no other messages. Reading stops at the
        first match, so this costs one chunk.
    """
    with bag.open("rb") as handle:
        for _schema, channel, message in make_reader(handle).iter_messages():
            if channel.topic != exclude:
                return message.log_time
    return 0


def _source_compression(summary) -> CompressionType:
    """Pick the compression the source bag already uses.

    Not a cosmetic choice: a reader built without zstd can open an lz4 bag and
    not the other way round, so a rewrite should not change what tools can read
    the recording.

    Args:
        summary: The source bag's summary section.

    Returns:
        The most common chunk compression, or none if the bag is unchunked.
    """
    counts = Counter(index.compression for index in summary.chunk_indexes)
    if not counts:
        return CompressionType.NONE
    name = counts.most_common(1)[0][0]
    return _COMPRESSION.get(name.lower(), CompressionType.ZSTD)


def _source_chunk_size(summary, default: int = 1024 * 1024) -> int:
    """Estimate the chunk size the source bag was written with.

    Chunk boundaries do not change what a file means, but they do change how
    much a reader must decompress to reach one message -- so a rewrite should
    not silently coarsen them. A writer flushes once a chunk passes its
    threshold, which makes the median uncompressed chunk a good estimate of it.

    Args:
        summary: The source bag's summary section.
        default: Used when the bag is unchunked.

    Returns:
        A chunk size in bytes, never below 64 KiB.
    """
    sizes = sorted(index.uncompressed_size for index in summary.chunk_indexes)
    if not sizes:
        return default
    return max(64 * 1024, sizes[len(sizes) // 2])


def _cdr_encoder(datatype: str, msgdef: str):
    """Build the CDR encoder for a ROS2 message definition.

    This is the same function ``mcap_ros2``'s writer uses internally. Going
    through the low-level MCAP writer instead of that one is what lets the
    source bag's header and compression survive the rewrite -- the ros2 writer
    stamps its own profile and library the moment it is constructed.

    Args:
        datatype: Fully qualified message type.
        msgdef: The concatenated ``.msg`` definition.

    Returns:
        A callable turning a message dict into CDR bytes.

    Raises:
        RuntimeError: If the definition does not parse.
    """
    from mcap_ros2._dynamic import serialize_dynamic

    types = serialize_dynamic(datatype, msgdef)
    if datatype not in types:
        msg = f"could not parse the schema for {datatype}"
        raise RuntimeError(msg)
    return types[datatype]


def write_into_bag(
    bag: Path,
    records: list[dict],
    *,
    topic: str = "/annotations/boxes",
    frame: str = "map",
    keyframe_interval_ns: int = 500_000_000,
    output: Path | None = None,
) -> tuple[int, int]:
    """Rewrite ``bag`` with the annotations added on ``topic``.

    Args:
        bag: The recording to add labels to.
        records: Rows from the annotator's exporter.
        topic: Topic to publish the SceneUpdates on.
        frame: Frame the poses are in.
        keyframe_interval_ns: Lifetime of a per-frame entity.
        output: Write here instead of replacing ``bag``.

    Returns:
        ``(messages copied, annotation messages written)``.

    Raises:
        RuntimeError: If the copy lost more than the annotations it replaced; the
            original is left untouched.

    Note:
        Existing messages on ``topic`` are dropped rather than carried over. A
        correction pass re-exports every box it knows about, so keeping the old
        ones would leave two copies of each annotation in the bag.
    """
    target = output or bag
    temp = target.with_suffix(target.suffix + ".partial")

    copied = 0
    replaced = 0
    with bag.open("rb") as source, temp.open("wb") as sink:
        reader = make_reader(source)
        summary = reader.get_summary()
        header = reader.get_header()
        expected = summary.statistics.message_count if summary.statistics else None

        # Not statistics.message_start_time: on a bag saved by an earlier version
        # that is the annotation message's own bad timestamp, and stamping the
        # static boxes with it would carry the fault forward.
        start_time = first_message_time(bag, exclude=topic)

        updates = scene_updates(records, frame, keyframe_interval_ns, start_time)

        low = Writer(
            sink,
            chunk_size=_source_chunk_size(summary),
            compression=_source_compression(summary),
        )
        # The source header is carried over verbatim: the profile decides how a
        # reader interprets the whole file, and adding annotations is no reason
        # to change it.
        low.start(profile=header.profile, library=header.library)

        # The annotation topic is rewritten below, so its channel is not carried
        # over -- and neither is its schema. A previous save's SceneUpdate
        # definition would otherwise survive as a schema record nobody
        # references, and a reader that parses every schema in the file sees two
        # incompatible definitions of the same type.
        kept = [c for c in summary.channels.values() if c.topic != topic]
        referenced = {c.schema_id for c in kept}
        schema_ids = {
            s.id: low.register_schema(name=s.name, encoding=s.encoding, data=s.data)
            for s in summary.schemas.values()
            if s.id in referenced
        }
        channel_ids = {
            c.id: low.register_channel(
                topic=c.topic,
                message_encoding=c.message_encoding,
                schema_id=schema_ids.get(c.schema_id, 0),
                metadata=c.metadata,
            )
            for c in kept
        }

        msgdef = SCHEMA_PATH.read_text()
        encode = _cdr_encoder(SCENE_UPDATE_TYPE, msgdef)
        scene_schema_id = low.register_schema(
            name=SCENE_UPDATE_TYPE,
            encoding=SchemaEncoding.ROS2,
            data=msgdef.encode(),
        )
        scene_channel_id = low.register_channel(
            topic=topic,
            message_encoding=MessageEncoding.CDR,
            schema_id=scene_schema_id,
        )

        def write_update(log_time: int, update: dict, sequence: int) -> None:
            low.add_message(
                channel_id=scene_channel_id,
                log_time=log_time,
                data=encode(update),
                publish_time=log_time,
                sequence=sequence,
            )

        # Merged by time: a reader is entitled to messages in nondecreasing log
        # order, and appending ours at the end would break that.
        pending = iter(updates)
        upcoming = next(pending, None)
        written = 0

        for _schema, channel, message in reader.iter_messages():
            if channel.topic == topic:
                # A previous save's annotations; this export supersedes them.
                replaced += 1
                continue

            while upcoming is not None and upcoming[0] <= message.log_time:
                write_update(upcoming[0], upcoming[1], written)
                written += 1
                upcoming = next(pending, None)

            low.add_message(
                channel_id=channel_ids[channel.id],
                log_time=message.log_time,
                data=message.data,
                publish_time=message.publish_time,
                sequence=message.sequence,
            )
            copied += 1

        while upcoming is not None:
            write_update(upcoming[0], upcoming[1], written)
            written += 1
            upcoming = next(pending, None)

        # Metadata records live outside the message stream and would otherwise be
        # dropped silently.
        for name, values in _metadata_records(bag):
            low.add_metadata(name, values)

        low.finish()

    if expected is not None and copied + replaced != expected:
        temp.unlink(missing_ok=True)
        msg = (
            f"copied {copied} + replaced {replaced} of {expected} messages; "
            f"{bag} left untouched"
        )
        raise RuntimeError(msg)
    if replaced:
        print(f"replaced {replaced} annotation message(s) from a previous save")

    os.replace(temp, target)
    return copied, written


def _metadata_records(bag: Path) -> list[tuple[str, dict]]:
    """Read the bag's metadata records, which the message stream excludes.

    Uses the reader's index rather than a linear scan. Scanning meant parsing
    every record in the file to find two, and any unrelated oddity anywhere in a
    900 MB recording aborted the whole save.

    Args:
        bag: Recording to read.

    Returns:
        ``(name, metadata)`` pairs, empty if they could not be read.
    """
    from mcap.reader import make_reader

    try:
        with bag.open("rb") as handle:
            return [(m.name, m.metadata) for m in make_reader(handle).iter_metadata()]
    except Exception as err:  # noqa: BLE001
        # Losing two metadata records is not worth failing a save over; say so
        # rather than dropping them silently.
        print(f"warning: could not read metadata from {bag}: {err}")
        return []


def load_jsonl(path: Path) -> tuple[list[dict], dict]:
    """Read the annotator's sidecar.

    Args:
        path: A ``.labels.jsonl`` file.

    Returns:
        ``(records, header)``.
    """
    lines = path.read_text().splitlines()
    header = json.loads(lines[0]) if lines else {}
    return [json.loads(line) for line in lines[1:] if line.strip()], header


def read_from_bag(
    bag: Path, topic: str = "/annotations/boxes"
) -> tuple[list[dict], dict[str, int]]:
    """Read annotations back out of a bag's SceneUpdate topic.

    The inverse of :func:`write_into_bag`, and the reason predictions and labels
    share a schema: whatever a model publishes on this topic loads here as
    editable boxes, so correcting a prediction is the same operation as
    correcting a human label.

    Args:
        bag: Recording to read.
        topic: Topic carrying ``foxglove_msgs/SceneUpdate``.

    Returns:
        ``(records, class name to id)`` in the same shape the annotator's
        exporter produces, so both paths feed one loader.
    """
    from mcap_ros2.reader import read_ros2_messages

    records: list[dict] = []
    classes: dict[str, int] = {}
    extra_cubes = 0

    for message in read_ros2_messages(str(bag), topics=[topic]):
        for entity in message.ros_msg.entities:
            if not entity.cubes:
                continue
            if len(entity.cubes) > 1:
                extra_cubes += len(entity.cubes) - 1
            cube = entity.cubes[0]

            metadata = {kv.key: kv.value for kv in entity.metadata}
            class_name = metadata.get("class")
            class_id = metadata.get("class_id")
            class_id = int(class_id) if class_id is not None else None
            if class_name is not None and class_id is not None:
                classes[class_name] = class_id

            # Lifetime zero means forever, which is how a static box is written.
            lifetime_ns = entity.lifetime.sec * 10**9 + entity.lifetime.nanosec
            is_static = lifetime_ns == 0

            stamp = entity.timestamp.sec * 10**9 + entity.timestamp.nanosec
            if stamp == 0:
                stamp = message.log_time_ns

            records.append(
                {
                    "track": entity.id,
                    "t": None if is_static else stamp,
                    "static": is_static,
                    "class_id": class_id,
                    "class": class_name,
                    "center": [
                        cube.pose.position.x,
                        cube.pose.position.y,
                        cube.pose.position.z,
                    ],
                    # SceneUpdate carries full extents; everything downstream
                    # works in half-extents.
                    "half_size": [cube.size.x / 2, cube.size.y / 2, cube.size.z / 2],
                    "quat": [
                        cube.pose.orientation.x,
                        cube.pose.orientation.y,
                        cube.pose.orientation.z,
                        cube.pose.orientation.w,
                    ],
                }
            )

    if extra_cubes:
        print(
            f"note: ignored {extra_cubes} extra cube(s); one box per entity is "
            "assumed, since the track id identifies a single object"
        )

    return records, classes
