"""Write the annotator's exported labels into the bag itself.

The annotator's Export button writes a ``<bag>.labels.jsonl`` sidecar; this puts
those labels into the recording as ``foxglove_msgs/SceneUpdate`` on their own
topic, so the bag carries its own annotations and Foxglove renders them without
configuration.

In place by default::

    python tools/annotator/save_labels.py

Annotations already in a bag can be rewritten instead of read from a sidecar,
which is how a bag saved by an older version is rebuilt on top of the untouched
original::

    python tools/annotator/save_labels.py --bag original.mcap \
        --from-bag labelled.mcap --output labelled.mcap

The bag is rewritten (MCAP keeps its index in a trailer, so there is no append),
which takes a few seconds for a 900 MB recording. The new file is built beside
the original and renamed over it only after its message count is verified, so an
interrupted save cannot leave a truncated recording.
"""

from __future__ import annotations

import argparse
import itertools
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from mcap_labels import (
    first_message_time,
    load_jsonl,
    read_from_bag,
    write_into_bag,
)

DEFAULT_BAG = Path(
    "/home/sschlaepfer/docker-data-exchange/rosbags/rosbag2_2026_08_02-19_29_27_0.mcap"
)


def keyframe_interval_ns(records: list[dict], fallback: int = 500_000_000) -> int:
    """Infer the keyframe spacing from the records themselves.

    The lifetime written for a per-frame box is this interval. Hard-coding it
    would make boxes blink whenever the feed ran at a different ``--hz``.

    Args:
        records: Annotation rows.
        fallback: Used when there are too few distinct timestamps to measure.

    Returns:
        The median gap between consecutive timestamps, in nanoseconds.
    """
    times = sorted({int(r["t"]) for r in records if r.get("t") is not None})
    if len(times) < 2:
        return fallback
    gaps = [b - a for a, b in itertools.pairwise(times) if b > a]
    return int(statistics.median(gaps)) if gaps else fallback


def _to_absolute(bag: Path, records: list[dict], topic: str) -> int:
    """Repair timestamps written before the feed's timeline was absolute.

    An earlier version indexed the viewer timeline by time since the start of
    the bag, so exported timestamps were a few seconds after the Unix epoch.
    The offset is recoverable -- it was the bag's own start time -- so those
    records are shifted in place rather than lost.

    Args:
        bag: Recording the records came from.
        records: Rows to fix, modified in place.
        topic: The annotation topic, excluded when locating the start. Its own
            bad timestamp is the bag's reported start time, so trusting the
            summary here would compare the fault against itself.

    Returns:
        How many records were shifted.
    """
    start = first_message_time(bag, exclude=topic)
    if not start:
        return 0

    # A real timestamp is inside the recording; anything far below its start is
    # a duration that was mistaken for one.
    stale = [r for r in records if r.get("t") is not None and r["t"] < start]
    for record in stale:
        record["t"] += start
    return len(stale)


def main() -> None:
    """Write a sidecar's labels into its bag.

    Raises:
        SystemExit: If there are no annotations to write -- no sidecar, an empty
            sidecar, or ``--from-bag`` on a bag with none.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bag", type=Path, default=DEFAULT_BAG)
    parser.add_argument(
        "--labels",
        type=Path,
        default=None,
        help="Sidecar to read. Defaults to <bag>.labels.jsonl.",
    )
    parser.add_argument("--topic", default="/annotations/boxes")
    parser.add_argument(
        "--from-bag",
        nargs="?",
        const="-",
        default=None,
        metavar="BAG",
        help="Take the annotations from a bag instead of a sidecar. With no "
        "value, from --bag itself. Give a path to read the labels from one bag "
        "and copy the messages from another -- which is how a bag saved by an "
        "older version is rebuilt on top of the pristine original.",
    )
    parser.add_argument("--frame", default="map")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write here instead of replacing the bag.",
    )
    args = parser.parse_args()

    if args.from_bag is not None:
        source = args.bag if args.from_bag == "-" else Path(args.from_bag)
        records, _ = read_from_bag(source, args.topic)
        if not records:
            print(f"{source} has no annotations on {args.topic}")
            raise SystemExit(1)
        # read_from_bag reports a static box as t=None, which is what a static
        # box means here too, so the round trip preserves the distinction.
        header = {}
        shifted = _to_absolute(source, records, args.topic)
        if shifted:
            print(f"shifted {shifted} record(s) from bag-relative to absolute time")
        print(f"re-writing {len(records)} record(s) from {source}")
    else:
        sidecar = args.labels or args.bag.with_name(f"{args.bag.stem}.labels.jsonl")
        if not sidecar.exists():
            print(f"no sidecar at {sidecar}; export from the annotator first")
            raise SystemExit(1)

        records, header = load_jsonl(sidecar)
        if not records:
            print(f"{sidecar} has no annotations")
            raise SystemExit(1)

    frame = header.get("frame", args.frame)
    interval = keyframe_interval_ns(records)
    tracks = len({r["track"] for r in records})
    print(
        f"{len(records)} record(s), {tracks} track(s), frame={frame}, "
        f"lifetime={interval / 1e9:.2f}s"
    )

    copied, written = write_into_bag(
        args.bag,
        records,
        topic=args.topic,
        frame=frame,
        keyframe_interval_ns=interval,
        output=args.output,
    )
    target = args.output or args.bag
    print(f"copied {copied} message(s), added {written} on {args.topic} -> {target}")

    # Read it back rather than trusting the write: this is the user's recording.
    back, classes = read_from_bag(target, args.topic)
    print(f"verified {len(back)} record(s) readable, classes={classes}")


if __name__ == "__main__":
    main()
