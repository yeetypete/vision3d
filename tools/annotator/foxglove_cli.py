"""Browse Foxglove recordings, pull one to annotate, push the labels back.

The annotator's workflow, without the local directory::

    python tools/annotator/foxglove_cli.py login --key fox_sk_...
    python tools/annotator/foxglove_cli.py list
    python tools/annotator/foxglove_cli.py pull <recording>      # -> local mcap
    python tools/annotator/foxglove_cli.py push <recording>      # <- sidecar
    python tools/annotator/foxglove_cli.py approve <recording>

``list`` shows where each recording sits in the labelling cycle: unlabelled,
annotated, or approved. That state lives in the sidecar's own MCAP metadata, so
it travels with the labels rather than in a database beside them.

The API key is org-scoped -- see :mod:`foxglove_api` on what that means for
external annotators.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from foxglove_api import (
    Client,
    FoxgloveError,
    Status,
    save_api_key,
    sidecar_name,
)
from mcap_labels import (
    ANNOTATION_TOPIC,
    first_message_time,
    read_annotations,
    write_sidecar,
)
from save_labels import keyframe_interval_ns

#: Where pulled recordings are cached.
DEFAULT_CACHE = Path.home() / ".cache" / "vision3d" / "foxglove"

#: Marks in the listing, so the cycle reads at a glance.
MARKS = {
    Status.UNLABELLED: " ",
    Status.ANNOTATED: "~",
    Status.APPROVED: "+",
}


def human_size(n: int) -> str:
    """Render a byte count compactly.

    Args:
        n: Bytes.

    Returns:
        A short string such as ``1.3 GB``.
    """
    size = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def find(client: Client, wanted: str, **filters):
    """Look up one recording by id, key, or filename.

    Args:
        client: API client.
        wanted: Id, key, or filename of the recording.
        **filters: Narrowing passed to the catalogue.

    Returns:
        The matching :class:`~foxglove_api.Item`.

    Raises:
        SystemExit: If nothing matches, or the name is ambiguous.
    """
    items = client.catalogue(**filters)
    matches = [
        item
        for item in items
        if wanted in (item.recording.id, item.recording.key, item.recording.name)
    ]
    if not matches:
        print(f"no recording matching {wanted!r}; try `list`")
        raise SystemExit(1)
    if len(matches) > 1:
        print(f"{wanted!r} matches {len(matches)} recordings; use the id")
        raise SystemExit(1)
    return matches[0]


def cmd_login(args: argparse.Namespace) -> None:
    """Save an API key for later runs.

    Args:
        args: Parsed arguments.
    """
    path = save_api_key(args.key)
    Client()  # Fail now rather than at first use.
    print(f"key saved to {path} (owner-only) and accepted")


def cmd_list(args: argparse.Namespace) -> None:
    """Show recordings and where each sits in the labelling cycle.

    Args:
        args: Parsed arguments.
    """
    client = Client(base_url=args.base_url)
    filters = {"limit": args.limit}
    if args.project:
        filters["projectId"] = args.project
    if args.device:
        filters["deviceId"] = args.device

    items = client.catalogue(**filters)
    if args.status:
        items = [i for i in items if i.status.value == args.status]
    if not items:
        print("no recordings")
        return

    width = max(len(i.recording.name) for i in items)
    print(f"  {'recording':{width}}  {'size':>9}  {'status':10}  device")
    for item in items:
        r = item.recording
        print(
            f"{MARKS[item.status]} {r.name:{width}}  {human_size(r.size):>9}  "
            f"{item.status.value:10}  {r.device_name or '-'}"
        )
    counts = {s: sum(1 for i in items if i.status is s) for s in Status}
    print(
        f"\n{len(items)} recording(s): "
        + ", ".join(f"{n} {s.value}" for s, n in counts.items() if n)
    )


def cmd_pull(args: argparse.Namespace) -> None:
    """Download a recording, and its sidecar if one exists.

    Args:
        args: Parsed arguments.
    """
    client = Client(base_url=args.base_url)
    item = find(client, args.recording)
    cache = args.out or DEFAULT_CACHE
    cache.mkdir(parents=True, exist_ok=True)

    target = cache / item.recording.name
    print(f"pulling {item.recording.name} ({human_size(item.recording.size)})…")
    client.download(
        target,
        recording_id=item.recording.id,
        start=args.start,
        end=args.end,
        topics=args.topic or None,
    )
    print(f"  -> {target} ({human_size(target.stat().st_size)})")

    if item.sidecar is not None:
        beside = cache / sidecar_name(item.recording.name)
        client.download(beside, recording_id=item.sidecar.id)
        print(f"  -> {beside} ({item.status.value})")
    else:
        print("  no sidecar yet; this recording is unlabelled")

    print(f"\nannotate it with:\n  cargo run -- {target}")


def cmd_push(args: argparse.Namespace) -> None:
    """Upload a local sidecar, replacing any earlier one.

    Args:
        args: Parsed arguments.

    Raises:
        SystemExit: If there is no sidecar to upload, or it is empty.
    """
    client = Client(base_url=args.base_url)
    item = find(client, args.recording)

    local = args.sidecar or (args.out or DEFAULT_CACHE) / sidecar_name(
        item.recording.name
    )
    if not local.exists():
        print(f"no sidecar at {local}; annotate and save first")
        raise SystemExit(1)

    records, _ = read_annotations(local, ANNOTATION_TOPIC)
    if not records:
        print(f"{local} holds no annotations")
        raise SystemExit(1)

    # Re-stamped rather than uploaded as-is: the status is metadata inside the
    # file, and this is the moment it changes.
    write_sidecar(
        local,
        records,
        keyframe_interval_ns=keyframe_interval_ns(records),
        source_bag=Path(item.recording.name),
        start_time_ns=first_message_time(local, exclude=ANNOTATION_TOPIC) or None,
        status=args.status,
    )

    print(f"uploading {local.name} ({len(records)} record(s), {args.status})…")
    client.upload(
        local,
        device_id=item.recording.device_id,
        project_id=args.project,
    )
    print("  uploaded; Foxglove will import it shortly")

    # Only once the new one is safely up.
    if item.sidecar is not None and not args.keep_old:
        client.delete_recording(item.sidecar.id)
        print(f"  retired the previous sidecar ({item.sidecar.id})")


def cmd_approve(args: argparse.Namespace) -> None:
    """Mark a recording's labels reviewed and approved.

    The status lives inside the sidecar, so approving re-stamps and re-uploads
    it. That needs the file locally; pull it first.

    Args:
        args: Parsed arguments.
    """
    args.status = Status.APPROVED.value
    cmd_push(args)


def main() -> None:
    """Dispatch a subcommand.

    Raises:
        SystemExit: If the command fails.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-url",
        default="https://api.foxglove.dev",
        help="API root. Point at a proxy to enforce per-annotator access.",
    )
    subs = parser.add_subparsers(dest="command", required=True)

    login = subs.add_parser("login", help="save an API key")
    login.add_argument("--key", required=True)
    login.set_defaults(func=cmd_login)

    listing = subs.add_parser("list", help="show recordings and their status")
    listing.add_argument("--project")
    listing.add_argument("--device")
    listing.add_argument("--limit", type=int, default=100)
    listing.add_argument("--status", choices=[s.value for s in Status])
    listing.set_defaults(func=cmd_list)

    pull = subs.add_parser("pull", help="download a recording to annotate")
    pull.add_argument("recording")
    pull.add_argument("--out", type=Path)
    pull.add_argument("--start", help="RFC 3339; server-side window")
    pull.add_argument("--end", help="RFC 3339; server-side window")
    pull.add_argument("--topic", action="append", help="repeatable")
    pull.set_defaults(func=cmd_pull)

    push = subs.add_parser("push", help="upload the sidecar")
    push.add_argument("recording")
    push.add_argument("--sidecar", type=Path)
    push.add_argument("--out", type=Path)
    push.add_argument("--project")
    push.add_argument(
        "--status", default=Status.ANNOTATED.value, choices=[s.value for s in Status]
    )
    push.add_argument(
        "--keep-old", action="store_true", help="leave the previous sidecar in place"
    )
    push.set_defaults(func=cmd_push)

    approve = subs.add_parser("approve", help="mark the labels reviewed")
    approve.add_argument("recording")
    approve.add_argument("--sidecar", type=Path)
    approve.add_argument("--out", type=Path)
    approve.add_argument("--project")
    approve.add_argument("--keep-old", action="store_true")
    approve.set_defaults(func=cmd_approve)

    args = parser.parse_args()
    try:
        args.func(args)
    except FoxgloveError as err:
        print(f"error: {err}")
        raise SystemExit(1) from err


if __name__ == "__main__":
    main()
