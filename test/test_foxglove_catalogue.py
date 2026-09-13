"""Pairing Foxglove recordings with their label sidecars.

The three-state workflow is derived, not stored: a recording is unlabelled when
no sidecar exists for it, and otherwise takes the status the sidecar carries in
its own MCAP metadata. Getting the pairing wrong would show an annotator work
that is already done, or hide work that is not.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "tools" / "annotator"))

foxglove_api = pytest.importorskip("foxglove_api")

Client = foxglove_api.Client
Status = foxglove_api.Status


def recording(name: str, **metadata: str):
    """Build a recording as the API would report it.

    Args:
        name: Filename.
        **metadata: Entries for this tool's metadata record.

    Returns:
        A :class:`foxglove_api.Recording`.
    """
    payload = {
        "id": f"id-{name}",
        "key": f"key-{name}",
        "path": f"/lake/{name}",
        "size": 1024,
        "importStatus": "complete",
        "device": {"id": "dev-1", "name": "machine"},
        "metadata": (
            [{"name": foxglove_api.METADATA_RECORD, "metadata": dict(metadata)}]
            if metadata
            else []
        ),
    }
    return foxglove_api._recording(payload)


def client_over(recordings: list) -> Client:
    """A client that returns a fixed listing instead of calling the API.

    Args:
        recordings: What :meth:`Client.recordings` should return.

    Returns:
        The client.
    """
    client = Client.__new__(Client)
    client.recordings = lambda **_: list(recordings)  # type: ignore[method-assign]
    return client


def test_sidecar_name_matches_the_local_convention() -> None:
    assert foxglove_api.sidecar_name("run.mcap") == "run.labels.mcap"


def test_a_recording_without_a_sidecar_is_unlabelled() -> None:
    items = client_over([recording("run.mcap")]).catalogue()
    assert [i.status for i in items] == [Status.UNLABELLED]
    assert items[0].sidecar is None


@pytest.mark.parametrize("status", ["annotated", "approved"])
def test_the_sidecar_carries_the_status(status: str) -> None:
    items = client_over(
        [
            recording("run.mcap"),
            recording("run.labels.mcap", source_bag="run.mcap", status=status),
        ]
    ).catalogue()

    assert len(items) == 1, "the sidecar was listed as a recording to annotate"
    assert items[0].status is Status(status)
    assert items[0].sidecar is not None


def test_a_sidecar_without_a_status_counts_as_annotated() -> None:
    """Its presence is the evidence; only approval needs saying."""
    items = client_over(
        [recording("run.mcap"), recording("run.labels.mcap", source_bag="run.mcap")]
    ).catalogue()
    assert items[0].status is Status.ANNOTATED


def test_a_sidecar_is_matched_by_filename_when_metadata_is_missing() -> None:
    """One written by hand, or by an older version, still pairs up."""
    items = client_over(
        [recording("run.mcap"), recording("run.labels.mcap")]
    ).catalogue()
    assert items[0].sidecar is not None
    assert items[0].status is Status.ANNOTATED


def test_the_newest_sidecar_wins() -> None:
    """Approving re-uploads, so a recording can briefly have two.

    The listing is newest-first, and the newer one is the truth.
    """
    items = client_over(
        [
            recording("run.mcap"),
            recording("run.labels.mcap", source_bag="run.mcap", status="approved"),
            recording("run.labels.mcap", source_bag="run.mcap", status="annotated"),
        ]
    ).catalogue()
    assert items[0].status is Status.APPROVED


def test_metadata_is_parsed_from_either_shape() -> None:
    """The API has reported metadata as a list of records and as a mapping."""
    as_list = foxglove_api._parse_metadata(
        [{"name": "vision3d.annotations", "metadata": {"status": "approved"}}]
    )
    as_map = foxglove_api._parse_metadata(
        {"vision3d.annotations": {"status": "approved"}}
    )
    assert as_list == as_map == {"vision3d.annotations": {"status": "approved"}}
