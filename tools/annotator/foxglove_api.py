"""Read recordings from Foxglove and write label sidecars back to it.

The annotator's local flow is unchanged: a recording goes in, a
``<name>.labels.mcap`` sidecar comes out, and the recording itself is never
modified. This module only changes where those two files live -- a Foxglove
organisation rather than a directory.

**On access.** Foxglove's API authenticates organisations, not people: keys are
created by org admins and carry the whole org's data with them. An annotator's
key can therefore read every recording in the organisation, whatever this tool
chooses to show them. The recording list here is a convenience, not a boundary.
If that matters, put a service in front that holds the org key and enforces a
per-annotator allow-list, and point ``base_url`` at it -- nothing else here has
to change.

Review state lives in the sidecar's own MCAP metadata, which Foxglove indexes,
so ``metadataQuery`` filters it server-side rather than this tool downloading
files to find out.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import requests

#: Foxglove's public API.
BASE_URL = "https://api.foxglove.dev"

#: Suffix marking a recording as this tool's labels rather than sensor data.
SIDECAR_SUFFIX = ".labels.mcap"

#: MCAP metadata record the sidecar writer stamps, and the keys within it.
METADATA_RECORD = "vision3d.annotations"
SOURCE_KEY = "source_bag"
STATUS_KEY = "status"

#: Where an API key is remembered between runs.
CONFIG_PATH = Path.home() / ".config" / "vision3d" / "foxglove.json"


class FoxgloveError(RuntimeError):
    """An API call failed, or was not configured."""


class Status(str, Enum):
    """Where a recording sits in the labelling cycle."""

    UNLABELLED = "unlabelled"
    ANNOTATED = "annotated"
    APPROVED = "approved"


@dataclass(frozen=True)
class Recording:
    """One recording as Foxglove describes it."""

    id: str
    key: str
    path: str
    size: int
    start: str | None
    end: str | None
    import_status: str
    device_id: str | None
    device_name: str | None
    metadata: dict[str, dict[str, str]]

    @property
    def name(self) -> str:
        """The recording's filename, without its directory."""
        return Path(self.path).name

    @property
    def is_sidecar(self) -> bool:
        """Whether this recording holds labels rather than sensor data."""
        return self.name.endswith(SIDECAR_SUFFIX)

    @property
    def annotations(self) -> dict[str, str]:
        """This tool's metadata record, empty if the recording has none."""
        return self.metadata.get(METADATA_RECORD, {})

    @property
    def duration_s(self) -> float | None:
        """Length in seconds, or None if the recording has no time range."""
        if not (self.start and self.end):
            return None
        try:
            began = datetime.fromisoformat(self.start)
            ended = datetime.fromisoformat(self.end)
        except ValueError:
            return None
        return (ended - began).total_seconds()


@dataclass(frozen=True)
class Item:
    """A recording together with its labelling state."""

    recording: Recording
    sidecar: Recording | None
    status: Status


def sidecar_name(recording_name: str) -> str:
    """The sidecar filename for a recording.

    Args:
        recording_name: The recording's filename, e.g. ``run.mcap``.

    Returns:
        ``run.labels.mcap``.
    """
    stem = recording_name.removesuffix(".mcap")
    return f"{stem}{SIDECAR_SUFFIX}"


def load_api_key(explicit: str | None = None) -> str:
    """Find the API key to authenticate with.

    Checked in order: the argument, ``FOXGLOVE_API_KEY``, then the saved
    config. Keys are org-scoped and admin-issued; see the module docstring.

    Args:
        explicit: A key supplied directly, which wins over everything else.

    Returns:
        The key.

    Raises:
        FoxgloveError: If no key is configured.
    """
    if explicit:
        return explicit
    if key := os.environ.get("FOXGLOVE_API_KEY"):
        return key
    if CONFIG_PATH.exists():
        try:
            saved = json.loads(CONFIG_PATH.read_text()).get("api_key")
        except (OSError, json.JSONDecodeError) as err:
            msg = f"could not read {CONFIG_PATH}: {err}"
            raise FoxgloveError(msg) from err
        if saved:
            return saved
    msg = (
        "no Foxglove API key: set FOXGLOVE_API_KEY, pass one in, or run "
        "`foxglove_cli.py login`"
    )
    raise FoxgloveError(msg)


def save_api_key(key: str, path: Path = CONFIG_PATH) -> Path:
    """Remember an API key for later runs.

    Written with owner-only permissions, since it grants access to the whole
    organisation's recordings.

    Args:
        key: The API key.
        path: Where to write it.

    Returns:
        The path written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"api_key": key}, indent=2) + "\n")
    path.chmod(0o600)
    return path


def _parse_metadata(raw: Any) -> dict[str, dict[str, str]]:
    """Normalise the metadata Foxglove reports on a recording.

    The API returns MCAP metadata records, which are named key/value maps. The
    exact shape has varied, so both a list of records and an already-flattened
    mapping are accepted.

    Args:
        raw: Whatever the API put in the ``metadata`` field.

    Returns:
        Record name to its key/value pairs.
    """
    if isinstance(raw, dict):
        # Either {record: {k: v}} or a single flat record.
        if all(isinstance(v, dict) for v in raw.values()):
            return {str(k): dict(v) for k, v in raw.items()}
        return {METADATA_RECORD: {str(k): str(v) for k, v in raw.items()}}

    out: dict[str, dict[str, str]] = {}
    for entry in raw or ():
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("name", METADATA_RECORD))
        values = entry.get("metadata", entry)
        if isinstance(values, dict):
            out[name] = {str(k): str(v) for k, v in values.items() if k != "name"}
    return out


def _recording(payload: dict) -> Recording:
    """Build a :class:`Recording` from one API object.

    Args:
        payload: A recording object from the API.

    Returns:
        The parsed recording.
    """
    device = payload.get("device") or {}
    return Recording(
        id=str(payload.get("id", "")),
        key=str(payload.get("key", "")),
        path=str(payload.get("path", "")),
        size=int(payload.get("size") or 0),
        start=payload.get("start"),
        end=payload.get("end"),
        import_status=str(payload.get("importStatus", "")),
        device_id=device.get("id"),
        device_name=device.get("name"),
        metadata=_parse_metadata(payload.get("metadata")),
    )


class Client:
    """Talks to one Foxglove organisation.

    Args:
        api_key: Key to authenticate with; discovered if omitted.
        base_url: API root. Point this at a proxy to enforce per-annotator
            access, which Foxglove's own keys cannot express.
        timeout: Seconds to wait on a request, excluding bulk transfers.
    """

    def __init__(
        self,
        api_key: str | None = None,
        *,
        base_url: str = BASE_URL,
        timeout: float = 60.0,
    ) -> None:
        self._key = load_api_key(api_key)
        self._base = base_url.rstrip("/")
        self._timeout = timeout
        self._session = requests.Session()
        self._session.headers.update(
            {
                "Authorization": f"Bearer {self._key}",
                "Accept": "application/json",
            }
        )

    def _url(self, path: str) -> str:
        return f"{self._base}/v1/{path.lstrip('/')}"

    def _check(self, response: requests.Response, what: str) -> None:
        """Raise with the server's own explanation, which is usually specific.

        Args:
            response: The response to check.
            what: What was being attempted, for the message.

        Raises:
            FoxgloveError: If the response is an error.
        """
        if response.ok:
            return
        detail = response.text.strip()
        try:
            body = response.json()
            detail = body.get("error") or body.get("message") or detail
        except ValueError:
            pass
        hint = ""
        if response.status_code in (401, 403):
            hint = " (is the API key valid, and does it have the right capability?)"
        msg = f"{what} failed: HTTP {response.status_code}{hint}: {detail[:400]}"
        raise FoxgloveError(msg)

    def recordings(self, **filters: Any) -> list[Recording]:
        """List recordings, newest first.

        Args:
            **filters: Passed through as query parameters -- ``deviceId``,
                ``projectId``, ``start``, ``end``, ``topic``, ``metadataQuery``,
                ``limit``, ``offset`` and the rest of the documented set.

        Returns:
            The recordings the key can see.
        """
        params = {"sortBy": "createdAt", "sortOrder": "desc", **filters}
        response = self._session.get(
            self._url("recordings"), params=params, timeout=self._timeout
        )
        self._check(response, "listing recordings")
        return [_recording(item) for item in response.json()]

    def catalogue(self, **filters: Any) -> list[Item]:
        """List recordings paired with their labelling state.

        Sidecars are matched to their recording by the ``source_bag`` the
        sidecar writer stamps, falling back to the filename convention for
        sidecars written by hand.

        Args:
            **filters: As :meth:`recordings`.

        Returns:
            One item per non-sidecar recording, newest first.
        """
        everything = self.recordings(**filters)
        sources = [r for r in everything if not r.is_sidecar]
        sidecars = [r for r in everything if r.is_sidecar]

        # Newest first, and `setdefault` keeps the first: approving re-uploads
        # the sidecar, so a source can briefly have more than one and the most
        # recent is the one that counts.
        by_source: dict[str, Recording] = {}
        for sidecar in sidecars:
            named = sidecar.annotations.get(SOURCE_KEY)
            stem = sidecar.name.removesuffix(SIDECAR_SUFFIX)
            by_source.setdefault(named or f"{stem}.mcap", sidecar)

        items = []
        for source in sources:
            sidecar = by_source.get(source.name)
            if sidecar is None:
                status = Status.UNLABELLED
            else:
                raw = sidecar.annotations.get(STATUS_KEY, Status.ANNOTATED.value)
                try:
                    status = Status(raw)
                except ValueError:
                    status = Status.ANNOTATED
            items.append(Item(recording=source, sidecar=sidecar, status=status))
        return items

    def delete_recording(self, recording_id: str) -> None:
        """Delete a recording and everything attached to it.

        Used to retire a superseded sidecar after a new one is safely up --
        never the other way round, so labels cannot be lost to a failed upload.

        Args:
            recording_id: The recording to delete.
        """
        response = self._session.delete(
            self._url(f"recordings/{recording_id}"), timeout=self._timeout
        )
        self._check(response, "deleting a recording")

    def download(
        self,
        out: Path,
        *,
        recording_id: str | None = None,
        recording_key: str | None = None,
        device_id: str | None = None,
        start: str | None = None,
        end: str | None = None,
        topics: list[str] | None = None,
        chunk_bytes: int = 1 << 22,
    ) -> Path:
        """Fetch recording data as MCAP.

        The server applies the topic and time filters, so a windowed load costs
        only the bytes it needs -- the same saving the local reader gets from
        the chunk index, without the round trips.

        Args:
            out: File to write. Replaced atomically.
            recording_id: Recording to fetch, by id.
            recording_key: Recording to fetch, by key.
            device_id: Device to fetch from; needs ``start`` and ``end``.
            start: Inclusive start, RFC 3339.
            end: Inclusive end, RFC 3339.
            topics: Restrict to these topics.
            chunk_bytes: Streaming buffer size.

        Returns:
            The path written.

        Raises:
            FoxgloveError: If no source is given, or the request fails.
        """
        body: dict[str, Any] = {}
        if recording_id:
            body["recordingId"] = recording_id
        elif recording_key:
            body["recordingKey"] = recording_key
        elif device_id:
            body["deviceId"] = device_id
        else:
            msg = "download needs a recording id, a recording key, or a device id"
            raise FoxgloveError(msg)

        if start:
            body["start"] = start
        if end:
            body["end"] = end
        if topics:
            body["topics"] = topics
        body["outputFormat"] = "mcap0"

        response = self._session.post(
            # `data/stream`, not `data/download`: the latter 404s.
            self._url("data/stream"),
            json=body,
            timeout=self._timeout,
            stream=True,
            allow_redirects=True,
        )
        self._check(response, "downloading data")

        temp = out.with_suffix(out.suffix + ".partial")
        temp.parent.mkdir(parents=True, exist_ok=True)
        written = 0
        with temp.open("wb") as sink:
            for chunk in response.iter_content(chunk_size=chunk_bytes):
                sink.write(chunk)
                written += len(chunk)
        if written == 0:
            temp.unlink(missing_ok=True)
            msg = "the download returned no data; check the time range and topics"
            raise FoxgloveError(msg)
        os.replace(temp, out)
        return out

    def upload(
        self,
        path: Path,
        *,
        device_id: str | None = None,
        device_name: str | None = None,
        key: str | None = None,
        project_id: str | None = None,
    ) -> dict:
        """Upload a file, which Foxglove imports as its own recording.

        Two steps, as the API requires: ask for a signed link, then PUT the
        bytes at it.

        Args:
            path: File to upload.
            device_id: Device to attach it to.
            device_name: Device to attach it to, by name.
            key: User-specified recording key.
            project_id: Project to import into.

        Returns:
            The API's response to the link request, which names the recording.

        Raises:
            FoxgloveError: If no owner is given, or either step fails.
        """
        if not (device_id or device_name or key):
            msg = "upload needs one of device_id, device_name or key"
            raise FoxgloveError(msg)

        body: dict[str, Any] = {"filename": path.name}
        if device_id:
            body["deviceId"] = device_id
        if device_name:
            body["deviceName"] = device_name
        if key:
            body["key"] = key
        if project_id:
            body["projectId"] = project_id

        response = self._session.post(
            self._url("data/upload"), json=body, timeout=self._timeout
        )
        self._check(response, "requesting an upload link")
        granted = response.json()

        link = granted.get("link") or granted.get("url")
        if not link:
            msg = f"the upload link response had no link: {granted}"
            raise FoxgloveError(msg)

        with path.open("rb") as source:
            put = requests.put(
                link,
                data=source,
                headers={
                    "Content-Type": "application/octet-stream",
                    "Content-Length": str(path.stat().st_size),
                },
                timeout=None,
            )
        self._check(put, "uploading the file")
        return granted
