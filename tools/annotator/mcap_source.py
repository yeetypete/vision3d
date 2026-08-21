"""Read a ROS 2 MCAP rosbag directly: transforms, merged lidar, and video.

No intermediate dataset. Everything comes from the bag: ``/tf`` and
``/tf_static`` for geometry, ``sensor_msgs/PointCloud2`` for the lidars, and
``sensor_msgs/CameraInfo`` plus ``foxglove_msgs/CompressedVideo`` for the
cameras.

Requires ``mcap`` and ``mcap-ros2-support``.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field

import numpy as np


def transform_matrix(translation, rotation) -> np.ndarray:
    """Build a 4x4 from a ROS ``Vector3`` and ``Quaternion``.

    Args:
        translation: Object with ``x``/``y``/``z``.
        rotation: Object with ``x``/``y``/``z``/``w``.

    Returns:
        A ``[4, 4]`` homogeneous transform.
    """
    x, y, z, w = rotation.x, rotation.y, rotation.z, rotation.w
    n = x * x + y * y + z * z + w * w
    s = 0.0 if n == 0.0 else 2.0 / n

    wx, wy, wz = s * w * x, s * w * y, s * w * z
    xx, xy, xz = s * x * x, s * x * y, s * x * z
    yy, yz, zz = s * y * y, s * y * z, s * z * z

    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = [
        [1.0 - (yy + zz), xy - wz, xz + wy],
        [xy + wz, 1.0 - (xx + zz), yz - wx],
        [xz - wy, yz + wx, 1.0 - (xx + yy)],
    ]
    out[:3, 3] = [translation.x, translation.y, translation.z]
    return out


@dataclass
class TransformTree:
    """A time-indexed TF graph supporting lookups between any two frames.

    Each edge keeps its samples in arrival order and a lookup takes the newest
    sample at or before the query time -- the same "latest known" rule tf2 uses
    when not interpolating. ``/tf`` here runs at roughly 300 Hz, so the staleness
    that introduces is well under a lidar frame.
    """

    #: (parent, child) -> [(time_ns, matrix)]
    edges: dict[tuple[str, str], list[tuple[int, np.ndarray]]] = field(
        default_factory=lambda: defaultdict(list)
    )
    #: frame -> neighbours, as (other_frame, parent_first) pairs
    neighbours: dict[str, set[tuple[str, bool]]] = field(
        default_factory=lambda: defaultdict(set)
    )

    def add(self, parent: str, child: str, time_ns: int, matrix: np.ndarray) -> None:
        """Record one transform sample.

        Args:
            parent: Parent frame id.
            child: Child frame id.
            time_ns: Stamp of this sample.
            matrix: ``[4, 4]`` ``parent_from_child`` transform.
        """
        self.edges[parent, child].append((time_ns, matrix))
        self.neighbours[parent].add((child, True))
        self.neighbours[child].add((parent, False))

    def _edge_at(self, parent: str, child: str, time_ns: int) -> np.ndarray | None:
        samples = self.edges.get((parent, child))
        if not samples:
            return None
        # Static transforms carry a single sample, often stamped 0; accept it
        # whatever the query time.
        if len(samples) == 1:
            return samples[0][1]
        best = None
        for stamp, matrix in samples:
            if stamp <= time_ns:
                best = matrix
            else:
                break
        return best if best is not None else samples[0][1]

    def lookup(self, target: str, source: str, time_ns: int) -> np.ndarray | None:
        """Resolve a transform between two frames.

        Breadth-first over the undirected graph, inverting an edge when it is
        traversed against its parent-to-child direction.

        Args:
            target: Frame to express the result in.
            source: Frame being transformed from.
            time_ns: Query time.

        Returns:
            ``target_from_source`` as a ``[4, 4]``, or ``None`` if the frames are
            not connected.
        """
        if target == source:
            return np.eye(4)

        queue = deque([(source, np.eye(4))])
        seen = {source}
        while queue:
            frame, so_far = queue.popleft()
            for other, parent_first in self.neighbours.get(frame, ()):
                if other in seen:
                    continue
                # Invariant: `so_far` is source_from_frame, so each step must
                # supply frame_from_other and post-multiply.
                if parent_first:
                    # `frame` is the parent of `other`, so the edge already is
                    # frame_from_other.
                    step = self._edge_at(frame, other, time_ns)
                else:
                    # The edge is other_from_frame; invert it.
                    edge = self._edge_at(other, frame, time_ns)
                    step = None if edge is None else np.linalg.inv(edge)
                if step is None:
                    continue

                combined = so_far @ step
                if other == target:
                    return np.linalg.inv(combined)
                seen.add(other)
                queue.append((other, combined))
        return None


#: PointCloud2 datatype ids we care about.
_FLOAT32 = 7


def pointcloud_xyz_reflectivity(msg) -> tuple[np.ndarray, np.ndarray | None]:
    """Extract xyz and reflectivity from a ``sensor_msgs/PointCloud2``.

    Reads straight out of the raw buffer using the declared field offsets rather
    than assuming a layout, since Livox clouds interleave ``offset_time``,
    ``tag`` and ``line`` between the values we want.

    Args:
        msg: A decoded ``PointCloud2`` message.

    Returns:
        ``(xyz [N, 3] float32, reflectivity [N] uint8 or None)``, with
        non-finite points dropped.

    Raises:
        ValueError: If a coordinate field is not float32.
    """
    raw = np.frombuffer(msg.data, dtype=np.uint8)
    stride = msg.point_step
    raw = raw[: (len(raw) // stride) * stride].reshape(-1, stride)

    offsets = {f.name: (f.offset, f.datatype) for f in msg.fields}
    xyz = np.empty((raw.shape[0], 3), dtype=np.float32)
    for i, name in enumerate("xyz"):
        offset, datatype = offsets[name]
        if datatype != _FLOAT32:
            raise ValueError(f"{name} is datatype {datatype}, expected float32")
        xyz[:, i] = raw[:, offset : offset + 4].copy().view(np.float32).ravel()

    reflectivity = None
    if "reflectivity" in offsets:
        offset, _ = offsets["reflectivity"]
        reflectivity = raw[:, offset]

    # The `self_filtered` clouds keep their original width and blank removed
    # returns as NaN, which would otherwise poison every downstream mean and
    # bounding box.
    finite = np.isfinite(xyz).all(axis=1)
    if not finite.all():
        xyz = xyz[finite]
        if reflectivity is not None:
            reflectivity = reflectivity[finite]

    return xyz, reflectivity


def reflectivity_colors(reflectivity: np.ndarray | None, count: int) -> np.ndarray:
    """Map reflectivity to RGB, or a flat grey when it is missing.

    Stands in for camera-projected colour, which is unavailable here: the images
    arrive as an h265 stream that only the viewer decodes.

    Args:
        reflectivity: Per-point reflectivity, or ``None``.
        count: Number of points, used when reflectivity is missing.

    Returns:
        ``[N, 3]`` uint8 colours.
    """
    if reflectivity is None:
        return np.full((count, 3), 140, dtype=np.uint8)

    # Most returns sit low, so a square-root ramp spreads them out.
    t = np.sqrt(reflectivity.astype(np.float32) / 255.0)
    colors = np.empty((count, 3), dtype=np.uint8)
    colors[:, 0] = (60 + 195 * t).astype(np.uint8)
    colors[:, 1] = (70 + 150 * t).astype(np.uint8)
    colors[:, 2] = (170 - 120 * t).astype(np.uint8)
    return colors


def colorize_from_cameras(
    points: np.ndarray,
    cameras: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    fallback: tuple[int, int, int] = (140, 140, 140),
) -> tuple[np.ndarray, float]:
    """Colour points by projecting them into decoded camera images.

    Cameras are handled one at a time rather than batched, because this rig mixes
    portrait and landscape sensors and their images cannot be stacked. Where
    fields of view overlap the nearest camera wins, which keeps the result
    deterministic.

    Args:
        points: ``[N, 3]`` points in the frame the extrinsics map from.
        cameras: One ``(rgb [H, W, 3], K [3, 3], cam_from_points [4, 4])`` per
            camera.
        fallback: Colour for points no camera sees.

    Returns:
        ``([N, 3] uint8 colours, fraction of points coloured)``.
    """
    colors = np.tile(np.asarray(fallback, dtype=np.uint8), (len(points), 1))
    if not len(points):
        return colors, 0.0

    nearest = np.full(len(points), np.inf, dtype=np.float32)

    for rgb, k, cam_from_points in cameras:
        local = points @ cam_from_points[:3, :3].T.astype(np.float32) + cam_from_points[
            :3, 3
        ].astype(np.float32)
        depth = local[:, 2]
        in_front = depth > 0.05
        safe = np.where(in_front, depth, 1.0)

        u = np.rint(k[0, 0] * local[:, 0] / safe + k[0, 2]).astype(np.int32)
        v = np.rint(k[1, 1] * local[:, 1] / safe + k[1, 2]).astype(np.int32)

        height, width = rgb.shape[:2]
        visible = (
            in_front
            & (u >= 0)
            & (u < width)
            & (v >= 0)
            & (v < height)
            & (depth < nearest)
        )
        if not visible.any():
            continue
        colors[visible] = rgb[v[visible], u[visible]]
        nearest[visible] = depth[visible]

    return colors, float(np.isfinite(nearest).mean())


class CameraDecoder:
    """Decodes one camera's h265 stream on its own thread.

    Five cameras at ~26 fps of 1920x1536 is around 130 fps of HEVC to decode, and
    doing it inline serialises all of it behind the bag reader. ffmpeg releases
    the GIL while decoding, so a thread per camera turns that into real
    parallelism.

    Hardware decode was measured and rejected: ``hevc_cuvid`` ran at 190 fps
    against software's 214 fps on this machine, because the frames have to come
    back to system memory to colour points with.

    Only the newest frame is kept. Colouring happens at the lidar keyframe rate,
    far below the video rate, so older frames are of no use -- but every sample
    still has to pass through the decoder, an h265 stream being decodable only in
    order.
    """

    def __init__(self, queue_size: int = 64) -> None:
        import queue as _queue
        import threading

        import av

        self._dropped = 0
        self._decode_errors = 0
        self._queue: _queue.Queue[bytes | None] = _queue.Queue(maxsize=queue_size)
        self._context = av.CodecContext.create("hevc", "r")
        self._latest = None
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while True:
            sample = self._queue.get()
            if sample is None:
                return
            try:
                for packet in self._context.parse(sample):
                    for frame in self._context.decode(packet):
                        self._latest = frame
            except Exception:  # noqa: BLE001
                # A corrupt or mid-stream-start sample must not kill the feed;
                # the count is reported once at the end.
                self._decode_errors += 1

    def submit(self, sample: bytes) -> None:
        """Queue a sample, dropping it if the decoder has fallen behind.

        Args:
            sample: One h265 access unit.
        """
        import queue as _queue

        try:
            self._queue.put_nowait(sample)
        except _queue.Full:
            # Better to lose colour fidelity on one keyframe than to stall the
            # bag reader; reported at the end.
            self._dropped += 1

    def latest_rgb(self) -> np.ndarray | None:
        """The most recently decoded frame as RGB, or ``None`` if there is none.

        Returns:
            ``[H, W, 3]`` uint8, or ``None``.
        """
        frame = self._latest
        return None if frame is None else frame.to_ndarray(format="rgb24")

    def close(self) -> tuple[int, int]:
        """Stop the worker thread.

        Returns:
            ``(samples dropped, decode errors)``.
        """
        self._queue.put(None)
        self._thread.join(timeout=2.0)
        return self._dropped, self._decode_errors
