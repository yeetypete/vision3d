"""Tests for project_box3d_to_image op."""

import math

import pytest
import torch

from vision3d.ops import box3d_corners, project_box3d_to_image, project_to_image
from vision3d.tensors import BoundingBox3DFormat

FMT = BoundingBox3DFormat.XYZLWHYPR
IMAGE_SIZE = (480, 640)

# Local corner signs in the order of ``box3d_corners``.
_CORNER_SIGNS = (
    (-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1),
    (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1),
)  # fmt: skip


@pytest.fixture
def identity_ext() -> torch.Tensor:
    return torch.eye(4)


@pytest.fixture
def simple_K() -> torch.Tensor:
    K = torch.eye(3)
    K[0, 0] = 500.0  # fx
    K[1, 1] = 500.0  # fy
    K[0, 2] = 320.0  # cx
    K[1, 2] = 240.0  # cy
    return K


def _box(x: float, y: float, z: float, size: float = 1.0) -> list[float]:
    return [x, y, z, size, size, size, 0.0, 0.0, 0.0]


def _random_scene(
    seed: int, n: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Random 9-DoF boxes around a camera, and a random source-to-camera pose.

    The boxes are drawn in the camera frame, so that many are in front of,
    behind, straddling, or beside the camera, then moved to the source frame.

    Returns:
        ``(boxes, extrinsics)`` with boxes ``[n, 9]`` in ``XYZLWHYPR``.
    """
    with torch.device("cpu"):
        boxes, ext = _random_scene_cpu(seed, n)
    return boxes.to(device), ext.to(device)


def _random_scene_cpu(seed: int, n: int) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)

    def uniform(low: float, high: float, *shape: int) -> torch.Tensor:
        return low + (high - low) * torch.rand(*shape, generator=g)

    rot_vec = uniform(-math.pi, math.pi, 3)
    skew = torch.zeros(3, 3)
    skew[0, 1], skew[0, 2], skew[1, 2] = -rot_vec[2], rot_vec[1], -rot_vec[0]
    rot = torch.linalg.matrix_exp(skew - skew.T)
    ext = torch.eye(4)
    ext[:3, :3] = rot
    ext[:3, 3] = uniform(-2.0, 2.0, 3)

    centers_cam = torch.stack(
        [uniform(-8, 8, n), uniform(-6, 6, n), uniform(-3, 12, n)], dim=-1
    )
    centers = (centers_cam - ext[:3, 3]) @ rot  # camera to source frame
    boxes = torch.cat(
        [centers, uniform(0.3, 5.0, n, 3), uniform(-math.pi, math.pi, n, 3)], dim=-1
    )
    return boxes, ext


def _convex_hull(points: list[list[float]]) -> list[list[float]]:
    """Convex hull by Andrew's monotone chain.

    Returns:
        Hull vertices, counterclockwise in (u, v), without collinear points.
    """
    points = sorted(points)

    def turn(o: list[float], a: list[float], b: list[float]) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list[list[float]] = []
    upper: list[list[float]] = []
    for chain, ordered in ((lower, points), (upper, reversed(points))):
        for p in ordered:
            while len(chain) >= 2 and turn(chain[-2], chain[-1], p) <= 0:
                chain.pop()
            chain.append(p)
    return lower[:-1] + upper[:-1]


def _reference(
    box: torch.Tensor,
    ext: torch.Tensor,
    intrinsics: torch.Tensor,
    image_size: tuple[int, int],
    near: float,
) -> list[float] | None:
    """Brute-force extent of the visible part of one box.

    The projected box is the convex hull of the corners in front of the plane
    ``z = near`` and the points where the box edges cross it. The extent of
    its visible part is that of the vertices of its intersection with the
    image: hull vertices inside the image, hull edge crossings of the image
    border, and image corners inside the hull.

    Returns:
        The extent, or ``None`` when no part of the hull is in the image.
    """
    corners = box3d_corners(box[None], FMT)[0]
    signs = torch.tensor(_CORNER_SIGNS, dtype=corners.dtype, device=corners.device)
    # Box edges join the corners whose local signs differ in one coordinate.
    edges = [
        (i, j) for i in range(8) for j in range(i) if (signs[i] != signs[j]).sum() == 1
    ]
    assert len(edges) == 12

    _uv, depth = project_to_image(corners, ext, intrinsics)
    points = [corners[i] for i in range(8) if depth[i] >= near]
    for i, j in edges:
        if (depth[i] - near) * (depth[j] - near) < 0:
            s = (near - depth[i]) / (depth[j] - depth[i])
            points.append(corners[i] + s * (corners[j] - corners[i]))
    if not points:
        return None
    uv, _depth = project_to_image(torch.stack(points), ext, intrinsics)
    hull = _convex_hull(uv.tolist())
    if len(hull) < 3:
        return None

    height, width = image_size

    def in_hull(p: list[float]) -> bool:
        return all(
            (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]) >= 0
            for a, b in zip(hull, hull[1:] + hull[:1], strict=True)
        )

    vertices = [p for p in hull if 0 <= p[0] <= width and 0 <= p[1] <= height]
    for a, b in zip(hull, hull[1:] + hull[:1], strict=True):
        for axis, bound, limit in (
            (0, 0, height),
            (0, width, height),
            (1, 0, width),
            (1, height, width),
        ):
            if (a[axis] - bound) * (b[axis] - bound) < 0:
                s = (bound - a[axis]) / (b[axis] - a[axis])
                other = a[1 - axis] + s * (b[1 - axis] - a[1 - axis])
                if 0 <= other <= limit:
                    vertices.append([bound, other] if axis == 0 else [other, bound])
    vertices += [[u, v] for u in (0, width) for v in (0, height) if in_hull([u, v])]
    if not vertices:
        return None
    us, vs = zip(*vertices, strict=True)
    return [min(us), min(vs), max(us), max(vs)]


class TestProjectBox3dToImage:
    def test_box_in_front_matches_projected_corners(
        self, identity_ext: torch.Tensor, simple_K: torch.Tensor
    ) -> None:
        boxes = torch.tensor([[0.2, -0.1, 10.0, 1.0, 2.0, 1.5, 0.3, 0.2, 0.1]])
        boxes_2d, valid = project_box3d_to_image(
            boxes, FMT, identity_ext, simple_K, IMAGE_SIZE
        )
        uv, _ = project_to_image(box3d_corners(boxes, FMT)[0], identity_ext, simple_K)
        expected = torch.cat([uv.amin(dim=0), uv.amax(dim=0)])
        torch.testing.assert_close(boxes_2d[0], expected)
        assert valid.tolist() == [True]

    def test_straddling_box_is_finite(
        self, identity_ext: torch.Tensor, simple_K: torch.Tensor
    ) -> None:
        # Center on the camera plane: half the box is behind the camera, so
        # projecting its corners directly gives NaN.
        boxes = torch.tensor([_box(0.0, 0.0, 0.0, size=4.0)])
        uv, _ = project_to_image(box3d_corners(boxes, FMT)[0], identity_ext, simple_K)
        assert uv.isnan().any()

        boxes_2d, valid = project_box3d_to_image(
            boxes, FMT, identity_ext, simple_K, IMAGE_SIZE
        )
        assert boxes_2d.isfinite().all()
        assert valid.tolist() == [True]
        # The box surrounds the camera, so it fills the whole image.
        torch.testing.assert_close(boxes_2d[0], torch.tensor([0.0, 0.0, 640.0, 480.0]))

    def test_box_behind_is_invalid(
        self, identity_ext: torch.Tensor, simple_K: torch.Tensor
    ) -> None:
        boxes = torch.tensor([_box(0.0, 0.0, -10.0)])
        boxes_2d, valid = project_box3d_to_image(
            boxes, FMT, identity_ext, simple_K, IMAGE_SIZE
        )
        assert valid.tolist() == [False]
        assert (boxes_2d == 0).all()

    def test_box_outside_image_is_invalid(
        self, identity_ext: torch.Tensor, simple_K: torch.Tensor
    ) -> None:
        boxes = torch.tensor([_box(100.0, 0.0, 10.0)])
        boxes_2d, valid = project_box3d_to_image(
            boxes, FMT, identity_ext, simple_K, IMAGE_SIZE
        )
        assert valid.tolist() == [False]
        assert (boxes_2d == 0).all()

    def test_flat_box_edge_on_is_invalid(
        self, identity_ext: torch.Tensor, simple_K: torch.Tensor
    ) -> None:
        # No extent in y and centered on the optical axis: the box lies in a
        # plane through the camera and projects to a line.
        boxes = torch.tensor([[0.0, 0.0, 10.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0]])
        _boxes_2d, valid = project_box3d_to_image(
            boxes, FMT, identity_ext, simple_K, IMAGE_SIZE
        )
        assert valid.tolist() == [False]

    def test_box_across_image_border(
        self, identity_ext: torch.Tensor, simple_K: torch.Tensor
    ) -> None:
        # A flat box (no extent in z) at depth 10 spanning x in [5.9, 6.9]
        # projects to u in [615, 665]: half of it is right of the image.
        boxes = torch.tensor([[6.4, 0.0, 10.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]])
        boxes_2d, valid = project_box3d_to_image(
            boxes, FMT, identity_ext, simple_K, IMAGE_SIZE
        )
        assert valid.tolist() == [True]
        torch.testing.assert_close(
            boxes_2d[0], torch.tensor([615.0, 215.0, 640.0, 265.0])
        )

    def test_rotated_box_across_image_corner_is_tight(
        self, identity_ext: torch.Tensor, simple_K: torch.Tensor
    ) -> None:
        # A flat square turned 45 degrees about the optical axis, at depth 10
        # (50 px per m), projects to a diamond with half-diagonal 50 px
        # centered at (650, -10), just beyond the top-right image corner. Its
        # visible part is the triangle (610, 0), (640, 0), (640, 30). Clamping
        # the extent of the corners to the image would give the looser box
        # (600, 0, 640, 40).
        side = math.sqrt(2.0)
        boxes = torch.tensor([[6.6, -5.0, 10.0, side, side, 0.0, math.pi / 4, 0, 0]])
        boxes_2d, valid = project_box3d_to_image(
            boxes, FMT, identity_ext, simple_K, IMAGE_SIZE
        )
        assert valid.tolist() == [True]
        torch.testing.assert_close(
            boxes_2d[0], torch.tensor([610.0, 0.0, 640.0, 30.0]), atol=1e-3, rtol=0
        )

    def test_box_cut_by_image_border(
        self, identity_ext: torch.Tensor, simple_K: torch.Tensor
    ) -> None:
        # A box spanning x in [0, 20] and z in [9, 11] reaches far beyond the
        # right image border, the plane x = 0.64 z. Its extent is that of the
        # near face at z = 9, cut at the border.
        boxes = torch.tensor([[10.0, 0.0, 10.0, 20.0, 1.0, 2.0, 0.0, 0.0, 0.0]])
        boxes_2d, valid = project_box3d_to_image(
            boxes, FMT, identity_ext, simple_K, IMAGE_SIZE
        )
        assert valid.tolist() == [True]
        torch.testing.assert_close(
            boxes_2d[0], torch.tensor([320.0, 240 - 500 / 18, 640.0, 240 + 500 / 18])
        )

    @pytest.mark.parametrize("seed", range(2))
    def test_matches_reference(self, seed: int, device: torch.device) -> None:
        near = 0.5
        image_size = (120, 160)
        intrinsics = torch.tensor(
            [[100.0, 0.0, 80.0], [0.0, 100.0, 60.0], [0.0, 0.0, 1.0]]
        )
        boxes, ext = _random_scene(seed, 32, device)
        boxes_2d, valid = project_box3d_to_image(
            boxes, FMT, ext, intrinsics, image_size, near=near
        )

        checked = 0
        for i in range(len(boxes)):
            ref = _reference(
                boxes[i].double(), ext.double(), intrinsics.double(), image_size, near
            )
            if not valid[i]:
                # Nothing visible, or only a sliver without area.
                if ref is not None:
                    u1, v1, u2, v2 = ref
                    assert (u2 - u1) * (v2 - v1) < 1e-6, f"box {i}"
                continue
            assert ref is not None, f"box {i}"
            torch.testing.assert_close(
                boxes_2d[i],
                torch.tensor(ref, dtype=boxes_2d.dtype),
                atol=1e-3,
                rtol=1e-5,
                msg=f"box {i}",
            )
            checked += 1
        assert checked >= 5, "the scene must have visible boxes"

    def test_rows_are_independent(
        self, simple_K: torch.Tensor, device: torch.device
    ) -> None:
        boxes, ext = _random_scene(0, 32, device)
        boxes = torch.cat([boxes, torch.zeros(1, 9)])  # degenerate point box
        batched = project_box3d_to_image(boxes, FMT, ext, simple_K, IMAGE_SIZE)
        rows = [
            project_box3d_to_image(box[None], FMT, ext, simple_K, IMAGE_SIZE)
            for box in boxes
        ]
        for got, expected in zip(batched, map(torch.cat, zip(*rows)), strict=True):
            torch.testing.assert_close(got, expected, atol=1e-3, rtol=1e-5)
        assert batched[1].any()
        assert not batched[1].all()

    def test_extrinsics_are_applied(self, simple_K: torch.Tensor) -> None:
        # The same boxes given in the camera frame, and in a source frame
        # rotated 90 degrees about x and shifted: (x, y, z) in the source
        # frame is (x, -z, y) + t in the camera frame. Axis-aligned source
        # boxes stay axis-aligned, with their y and z sizes swapped.
        boxes_cam = torch.tensor(
            [
                [0.5, -0.2, 10.0, 1.0, 2.0, 3.0],
                [1.0, 0.5, 0.5, 2.0, 1.5, 4.0],  # straddles the camera plane
                [7.0, 1.0, 10.0, 2.0, 1.0, 1.0],  # crosses the image border
            ]
        )
        ext = torch.zeros(4, 4)
        ext[0, 0] = ext[2, 1] = ext[3, 3] = 1.0
        ext[1, 2] = -1.0
        ext[:3, 3] = torch.tensor([0.3, -0.4, 2.0])
        centers = (boxes_cam[:, :3] - ext[:3, 3]) @ ext[:3, :3]
        sizes = boxes_cam[:, [3, 5, 4]]
        boxes_src = torch.cat([centers, sizes], dim=-1)

        fmt = BoundingBox3DFormat.XYZLWH
        expected = project_box3d_to_image(
            boxes_cam, fmt, torch.eye(4), simple_K, IMAGE_SIZE
        )
        got = project_box3d_to_image(boxes_src, fmt, ext, simple_K, IMAGE_SIZE)
        for g, e in zip(got, expected, strict=True):
            torch.testing.assert_close(g, e)
        assert expected[1].all()

    def test_all_formats_agree(
        self, identity_ext: torch.Tensor, simple_K: torch.Tensor
    ) -> None:
        boxes = {
            BoundingBox3DFormat.XYZXYZ: [[-0.5, -1.0, 9.0, 0.5, 1.0, 11.0]],
            BoundingBox3DFormat.XYZLWH: [[0.0, 0.0, 10.0, 1.0, 2.0, 2.0]],
            BoundingBox3DFormat.XYZLWHY: [[0.0, 0.0, 10.0, 1.0, 2.0, 2.0, 0.0]],
            BoundingBox3DFormat.XYZLWHYPR: [[0.0, 0.0, 10.0, 1.0, 2.0, 2.0, 0, 0, 0]],
        }
        results = [
            project_box3d_to_image(
                torch.tensor(b, dtype=torch.float32),
                fmt,
                identity_ext,
                simple_K,
                IMAGE_SIZE,
            )[0]
            for fmt, b in boxes.items()
        ]
        for result in results[1:]:
            torch.testing.assert_close(result, results[0])

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_dtype(self, dtype: torch.dtype, simple_K: torch.Tensor) -> None:
        boxes = torch.tensor([_box(0.0, 0.0, 10.0)], dtype=dtype)
        ext = torch.eye(4, dtype=dtype)
        outputs = project_box3d_to_image(
            boxes, FMT, ext, simple_K.to(dtype), IMAGE_SIZE
        )
        assert [o.dtype for o in outputs] == [dtype, torch.bool]

    def test_mixed_dtype(
        self, identity_ext: torch.Tensor, simple_K: torch.Tensor
    ) -> None:
        # float64 boxes (common from numpy) with float32 camera matrices.
        boxes = torch.tensor([_box(1.0, 0.0, 0.5, size=2.0)], dtype=torch.float64)
        got = project_box3d_to_image(boxes, FMT, identity_ext, simple_K, IMAGE_SIZE)
        expected = project_box3d_to_image(
            boxes, FMT, identity_ext.double(), simple_K.double(), IMAGE_SIZE
        )
        for g, e in zip(got, expected, strict=True):
            torch.testing.assert_close(g, e)

    def test_gradients(self, simple_K: torch.Tensor) -> None:
        # One box inside the image, one crossing its border and the camera
        # plane, away from points where the output is not differentiable.
        boxes = torch.tensor(
            [
                [0.3, -0.2, 10.0, 2.0, 1.5, 1.0, 0.4, 0.2, -0.3],
                [3.0, 0.5, 0.4, 2.0, 1.5, 3.0, 0.3, -0.2, 0.1],
            ],
            dtype=torch.float64,
            requires_grad=True,
        )
        ext = torch.eye(4, dtype=torch.float64)
        K = simple_K.double()

        def fn(b: torch.Tensor) -> torch.Tensor:
            return project_box3d_to_image(b, FMT, ext, K, IMAGE_SIZE, near=0.1)[0]

        assert torch.autograd.gradcheck(fn, (boxes,))

    def test_torch_compile(
        self, identity_ext: torch.Tensor, simple_K: torch.Tensor
    ) -> None:
        boxes, ext = _random_scene(1, 16, identity_ext.device)
        compiled = torch.compile(project_box3d_to_image, fullgraph=True)
        expected = project_box3d_to_image(boxes, FMT, ext, simple_K, IMAGE_SIZE)
        got = compiled(boxes, FMT, ext, simple_K, IMAGE_SIZE)
        for g, e in zip(got, expected, strict=True):
            torch.testing.assert_close(g, e)

    def test_empty_boxes(
        self, identity_ext: torch.Tensor, simple_K: torch.Tensor
    ) -> None:
        boxes = torch.zeros(0, 9)
        boxes_2d, valid = project_box3d_to_image(
            boxes, FMT, identity_ext, simple_K, IMAGE_SIZE
        )
        assert boxes_2d.shape == (0, 4)
        assert valid.shape == (0,)

    def test_rejects_nonpositive_near(
        self, identity_ext: torch.Tensor, simple_K: torch.Tensor
    ) -> None:
        with pytest.raises(ValueError, match="near must be positive"):
            project_box3d_to_image(
                torch.zeros(1, 9), FMT, identity_ext, simple_K, IMAGE_SIZE, near=0.0
            )
