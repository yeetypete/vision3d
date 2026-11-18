"""Tests for box3d_truncation op."""

import math

import pytest
import torch

from vision3d.ops import box3d_corners, box3d_truncation, project_to_image
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


def _reference(
    box: torch.Tensor,
    ext: torch.Tensor,
    intrinsics: torch.Tensor,
    image_size: tuple[int, int],
    samples: int = 200_000,
) -> float:
    """Fraction of uniform random points in one box outside the image.

    Returns:
        The sampled truncation.
    """
    corners = box3d_corners(box[None], FMT)[0]
    signs = torch.tensor(_CORNER_SIGNS, dtype=corners.dtype, device=corners.device)
    g = torch.Generator().manual_seed(0)
    local = torch.rand(samples, 3, generator=g, dtype=corners.dtype, device="cpu")
    local = (local * 2 - 1).to(corners.device)
    # Trilinear interpolation of the corners maps the cube uniformly to the box.
    weights = ((1 + local[:, None] * signs) / 2).prod(dim=-1)  # [S, 8]
    uv, depth = project_to_image(weights @ corners, ext, intrinsics)
    height, width = image_size
    seen = (depth > 0) & (uv >= 0).all(-1) & (uv[:, 0] <= width) & (uv[:, 1] <= height)
    return 1 - seen.double().mean().item()


class TestBox3dTruncation:
    def test_box_in_view_is_zero(
        self, identity_ext: torch.Tensor, simple_K: torch.Tensor
    ) -> None:
        boxes = torch.tensor([[0.2, -0.1, 10.0, 1.0, 2.0, 1.5, 0.3, 0.2, 0.1]])
        truncation = box3d_truncation(boxes, FMT, identity_ext, simple_K, IMAGE_SIZE)
        assert truncation.tolist() == [0.0]

    def test_box_behind_is_one(
        self, identity_ext: torch.Tensor, simple_K: torch.Tensor
    ) -> None:
        boxes = torch.tensor([_box(0.0, 0.0, -10.0)])
        truncation = box3d_truncation(boxes, FMT, identity_ext, simple_K, IMAGE_SIZE)
        assert truncation.tolist() == [1.0]

    def test_box_outside_image_is_one(
        self, identity_ext: torch.Tensor, simple_K: torch.Tensor
    ) -> None:
        boxes = torch.tensor([_box(100.0, 0.0, 10.0)])
        truncation = box3d_truncation(boxes, FMT, identity_ext, simple_K, IMAGE_SIZE)
        assert truncation.tolist() == [1.0]

    def test_box_around_camera(
        self, identity_ext: torch.Tensor, simple_K: torch.Tensor
    ) -> None:
        # Centered on the camera: the part in view is the field-of-view
        # pyramid cut off by the far face, z <= 2 within |x| <= 0.64 z and
        # |y| <= 0.48 z, of volume 4 * 0.64 * 0.48 * 8 / 3, out of 64.
        boxes = torch.tensor([_box(0.0, 0.0, 0.0, size=4.0)])
        truncation = box3d_truncation(boxes, FMT, identity_ext, simple_K, IMAGE_SIZE)
        expected = 1 - 4 * 0.64 * 0.48 * 8 / 3 / 64
        torch.testing.assert_close(truncation, torch.tensor([expected]))

    def test_volume_outside_view(
        self, identity_ext: torch.Tensor, simple_K: torch.Tensor
    ) -> None:
        # A box spanning x in [0, 20] and z in [9, 11], well inside the view
        # in y. The right image border is the plane x = 0.64 z, so the part in
        # view has volume 0.32 (11**2 - 9**2) = 12.8 times the y size, out of
        # 40 times the y size.
        boxes = torch.tensor([[10.0, 0.0, 10.0, 20.0, 1.0, 2.0, 0.0, 0.0, 0.0]])
        truncation = box3d_truncation(boxes, FMT, identity_ext, simple_K, IMAGE_SIZE)
        torch.testing.assert_close(truncation, torch.tensor([1 - 12.8 / 40]))

    def test_flat_box_uses_area(
        self, identity_ext: torch.Tensor, simple_K: torch.Tensor
    ) -> None:
        # A flat box (no extent in z) at depth 10 spanning x in [5.9, 6.9]
        # projects to u in [615, 665]: half of it is right of the image.
        boxes = torch.tensor([[6.4, 0.0, 10.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]])
        truncation = box3d_truncation(boxes, FMT, identity_ext, simple_K, IMAGE_SIZE)
        torch.testing.assert_close(truncation, torch.tensor([0.5]))

    def test_rotated_flat_box_across_image_corner(
        self, identity_ext: torch.Tensor, simple_K: torch.Tensor
    ) -> None:
        # A flat square turned 45 degrees about the optical axis, at depth 10
        # (50 px per m), projects to a diamond with half-diagonal 50 px
        # centered at (650, -10), just beyond the top-right image corner. Its
        # visible part is a triangle of area 450, out of 5000 for the diamond.
        side = math.sqrt(2.0)
        boxes = torch.tensor([[6.6, -5.0, 10.0, side, side, 0.0, math.pi / 4, 0, 0]])
        truncation = box3d_truncation(boxes, FMT, identity_ext, simple_K, IMAGE_SIZE)
        torch.testing.assert_close(truncation, torch.tensor([0.91]))

    def test_flat_box_edge_on_is_inside(
        self, identity_ext: torch.Tensor, simple_K: torch.Tensor
    ) -> None:
        # No extent in y and centered on the optical axis: the box lies in a
        # plane through the camera. It projects to a line, but it is inside
        # the field of view.
        boxes = torch.tensor([[0.0, 0.0, 10.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0]])
        truncation = box3d_truncation(boxes, FMT, identity_ext, simple_K, IMAGE_SIZE)
        assert truncation.tolist() == [0.0]

    def test_point_box_is_one(
        self, identity_ext: torch.Tensor, simple_K: torch.Tensor
    ) -> None:
        boxes = torch.tensor([[0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        truncation = box3d_truncation(boxes, FMT, identity_ext, simple_K, IMAGE_SIZE)
        assert truncation.tolist() == [1.0]

    @pytest.mark.parametrize("seed", range(2))
    def test_matches_reference(self, seed: int, device: torch.device) -> None:
        image_size = (120, 160)
        intrinsics = torch.tensor(
            [[100.0, 0.0, 80.0], [0.0, 100.0, 60.0], [0.0, 0.0, 1.0]]
        )
        boxes, ext = _random_scene(seed, 48, device)
        truncation = box3d_truncation(boxes, FMT, ext, intrinsics, image_size)

        partial = 0
        for i in range(len(boxes)):
            ref = _reference(
                boxes[i].double(), ext.double(), intrinsics.double(), image_size
            )
            # Six standard deviations of the sampled fraction.
            assert abs(truncation[i].item() - ref) <= 0.007, f"box {i}"
            partial += 0.01 < ref < 0.99
        assert partial >= 5, "the scene must have partly visible boxes"

    def test_rows_are_independent(
        self, simple_K: torch.Tensor, device: torch.device
    ) -> None:
        boxes, ext = _random_scene(0, 32, device)
        boxes = torch.cat([boxes, torch.zeros(1, 9)])  # degenerate point box
        batched = box3d_truncation(boxes, FMT, ext, simple_K, IMAGE_SIZE)
        rows = torch.cat(
            [
                box3d_truncation(box[None], FMT, ext, simple_K, IMAGE_SIZE)
                for box in boxes
            ]
        )
        torch.testing.assert_close(batched, rows, atol=1e-5, rtol=1e-5)

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
        expected = box3d_truncation(boxes_cam, fmt, torch.eye(4), simple_K, IMAGE_SIZE)
        got = box3d_truncation(boxes_src, fmt, ext, simple_K, IMAGE_SIZE)
        torch.testing.assert_close(got, expected)
        assert expected[0] == 0
        assert (expected[1:] > 0).all()

    def test_all_formats_agree(
        self, identity_ext: torch.Tensor, simple_K: torch.Tensor
    ) -> None:
        # Crosses the right image border, so the truncation is not trivial.
        boxes = {
            BoundingBox3DFormat.XYZXYZ: [[5.5, -1.0, 9.0, 7.5, 1.0, 11.0]],
            BoundingBox3DFormat.XYZLWH: [[6.5, 0.0, 10.0, 2.0, 2.0, 2.0]],
            BoundingBox3DFormat.XYZLWHY: [[6.5, 0.0, 10.0, 2.0, 2.0, 2.0, 0.0]],
            BoundingBox3DFormat.XYZLWHYPR: [[6.5, 0.0, 10.0, 2.0, 2.0, 2.0, 0, 0, 0]],
        }
        results = [
            box3d_truncation(torch.tensor(b), fmt, identity_ext, simple_K, IMAGE_SIZE)
            for fmt, b in boxes.items()
        ]
        assert 0 < results[0] < 1
        for result in results[1:]:
            torch.testing.assert_close(result, results[0])

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_dtype(self, dtype: torch.dtype, simple_K: torch.Tensor) -> None:
        boxes = torch.tensor([_box(0.0, 0.0, 10.0)], dtype=dtype)
        ext = torch.eye(4, dtype=dtype)
        truncation = box3d_truncation(boxes, FMT, ext, simple_K.to(dtype), IMAGE_SIZE)
        assert truncation.dtype == dtype

    def test_small_box_far_away(
        self, identity_ext: torch.Tensor, simple_K: torch.Tensor
    ) -> None:
        # A 0.2 m box at 1 km, centered on the right image border plane
        # x = 0.64 z: half of it is outside, also from float32 input.
        box = [640.0, 0.0, 1000.0, 0.2, 0.2, 0.2, 0.3, 0.2, 0.1]
        expected = box3d_truncation(
            torch.tensor([box], dtype=torch.float64),
            FMT,
            identity_ext.double(),
            simple_K.double(),
            IMAGE_SIZE,
        )
        torch.testing.assert_close(
            expected, torch.tensor([0.5], dtype=torch.float64), atol=1e-6, rtol=0
        )
        got = box3d_truncation(
            torch.tensor([box]), FMT, identity_ext, simple_K, IMAGE_SIZE
        )
        torch.testing.assert_close(got, torch.tensor([0.5]), atol=1e-3, rtol=0)

    @pytest.mark.parametrize("thickness", [0.0, 1e-12, 1e-7, 1e-5, 1e-3])
    def test_nearly_flat_box_is_continuous(
        self, thickness: float, identity_ext: torch.Tensor, simple_K: torch.Tensor
    ) -> None:
        # As test_flat_box_uses_area, with a small thickness in z.
        boxes = torch.tensor([[6.4, 0.0, 10.0, 1.0, 1.0, thickness, 0.0, 0.0, 0.0]])
        truncation = box3d_truncation(boxes, FMT, identity_ext, simple_K, IMAGE_SIZE)
        torch.testing.assert_close(truncation, torch.tensor([0.5]), atol=1e-4, rtol=0)

    def test_non_rigid_extrinsics(
        self, identity_ext: torch.Tensor, simple_K: torch.Tensor
    ) -> None:
        # Scaling z by 2 maps the source box to the camera-frame box.
        fmt = BoundingBox3DFormat.XYZLWH
        ext = torch.diag(torch.tensor([1.0, 1.0, 2.0, 1.0]))
        got = box3d_truncation(
            torch.tensor([[6.5, 0.0, 5.0, 2.0, 2.0, 1.0]]),
            fmt,
            ext,
            simple_K,
            IMAGE_SIZE,
        )
        expected = box3d_truncation(
            torch.tensor([[6.5, 0.0, 10.0, 2.0, 2.0, 2.0]]),
            fmt,
            identity_ext,
            simple_K,
            IMAGE_SIZE,
        )
        assert 0 < expected < 1
        torch.testing.assert_close(got, expected)

    def test_negative_sizes(
        self, identity_ext: torch.Tensor, simple_K: torch.Tensor
    ) -> None:
        # Swapped corners give the same box with inward oriented faces.
        fmt = BoundingBox3DFormat.XYZXYZ
        boxes = torch.tensor(
            [[5.5, -1.0, 9.0, 7.5, 1.0, 11.0], [7.5, 1.0, 11.0, 5.5, -1.0, 9.0]]
        )
        truncation = box3d_truncation(boxes, fmt, identity_ext, simple_K, IMAGE_SIZE)
        assert 0 < truncation[0] < 1
        torch.testing.assert_close(truncation[1], truncation[0])

    def test_camera_on_other_device(
        self, identity_ext: torch.Tensor, simple_K: torch.Tensor, device: torch.device
    ) -> None:
        boxes = torch.tensor([_box(6.0, 0.0, 10.0, size=2.0)])
        got = box3d_truncation(
            boxes, FMT, identity_ext.cpu(), simple_K.cpu(), IMAGE_SIZE
        )
        expected = box3d_truncation(boxes, FMT, identity_ext, simple_K, IMAGE_SIZE)
        assert got.device == boxes.device
        torch.testing.assert_close(got, expected)

    def test_rejects_batched_cameras(self, simple_K: torch.Tensor) -> None:
        with pytest.raises(ValueError, match="extrinsics"):
            box3d_truncation(
                torch.zeros(1, 9), FMT, torch.eye(4)[None], simple_K, IMAGE_SIZE
            )

    def test_mixed_dtype(
        self, identity_ext: torch.Tensor, simple_K: torch.Tensor
    ) -> None:
        # float64 boxes (common from numpy) with float32 camera matrices.
        boxes = torch.tensor([_box(1.0, 0.0, 0.5, size=2.0)], dtype=torch.float64)
        got = box3d_truncation(boxes, FMT, identity_ext, simple_K, IMAGE_SIZE)
        expected = box3d_truncation(
            boxes, FMT, identity_ext.double(), simple_K.double(), IMAGE_SIZE
        )
        torch.testing.assert_close(got, expected)

    def test_gradients(self, simple_K: torch.Tensor) -> None:
        # Boxes crossing the image border and the camera plane, away from
        # points where the output is not differentiable.
        boxes = torch.tensor(
            [
                [6.0, -0.2, 10.0, 2.0, 1.5, 1.0, 0.4, 0.2, -0.3],
                [3.0, 0.5, 0.4, 2.0, 1.5, 3.0, 0.3, -0.2, 0.1],
            ],
            dtype=torch.float64,
            requires_grad=True,
        )
        ext = torch.eye(4, dtype=torch.float64)
        K = simple_K.double()

        def fn(b: torch.Tensor) -> torch.Tensor:
            return box3d_truncation(b, FMT, ext, K, IMAGE_SIZE)

        assert torch.autograd.gradcheck(fn, (boxes,))

    def test_torch_compile(
        self, identity_ext: torch.Tensor, simple_K: torch.Tensor
    ) -> None:
        boxes, ext = _random_scene(1, 16, identity_ext.device)
        compiled = torch.compile(box3d_truncation, fullgraph=True)
        expected = box3d_truncation(boxes, FMT, ext, simple_K, IMAGE_SIZE)
        got = compiled(boxes, FMT, ext, simple_K, IMAGE_SIZE)
        torch.testing.assert_close(got, expected)

    def test_empty_boxes(
        self, identity_ext: torch.Tensor, simple_K: torch.Tensor
    ) -> None:
        boxes = torch.zeros(0, 9)
        truncation = box3d_truncation(boxes, FMT, identity_ext, simple_K, IMAGE_SIZE)
        assert truncation.shape == (0,)
