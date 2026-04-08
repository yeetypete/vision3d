"""Tests for project_to_image op."""

import torch

from vision3d.ops import project_to_image


def _identity_extrinsics() -> torch.Tensor:
    return torch.eye(4)


def _simple_intrinsics(
    fx: float = 500, fy: float = 500, cx: float = 320, cy: float = 240
) -> torch.Tensor:
    K = torch.eye(3)
    K[0, 0] = fx
    K[1, 1] = fy
    K[0, 2] = cx
    K[1, 2] = cy
    return K


class TestProjectToImageBasic:
    def test_point_on_optical_axis(self) -> None:
        """Point at (0, 0, 10) in camera frame -> projects to principal point."""
        pts = torch.tensor([[0.0, 0.0, 10.0]])
        ext = _identity_extrinsics()
        K = _simple_intrinsics()

        uv, depth = project_to_image(pts, ext, K)

        assert depth[0].isclose(torch.tensor(10.0))
        assert uv[0, 0].isclose(torch.tensor(320.0))  # cx
        assert uv[0, 1].isclose(torch.tensor(240.0))  # cy

    def test_known_projection(self) -> None:
        """Point at (1, 0, 5) -> u = fx * 1/5 + cx = 500*0.2 + 320 = 420."""
        pts = torch.tensor([[1.0, 0.0, 5.0]])
        ext = _identity_extrinsics()
        K = _simple_intrinsics()

        uv, depth = project_to_image(pts, ext, K)

        assert depth[0].isclose(torch.tensor(5.0))
        assert uv[0, 0].isclose(torch.tensor(420.0))
        assert uv[0, 1].isclose(torch.tensor(240.0))

    def test_batch_shape(self) -> None:
        pts = torch.rand(10, 3)
        ext = _identity_extrinsics()
        K = _simple_intrinsics()

        uv, depth = project_to_image(pts, ext, K)

        assert uv.shape == (10, 2)
        assert depth.shape == (10,)


class TestProjectToImageExtrinsics:
    def test_translation_extrinsics(self) -> None:
        """Extrinsics with translation: camera shifted 5 units along Z."""
        pts = torch.tensor([[0.0, 0.0, 10.0]])
        ext = torch.eye(4)
        ext[2, 3] = 5.0  # camera is 5 units "ahead" in Z
        K = _simple_intrinsics()

        _uv, depth = project_to_image(pts, ext, K)

        # Effective depth = 10 + 5 = 15
        assert depth[0].isclose(torch.tensor(15.0))

    def test_behind_camera(self) -> None:
        """Point behind camera has negative depth."""
        pts = torch.tensor([[0.0, 0.0, -5.0]])
        ext = _identity_extrinsics()
        K = _simple_intrinsics()

        _uv, depth = project_to_image(pts, ext, K)

        assert depth[0] < 0


class TestProjectToImageEmpty:
    def test_empty_points(self) -> None:
        pts = torch.zeros(0, 3)
        ext = _identity_extrinsics()
        K = _simple_intrinsics()

        uv, depth = project_to_image(pts, ext, K)

        assert uv.shape == (0, 2)
        assert depth.shape == (0,)
