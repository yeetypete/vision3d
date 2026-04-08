"""Tests for project_to_image op."""

import pytest
import torch

from vision3d.ops import project_to_image


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


class TestProjection:
    def test_point_on_optical_axis(
        self, identity_ext: torch.Tensor, simple_K: torch.Tensor
    ) -> None:
        pts = torch.tensor([[0.0, 0.0, 10.0]])
        uv, depth = project_to_image(pts, identity_ext, simple_K)
        assert depth[0].isclose(torch.tensor(10.0))
        assert uv[0, 0].isclose(torch.tensor(320.0))
        assert uv[0, 1].isclose(torch.tensor(240.0))

    def test_known_projection(
        self, identity_ext: torch.Tensor, simple_K: torch.Tensor
    ) -> None:
        pts = torch.tensor([[1.0, 0.0, 5.0]])
        uv, depth = project_to_image(pts, identity_ext, simple_K)
        assert depth[0].isclose(torch.tensor(5.0))
        assert uv[0, 0].isclose(torch.tensor(420.0))  # fx * 1/5 + cx
        assert uv[0, 1].isclose(torch.tensor(240.0))

    def test_batch_shape(
        self, identity_ext: torch.Tensor, simple_K: torch.Tensor
    ) -> None:
        pts = torch.rand(10, 3)
        uv, depth = project_to_image(pts, identity_ext, simple_K)
        assert uv.shape == (10, 2)
        assert depth.shape == (10,)

    def test_translation_extrinsics(self, simple_K: torch.Tensor) -> None:
        pts = torch.tensor([[0.0, 0.0, 10.0]])
        ext = torch.eye(4)
        ext[2, 3] = 5.0
        _uv, depth = project_to_image(pts, ext, simple_K)
        assert depth[0].isclose(torch.tensor(15.0))

    def test_behind_camera(
        self, identity_ext: torch.Tensor, simple_K: torch.Tensor
    ) -> None:
        pts = torch.tensor([[0.0, 0.0, -5.0]])
        uv, depth = project_to_image(pts, identity_ext, simple_K)
        assert depth[0] < 0
        assert uv[0].isnan().all()

    def test_empty_points(
        self, identity_ext: torch.Tensor, simple_K: torch.Tensor
    ) -> None:
        pts = torch.zeros(0, 3)
        uv, depth = project_to_image(pts, identity_ext, simple_K)
        assert uv.shape == (0, 2)
        assert depth.shape == (0,)
