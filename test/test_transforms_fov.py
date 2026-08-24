"""Tests for camera field-of-view filtering."""

import torch

from vision3d.transforms.functional import get_fov_mask


class TestGetFovMask:
    def test_selects_points_within_image(self) -> None:
        projection_matrix = torch.tensor(
            [
                [10.0, .0, 5.0, .0],
                [.0, 10.0, 4.0, .0],
                [.0, .0, 1.0, .0],
            ]
        )
        points = torch.tensor(
            [
                [.0, .0, 1.0],  # Inside: (5, 4). --> True.
                [-.5, -.4, 1.0],  # Top-left boundary: (0, 0). --> True.
                [-.6, 0.0, 1.0],  # Left of image.  --> False.
                [.5, .0, 1.0],  # Right boundary: u == width. --> False.
                [.0, -.5, 1.0],  # Above image. --> False.
                [.0, .4, 1.0],  # Bottom boundary: v == height. --> False.
                [.0, .0, .0],  # Zero depth. --> False.
                [.0, .0, -1.0],  # Behind camera. Confusion. --> False.
            ]
        )

        mask = get_fov_mask(points, projection_matrix, (8, 10))

        expected = torch.tensor([True, True, False, False, False, False, False, False])
        assert torch.equal(mask, expected)

    def test_empty_points(self) -> None:
        mask = get_fov_mask(torch.empty(0, 3), torch.eye(3, 4), (8, 10))

        assert mask.shape == (0,)
        assert mask.dtype == torch.bool

    def test_does_not_modify_inputs(self) -> None:
        points = torch.randn(5, 3)
        projection_matrix = torch.randn(3, 4)
        original_points = points.clone()
        original_projection_matrix = projection_matrix.clone()

        get_fov_mask(points, projection_matrix, (8, 10))

        assert torch.equal(points, original_points)
        assert torch.equal(projection_matrix, original_projection_matrix)

    def test_preserves_device(self, device: torch.device) -> None:
        points = torch.tensor([[.0, .0, 1.0]], device=device)
        projection_matrix = torch.eye(3, 4, device=device)

        mask = get_fov_mask(points, projection_matrix, (1, 1))

        assert mask.device == device
