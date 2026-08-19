"""Tests for :mod:`vision3d.viz._blueprint` panel layout.

A blueprint is a declarative container, so these assert the arrangement it
describes -- how many panels and how wide the grid -- rather than rendering it.
"""

import pytest

from vision3d.viz import camera_grid

# Blueprints are device-independent.
pytestmark = pytest.mark.skip_device("cuda")

_RIG = ("CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_BACK_RIGHT", "CAM_BACK", "CAM_BACK_LEFT")


def test_declared_grid_is_used_verbatim() -> None:
    blueprint = camera_grid(_RIG, ((4, 0, 1), (2, 3)))
    assert blueprint.grid_columns == 3
    assert len(list(blueprint.contents)) == 5


@pytest.mark.parametrize(
    ("num_cameras", "max_columns", "expected_columns"),
    [
        (1, 3, 1),  # narrower than a row: the grid shrinks to fit
        (3, 3, 3),
        (5, 3, 3),  # wraps to 3 + 2
        (5, 2, 2),  # wraps to 2 + 2 + 1
        (5, 9, 5),
    ],
)
def test_missing_grid_wraps_into_rows(
    num_cameras: int, max_columns: int, expected_columns: int
) -> None:
    """Without a declared layout the panels wrap in tensor order."""
    blueprint = camera_grid(_RIG[:num_cameras], None, max_columns=max_columns)
    assert blueprint.grid_columns == expected_columns
    assert len(list(blueprint.contents)) == num_cameras


def test_a_rig_with_no_cameras_yields_no_panels() -> None:
    blueprint = camera_grid((), None)
    assert len(list(blueprint.contents)) == 0


def test_max_columns_must_be_positive() -> None:
    with pytest.raises(ValueError, match="max_columns"):
        camera_grid(_RIG, None, max_columns=0)


def test_grid_index_out_of_range_is_rejected() -> None:
    with pytest.raises(ValueError, match="out of range"):
        camera_grid(_RIG, ((0, 99),))
