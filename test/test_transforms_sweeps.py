"""Tests for multi-sweep aggregation."""

import numpy as np
import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from vision3d.transforms.functional import accumulate_sweeps

_elements = st.floats(
    min_value=-30.0, max_value=30.0, allow_nan=False, allow_infinity=False, width=32
)


@st.composite
def _problems(
    draw: st.DrawFn,
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    # Draw a (sweeps, transforms, time_offsets) problem of random shape.
    num_sweeps = draw(st.integers(min_value=1, max_value=4))
    num_features = draw(st.integers(min_value=0, max_value=3))
    counts = draw(
        st.lists(
            st.integers(min_value=0, max_value=6),
            min_size=num_sweeps,
            max_size=num_sweeps,
        )
    )
    sweeps = [
        draw(arrays(np.float32, (n, 3 + num_features), elements=_elements))
        for n in counts
    ]
    transforms = draw(arrays(np.float32, (num_sweeps, 4, 4), elements=_elements))
    time_offsets = draw(arrays(np.float32, (num_sweeps,), elements=_elements))
    return sweeps, transforms, time_offsets


def _accumulate_sweeps_reference(
    sweeps: list[np.ndarray], transforms: np.ndarray, time_offsets: np.ndarray
) -> np.ndarray:
    # Independent float64 reference for accumulate_sweeps. Applies each
    # transform as a homogeneous [3, 4] operator (a different formulation than
    # the implementation's xyz @ R.T + t) and appends the time offset, so the
    # comparison checks the math rather than a copy of it.
    rows: list[np.ndarray] = []
    for points, transform, dt in zip(sweeps, transforms, time_offsets):
        xyz = points[:, :3].astype(np.float64)
        homogeneous = np.concatenate([xyz, np.ones((xyz.shape[0], 1))], axis=1)
        moved = homogeneous @ transform[:3, :].astype(np.float64).T
        features = points[:, 3:].astype(np.float64)
        times = np.full((points.shape[0], 1), float(dt))
        rows.append(np.concatenate([moved, features, times], axis=1))
    return np.concatenate(rows, axis=0)


class TestAccumulateSweeps:
    @pytest.mark.skip_device("cuda")
    @settings(deadline=None, max_examples=200)
    @given(problem=_problems())
    def test_matches_numpy_reference(
        self, problem: tuple[list[np.ndarray], np.ndarray, np.ndarray]
    ) -> None:
        sweeps, transforms, time_offsets = problem
        out = accumulate_sweeps(
            [torch.from_numpy(s) for s in sweeps],
            torch.from_numpy(transforms),
            torch.from_numpy(time_offsets),
        )
        ref = torch.from_numpy(
            _accumulate_sweeps_reference(sweeps, transforms, time_offsets)
        )
        torch.testing.assert_close(out.double(), ref, atol=1e-2, rtol=1e-4)

    def test_does_not_modify_input(self) -> None:
        points = torch.randn(5, 5)
        original = points.clone()
        accumulate_sweeps([points], torch.eye(4)[None], torch.tensor([0.0]))
        assert torch.equal(points, original)

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one sweep"):
            accumulate_sweeps([], torch.empty(0, 4, 4), torch.empty(0))

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="one transform and time offset"):
            accumulate_sweeps(
                [torch.randn(3, 5), torch.randn(2, 5)],
                torch.eye(4).expand(2, 4, 4),
                torch.tensor([0.0]),
            )
