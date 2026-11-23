"""Tests for the rotation-matrix/quaternion conversions in :mod:`vision3d.ops`."""

import torch
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from pyquaternion import Quaternion

from vision3d.ops._rotation import (
    _quaternion_to_rotation_matrix,
    _rotation_matrix_to_quaternion,
)


@settings(deadline=None)
@given(
    components=st.lists(
        st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        min_size=4,
        max_size=4,
    )
)
def test_conversions_match_pyquaternion(components: list[float]) -> None:
    raw = torch.tensor(components, dtype=torch.float64)
    # A near-zero vector has no well-defined rotation.
    assume(float(raw.norm()) > 1e-6)
    q = raw / raw.norm()
    ref = torch.as_tensor(Quaternion(q.tolist()).rotation_matrix)

    # Non-unit quaternions are normalized.
    torch.testing.assert_close(_quaternion_to_rotation_matrix(raw), ref)

    # q and -q are the same rotation.
    back = _rotation_matrix_to_quaternion(ref)
    torch.testing.assert_close(back * torch.sign(back @ q), q)
