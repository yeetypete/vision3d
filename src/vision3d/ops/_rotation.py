"""Conversions between rotation matrices and unit quaternions."""

import torch
from torch import Tensor


def _quaternion_to_matrix(q: Tensor) -> Tensor:
    """Convert unit quaternions ``(w, x, y, z)`` to rotation matrices.

    Args:
        q: Quaternions ``[..., 4]``, scalar first.

    Returns:
        Rotation matrices ``[..., 3, 3]``.
    """
    w, x, y, z = q.unbind(-1)
    return torch.stack(
        [
            1 - 2 * (y * y + z * z),
            2 * (x * y - w * z),
            2 * (x * z + w * y),
            2 * (x * y + w * z),
            1 - 2 * (x * x + z * z),
            2 * (y * z - w * x),
            2 * (x * z - w * y),
            2 * (y * z + w * x),
            1 - 2 * (x * x + y * y),
        ],
        dim=-1,
    ).unflatten(-1, (3, 3))


def _matrix_to_quaternion(rot: Tensor) -> Tensor:
    """Convert rotation matrices to unit quaternions ``(w, x, y, z)``.

    Row ``i`` of the ``4x4`` candidate matrix is ``4 * q_i * q``. Picking the
    row with the largest diagonal ``4 * q_i**2`` avoids dividing by a value
    near zero.

    Args:
        rot: Rotation matrices ``[..., 3, 3]``.

    Returns:
        Quaternions ``[..., 4]``, scalar first, with an arbitrary sign.
    """
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = rot.flatten(-2).unbind(-1)
    candidates = torch.stack(
        [
            1 + m00 + m11 + m22,
            m21 - m12,
            m02 - m20,
            m10 - m01,
            m21 - m12,
            1 + m00 - m11 - m22,
            m01 + m10,
            m02 + m20,
            m02 - m20,
            m01 + m10,
            1 - m00 + m11 - m22,
            m12 + m21,
            m10 - m01,
            m02 + m20,
            m12 + m21,
            1 - m00 - m11 + m22,
        ],
        dim=-1,
    ).unflatten(-1, (4, 4))
    best = candidates.diagonal(dim1=-2, dim2=-1).argmax(-1)
    q = torch.take_along_dim(candidates, best[..., None, None], dim=-2).squeeze(-2)
    return q / q.norm(dim=-1, keepdim=True)
