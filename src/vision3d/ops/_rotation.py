"""Conversions between rotation matrices and unit quaternions."""

import torch
from torch import Tensor


def _quaternion_to_rotation_matrix(rotation_wxyz: Tensor) -> Tensor:
    """Convert one or more quaternions ``(w, x, y, z)`` to rotation matrices.

    Args:
        rotation_wxyz: Quaternions ``[..., 4]``, scalar first.

    Returns:
        Rotation matrices ``[..., 3, 3]``.
    """
    w, x, y, z = rotation_wxyz.unbind(-1)
    s = 2 / (rotation_wxyz * rotation_wxyz).sum(-1)
    return torch.stack(
        [
            1 - s * (y * y + z * z),
            s * (x * y - w * z),
            s * (x * z + w * y),
            s * (x * y + w * z),
            1 - s * (x * x + z * z),
            s * (y * z - w * x),
            s * (x * z - w * y),
            s * (y * z + w * x),
            1 - s * (x * x + y * y),
        ],
        dim=-1,
    ).unflatten(-1, (3, 3))


def _rotation_matrix_to_quaternion(rotation_matrix: Tensor) -> Tensor:
    """Convert one or more rotation matrices to unit quaternions ``(w, x, y, z)``.

    Row ``i`` of the ``4x4`` candidate matrix is ``4 * q_i * q``. Picking the
    row with the largest diagonal ``4 * q_i**2`` avoids dividing by a value
    near zero.

    Args:
        rotation_matrix: Rotation matrices ``[..., 3, 3]``.

    Returns:
        Quaternions ``[..., 4]``, scalar first, with an arbitrary sign.
    """
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = rotation_matrix.flatten(-2).unbind(-1)
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
