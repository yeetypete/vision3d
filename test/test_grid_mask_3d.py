from copy import deepcopy
from typing import TYPE_CHECKING

import pytest
import torch
from common_utils import make_grid_affine, make_grid_mask_3d
from torchvision import tv_tensors

from vision3d.tensors import GridMask3D, wrap

if TYPE_CHECKING:
    from collections.abc import Callable


class TestConstruction:
    @pytest.mark.parametrize("spatial_shape", [(8, 8), (4, 8, 8)])
    def test_accepts_planar_and_volume(self, spatial_shape: tuple[int, ...]) -> None:
        mask = make_grid_mask_3d(channels=3, spatial_shape=spatial_shape)
        assert mask.shape == (3, *spatial_shape)

    @pytest.mark.parametrize("ndim", [1, 2, 5])
    def test_rejects_ndim(self, ndim: int) -> None:
        with pytest.raises(ValueError, match="3D tensor"):
            GridMask3D(torch.zeros((2,) * ndim), affine=torch.eye(4))

    @pytest.mark.parametrize(
        "dtype", [torch.bool, torch.uint8, torch.int64, torch.float32]
    )
    def test_accepts_any_dtype(self, dtype: torch.dtype) -> None:
        mask = GridMask3D(torch.zeros(1, 4, 4, dtype=dtype), affine=torch.eye(4))
        assert mask.dtype == dtype

    def test_affine_stored_as_cpu_float64(self) -> None:
        mask = GridMask3D(
            torch.zeros(1, 4, 4), affine=torch.eye(4, dtype=torch.float32)
        )
        assert mask.affine.dtype == torch.float64
        assert mask.affine.device == torch.device("cpu")

    def test_affine_is_copied(self) -> None:
        # If affine is float64 on the CPU, torch.as_tensor does not copy it.
        affine = torch.eye(4, dtype=torch.float64, device="cpu")
        mask = GridMask3D(torch.zeros(1, 4, 4), affine=affine)
        affine[0, 3] = 1.0
        torch.testing.assert_close(
            mask.affine, torch.eye(4, dtype=torch.float64, device="cpu")
        )

    def test_accepts_nested_list_affine(self) -> None:
        mask = GridMask3D(torch.zeros(1, 4, 4), affine=torch.eye(4).tolist())
        torch.testing.assert_close(
            mask.affine, torch.eye(4, dtype=torch.float64, device="cpu")
        )

    @pytest.mark.parametrize("shape", [(3, 3), (3, 4), (4, 4, 1)])
    def test_rejects_affine_shape(self, shape: tuple[int, ...]) -> None:
        with pytest.raises(ValueError, match=r"\[4, 4\] affine"):
            GridMask3D(torch.zeros(1, 4, 4), affine=torch.zeros(shape))

    def test_rejects_non_homogeneous_affine(self) -> None:
        affine = torch.eye(4)
        affine[3, 0] = 1.0
        with pytest.raises(ValueError, match="last affine row"):
            GridMask3D(torch.zeros(1, 4, 4), affine=affine)

    def test_wrapping_no_copy(self) -> None:
        tensor = torch.zeros(1, 4, 4, dtype=torch.int64)
        mask = GridMask3D(tensor, affine=torch.eye(4))
        assert mask.data_ptr() == tensor.data_ptr()

    def test_repr_includes_affine(self) -> None:
        mask = make_grid_mask_3d()
        assert "GridMask3D" in repr(mask)
        assert "affine=" in repr(mask)


class TestTorchFunction:
    @pytest.mark.parametrize("return_type", ["Tensor", "TVTensor"])
    def test_to_wrapping(self, return_type: str) -> None:
        mask = make_grid_mask_3d()
        with tv_tensors.set_return_type(return_type):
            mask_to = mask.to(torch.float64)
        assert type(mask_to) is type(mask)
        assert mask_to.dtype is torch.float64

    @pytest.mark.parametrize("return_type", ["Tensor", "TVTensor"])
    def test_clone_wrapping(self, return_type: str) -> None:
        mask = make_grid_mask_3d()
        with tv_tensors.set_return_type(return_type):
            mask_clone = mask.clone()
        assert type(mask_clone) is type(mask)
        assert mask_clone.data_ptr() != mask.data_ptr()

    @pytest.mark.parametrize("return_type", ["Tensor", "TVTensor"])
    def test_detach_wrapping(self, return_type: str) -> None:
        mask = make_grid_mask_3d(dtype=torch.float32)
        mask = mask.requires_grad_(True)
        with tv_tensors.set_return_type(return_type):
            mask_detached = mask.detach()
        assert type(mask_detached) is type(mask)

    @pytest.mark.parametrize("return_type", ["Tensor", "TVTensor"])
    def test_force_subclass_preserves_metadata(self, return_type: str) -> None:
        mask = make_grid_mask_3d(dtype=torch.float32)
        ops: list[Callable[[GridMask3D], torch.Tensor]] = [
            lambda m: m.clone(),
            lambda m: m.to(torch.float64),
            lambda m: m.detach(),
        ]
        with tv_tensors.set_return_type(return_type):
            for op in ops:
                result = op(mask)
                assert isinstance(result, GridMask3D)
                torch.testing.assert_close(result.affine, mask.affine)

    @pytest.mark.parametrize("return_type", ["Tensor", "TVTensor"])
    def test_other_op_no_wrapping(self, return_type: str) -> None:
        mask = make_grid_mask_3d()
        with tv_tensors.set_return_type(return_type):
            output = mask * 2
        expected_type = type(mask) if return_type == "TVTensor" else torch.Tensor
        assert type(output) is expected_type

    @pytest.mark.parametrize("return_type", ["Tensor", "TVTensor"])
    def test_unbind_preserves_metadata(self, return_type: str) -> None:
        mask = make_grid_mask_3d(channels=3)
        with tv_tensors.set_return_type(return_type):
            parts = torch.unbind(mask)
        expected_type = type(mask) if return_type == "TVTensor" else torch.Tensor
        for part in parts:
            assert type(part) is expected_type
        if return_type == "TVTensor":
            first = parts[0]
            assert isinstance(first, GridMask3D)
            torch.testing.assert_close(first.affine, mask.affine)

    def test_force_subclass_preserves_custom_subclass(self) -> None:
        class MyGridMask3D(GridMask3D):
            pass

        mask = MyGridMask3D(torch.zeros(2, 4, 4), affine=make_grid_affine())

        with tv_tensors.set_return_type("TVTensor"):
            clone = mask.clone()
            mask_to = mask.to(torch.float64)
            detached = mask.detach()
            first = torch.unbind(mask)[0]

        for output in [clone, mask_to, detached, first]:
            assert type(output) is MyGridMask3D
            torch.testing.assert_close(output.affine, mask.affine)

    @pytest.mark.parametrize("requires_grad", [False, True])
    def test_deepcopy(self, requires_grad: bool) -> None:
        mask = make_grid_mask_3d(dtype=torch.float32)
        mask = mask.requires_grad_(requires_grad)
        assert isinstance(mask, GridMask3D)

        mask_copy = deepcopy(mask)

        assert mask_copy is not mask
        assert mask_copy.data_ptr() != mask.data_ptr()
        assert torch.equal(mask_copy, mask)
        assert type(mask_copy) is GridMask3D
        torch.testing.assert_close(mask_copy.affine, mask.affine)
        assert mask_copy.requires_grad is requires_grad


class TestWrap:
    def test_preserves_type_and_affine(self) -> None:
        mask = make_grid_mask_3d()
        output = mask * 2
        wrapped = wrap(output, like=mask)
        assert type(wrapped) is GridMask3D
        assert wrapped.data_ptr() == output.data_ptr()
        torch.testing.assert_close(wrapped.affine, mask.affine)

    def test_override_affine(self) -> None:
        mask = make_grid_mask_3d()
        new_affine = make_grid_affine(cell_size=2.0)
        wrapped = wrap(mask.as_subclass(torch.Tensor), like=mask, affine=new_affine)
        assert isinstance(wrapped, GridMask3D)
        torch.testing.assert_close(wrapped.affine, new_affine.cpu())

    def test_shares_memory(self) -> None:
        mask = make_grid_mask_3d()
        new_data = torch.zeros(1, 8, 8, dtype=torch.int64)
        wrapped = wrap(new_data, like=mask)
        assert wrapped.data_ptr() == new_data.data_ptr()

    def test_preserves_subclass(self) -> None:
        class MyGridMask3D(GridMask3D):
            pass

        mask = MyGridMask3D(torch.zeros(1, 4, 4), affine=make_grid_affine())
        output = mask * 2
        wrapped = wrap(output, like=mask)
        assert type(wrapped) is MyGridMask3D
