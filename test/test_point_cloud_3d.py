from collections.abc import Callable, Generator
from copy import deepcopy

import pytest
import torch
from common_utils import make_point_cloud_3d
from torchvision import tv_tensors

from vision3d.tensors import PointCloud3D, wrap


@pytest.fixture(autouse=True)
def _restore_tensor_return_type() -> Generator[None]:
    yield
    tv_tensors.set_return_type("Tensor")


class TestConstruction:
    def test_instance(self) -> None:
        pc = make_point_cloud_3d(num_points=50)
        assert isinstance(pc, torch.Tensor)
        assert isinstance(pc, PointCloud3D)
        assert pc.shape == (50, 3)

    def test_with_features(self) -> None:
        pc = make_point_cloud_3d(num_points=10, num_features=4)
        assert pc.shape == (10, 7)

    def test_rejects_1d(self) -> None:
        with pytest.raises(ValueError, match="2D"):
            PointCloud3D(torch.rand(10))

    def test_rejects_3d(self) -> None:
        with pytest.raises(ValueError, match="2D"):
            PointCloud3D(torch.rand(2, 10, 3))

    def test_rejects_fewer_than_3_columns(self) -> None:
        with pytest.raises(ValueError, match="at least 3"):
            PointCloud3D(torch.rand(10, 2))

    @pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.float64])
    def test_accepts_floating_point(self, dtype: torch.dtype) -> None:
        PointCloud3D(torch.rand(5, 3, dtype=dtype))

    def test_rejects_integer(self) -> None:
        with pytest.raises(ValueError, match="floating point"):
            PointCloud3D(torch.randint(0, 10, (5, 3)))

    def test_dtype(self) -> None:
        pc = make_point_cloud_3d(dtype=torch.float64)
        assert pc.dtype == torch.float64

    def test_wrapping_no_copy(self) -> None:
        tensor = torch.rand(10, 3)
        pc = PointCloud3D(tensor)
        assert pc.data_ptr() == tensor.data_ptr()

    def test_repr(self) -> None:
        pc = make_point_cloud_3d(num_points=3)
        assert "PointCloud3D" in repr(pc)


class TestTorchFunction:
    @pytest.mark.parametrize("return_type", ["Tensor", "TVTensor"])
    def test_to_tv_tensor_reference(self, return_type: str) -> None:
        tensor = torch.rand((10, 3), dtype=torch.float64)
        pc = make_point_cloud_3d(num_points=10)

        with tv_tensors.set_return_type(return_type):
            tensor_to = tensor.to(pc)

        expected_type = type(pc) if return_type == "TVTensor" else torch.Tensor
        assert type(tensor_to) is expected_type
        assert tensor_to.dtype is pc.dtype
        assert type(tensor) is torch.Tensor

    @pytest.mark.parametrize("return_type", ["Tensor", "TVTensor"])
    def test_to_wrapping(self, return_type: str) -> None:
        pc = make_point_cloud_3d()
        with tv_tensors.set_return_type(return_type):
            pc_to = pc.to(torch.float64)
        assert type(pc_to) is type(pc)
        assert pc_to.dtype is torch.float64

    @pytest.mark.parametrize("return_type", ["Tensor", "TVTensor"])
    def test_clone_wrapping(self, return_type: str) -> None:
        pc = make_point_cloud_3d()
        with tv_tensors.set_return_type(return_type):
            pc_clone = pc.clone()
        assert type(pc_clone) is type(pc)
        assert pc_clone.data_ptr() != pc.data_ptr()

    @pytest.mark.parametrize("return_type", ["Tensor", "TVTensor"])
    def test_detach_wrapping(self, return_type: str) -> None:
        pc = make_point_cloud_3d(dtype=torch.float32)
        pc = pc.requires_grad_(True)
        with tv_tensors.set_return_type(return_type):
            pc_detached = pc.detach()
        assert type(pc_detached) is type(pc)

    @pytest.mark.parametrize("return_type", ["Tensor", "TVTensor"])
    def test_requires_grad_wrapping(self, return_type: str) -> None:
        pc = make_point_cloud_3d(dtype=torch.float32)
        assert not pc.requires_grad
        with tv_tensors.set_return_type(return_type):
            pc_rg = pc.requires_grad_(True)
        assert type(pc_rg) is type(pc)
        assert pc.requires_grad

    @pytest.mark.parametrize("return_type", ["Tensor", "TVTensor"])
    def test_other_op_no_wrapping(self, return_type: str) -> None:
        pc = make_point_cloud_3d()
        with tv_tensors.set_return_type(return_type):
            output = pc * 2
        expected_type = type(pc) if return_type == "TVTensor" else torch.Tensor
        assert type(output) is expected_type

    @pytest.mark.parametrize("return_type", ["Tensor", "TVTensor"])
    def test_inplace_op_no_wrapping(self, return_type: str) -> None:
        pc = make_point_cloud_3d()
        original_type = type(pc)
        with tv_tensors.set_return_type(return_type):
            output = pc.add_(0)
        expected_type = type(pc) if return_type == "TVTensor" else torch.Tensor
        assert type(output) is expected_type
        assert type(pc) is original_type

    @pytest.mark.parametrize(
        "op",
        [
            lambda p: p.cpu().numpy(),
            lambda p: p.tolist(),
            lambda p: p.max(dim=-1),
        ],
    )
    def test_no_tensor_output_op_no_wrapping(
        self, op: Callable[[PointCloud3D], object]
    ) -> None:
        pc = make_point_cloud_3d()
        output = op(pc)
        assert type(output) is not PointCloud3D

    @pytest.mark.parametrize("return_type", ["Tensor", "TVTensor"])
    @pytest.mark.parametrize(
        "op",
        [
            lambda p: p + torch.rand(*p.shape),
            lambda p: torch.rand(*p.shape) + p,
            lambda p: p * torch.rand(*p.shape),
            lambda p: p + 3,
            lambda p: p + p,
            lambda p: p.sum(),
            lambda p: p.reshape(-1),
            lambda p: p.int(),
            lambda p: torch.stack([p, p]),
            lambda p: torch.chunk(p, 2)[0],
        ],
    )
    def test_usual_operations(
        self, return_type: str, op: Callable[[PointCloud3D], object]
    ) -> None:
        pc = make_point_cloud_3d(num_points=10)
        with tv_tensors.set_return_type(return_type):
            out = op(pc)
        expected_type = type(pc) if return_type == "TVTensor" else torch.Tensor
        assert type(out) is expected_type

    @pytest.mark.parametrize("return_type", ["Tensor", "TVTensor"])
    def test_unbind(self, return_type: str) -> None:
        pc = make_point_cloud_3d(num_points=5)
        with tv_tensors.set_return_type(return_type):
            parts = torch.unbind(pc)
        expected_type = type(pc) if return_type == "TVTensor" else torch.Tensor
        for part in parts:
            assert type(part) is expected_type

    @pytest.mark.parametrize("requires_grad", [False, True])
    def test_deepcopy(self, requires_grad: bool) -> None:
        pc = make_point_cloud_3d(dtype=torch.float32)
        pc = pc.requires_grad_(requires_grad)

        pc_copy = deepcopy(pc)

        assert pc_copy is not pc
        assert pc_copy.data_ptr() != pc.data_ptr()
        assert torch.equal(pc_copy, pc)
        assert type(pc_copy) is PointCloud3D
        assert pc_copy.requires_grad is requires_grad


class TestWrap:
    def test_preserves_type_and_memory(self) -> None:
        pc = make_point_cloud_3d(num_points=10)
        output = pc * 2
        wrapped = wrap(output, like=pc)
        assert type(wrapped) is PointCloud3D
        assert wrapped.data_ptr() == output.data_ptr()

    def test_shares_memory(self) -> None:
        pc = make_point_cloud_3d(num_points=10)
        new_data = torch.rand(10, 3)
        wrapped = wrap(new_data, like=pc)
        assert wrapped.data_ptr() == new_data.data_ptr()

    def test_preserves_subclass(self) -> None:
        class MyPointCloud3D(PointCloud3D):
            pass

        pc = MyPointCloud3D(torch.rand(10, 3))
        output = pc * 2
        wrapped = wrap(output, like=pc)
        assert type(wrapped) is MyPointCloud3D
