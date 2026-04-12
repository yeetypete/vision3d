import pytest
import torch
from torchvision.tv_tensors import TVTensor

from vision3d.transforms.functional import register_kernel
from vision3d.transforms.functional._registry import _get_kernel


class TestRegisterKernel:
    def test_register_and_dispatch(self) -> None:
        class MyTensor(TVTensor):
            pass

        def my_functional[*S](inpt: torch.Tensor[*S]) -> torch.Tensor[*S]:
            return inpt

        kernel_was_called = False

        @register_kernel(my_functional, MyTensor)
        def my_kernel[*S](inpt: torch.Tensor[*S]) -> torch.Tensor[*S]:
            nonlocal kernel_was_called
            kernel_was_called = True
            return inpt

        t = MyTensor(torch.rand(3, 3))
        kernel = _get_kernel(my_functional, type(t))
        kernel(t)
        assert kernel_was_called

    def test_duplicate_registration_raises(self) -> None:
        class MyTensor(TVTensor):
            pass

        def my_functional[*S](inpt: torch.Tensor[*S]) -> torch.Tensor[*S]:
            return inpt

        register_kernel(my_functional, MyTensor)(lambda x: x)

        with pytest.raises(ValueError, match="already has a kernel"):
            register_kernel(my_functional, MyTensor)(lambda x: x)

    def test_tv_tensor_wrapper_auto_wraps(self) -> None:
        """With tv_tensor_wrapper=True, kernel gets pure tensor, output is re-wrapped."""

        class CustomTVTensor(TVTensor):
            pass

        def my_functional[*S](inpt: torch.Tensor[*S]) -> torch.Tensor[*S]:
            return inpt

        received_type = None

        @register_kernel(my_functional, CustomTVTensor)
        def my_kernel[*S](inpt: torch.Tensor[*S]) -> torch.Tensor[*S]:
            nonlocal received_type
            received_type = type(inpt)
            return inpt

        t = CustomTVTensor(torch.rand(3, 3))
        kernel = _get_kernel(my_functional, CustomTVTensor)
        out = kernel(t)

        assert received_type is torch.Tensor  # unwrapped
        assert isinstance(out, CustomTVTensor)  # re-wrapped

    def test_tv_tensor_wrapper_false(self) -> None:
        """With tv_tensor_wrapper=False, kernel gets the full TVTensor."""

        class CustomTVTensor(TVTensor):
            pass

        def my_functional[*S](inpt: torch.Tensor[*S]) -> torch.Tensor[*S]:
            return inpt

        received_type = None

        @register_kernel(my_functional, CustomTVTensor, tv_tensor_wrapper=False)
        def my_kernel[*S](inpt: torch.Tensor[*S]) -> torch.Tensor[*S]:
            nonlocal received_type
            received_type = type(inpt)
            return inpt

        t = CustomTVTensor(torch.rand(3, 3))
        kernel = _get_kernel(my_functional, CustomTVTensor)
        kernel(t)

        assert received_type is CustomTVTensor


class TestGetKernel:
    def test_passthrough_for_plain_tensor(self) -> None:
        def my_functional[*S](inpt: torch.Tensor[*S]) -> torch.Tensor[*S]:
            return inpt

        t = torch.rand(3, 3)
        kernel = _get_kernel(my_functional, type(t))
        out = kernel(t, some_kwarg=True)
        assert out is t

    def test_passthrough_for_unregistered_tv_tensor(self) -> None:
        class UnknownTensor(TVTensor):
            pass

        def my_functional[*S](inpt: torch.Tensor[*S]) -> torch.Tensor[*S]:
            return inpt

        t = UnknownTensor(torch.rand(3, 3))
        kernel = _get_kernel(my_functional, type(t))
        out = kernel(t)
        assert out is t

    def test_subclass_inherits_parent_kernel(self) -> None:
        class ParentTensor(TVTensor):
            pass

        class ChildTensor(ParentTensor):
            pass

        def my_functional[*S](inpt: torch.Tensor[*S]) -> torch.Tensor[*S]:
            return inpt

        parent_called = False

        @register_kernel(my_functional, ParentTensor, tv_tensor_wrapper=False)
        def parent_kernel[*S](inpt: torch.Tensor[*S]) -> torch.Tensor[*S]:
            nonlocal parent_called
            parent_called = True
            return inpt

        child = ChildTensor(torch.rand(3, 3))
        kernel = _get_kernel(my_functional, type(child))
        kernel(child)
        assert parent_called

    def test_exact_match_over_parent(self) -> None:
        class ParentTensor(TVTensor):
            pass

        class ChildTensor(ParentTensor):
            pass

        def my_functional[*S](inpt: torch.Tensor[*S]) -> torch.Tensor[*S]:
            return inpt

        child_called = False
        parent_called = False

        @register_kernel(my_functional, ParentTensor, tv_tensor_wrapper=False)
        def parent_kernel[*S](inpt: torch.Tensor[*S]) -> torch.Tensor[*S]:
            nonlocal parent_called
            parent_called = True
            return inpt

        @register_kernel(my_functional, ChildTensor, tv_tensor_wrapper=False)
        def child_kernel[*S](inpt: torch.Tensor[*S]) -> torch.Tensor[*S]:
            nonlocal child_called
            child_called = True
            return inpt

        child = ChildTensor(torch.rand(3, 3))
        kernel = _get_kernel(my_functional, type(child))
        kernel(child)
        assert child_called
        assert not parent_called
