from copy import deepcopy
from typing import TYPE_CHECKING

import pytest
import torch
import torch.cuda
from common_utils import make_bounding_boxes_3d
from torchvision import tv_tensors

from vision3d.tensors import BoundingBox3DFormat, BoundingBoxes3D, wrap

if TYPE_CHECKING:
    from collections.abc import Callable, Generator


@pytest.fixture(autouse=True)
def _restore_tensor_return_type() -> Generator[None]:
    yield
    tv_tensors.set_return_type("Tensor")


class TestBoundingBox3DFormat:
    @pytest.mark.parametrize(
        ("format", "is_rotated_expected"),
        [
            (BoundingBox3DFormat.XYZXYZ, False),
            (BoundingBox3DFormat.XYZLWH, False),
            (BoundingBox3DFormat.XYZLWHY, True),
            (BoundingBox3DFormat.XYZLWHYPR, True),
        ],
    )
    def test_is_rotated(
        self, format: BoundingBox3DFormat, is_rotated_expected: bool
    ) -> None:
        assert BoundingBox3DFormat.is_rotated(format) == is_rotated_expected


FORMATS_AND_K = [
    (BoundingBox3DFormat.XYZXYZ, 6),
    (BoundingBox3DFormat.XYZLWH, 6),
    (BoundingBox3DFormat.XYZLWHY, 7),
    (BoundingBox3DFormat.XYZLWHYPR, 9),
]


class TestConstruction:
    @pytest.mark.parametrize(("format", "k"), FORMATS_AND_K)
    def test_instance(self, format: BoundingBox3DFormat, k: int) -> None:
        bbox = make_bounding_boxes_3d(format=format, num_boxes=5)
        assert isinstance(bbox, torch.Tensor)
        assert isinstance(bbox, BoundingBoxes3D)
        assert bbox.ndim == 2
        assert bbox.shape == (5, k)
        assert bbox.format == format

    @pytest.mark.parametrize(
        "format",
        ["XYZXYZ", "XYZLWH", "XYZLWHY", "XYZLWHYPR"],
    )
    def test_format_from_string(self, format: str) -> None:
        bbox = BoundingBoxes3D(torch.rand(1, 9), format=format)
        assert bbox.format == BoundingBox3DFormat[format]

    def test_1d_unsqueezed_to_2d(self) -> None:
        bbox = BoundingBoxes3D(
            [1, 2, 3, 4, 5, 6, 0.1, 0.2, 0.3],
            format=BoundingBox3DFormat.XYZLWHYPR,
        )
        assert bbox.ndim == 2
        assert bbox.shape == (1, 9)

    @pytest.mark.parametrize(
        ("data", "input_requires_grad", "expected_requires_grad"),
        [
            ([[0.0] * 9], None, False),
            ([[0.0] * 9], False, False),
            ([[0.0] * 9], True, True),
            (torch.rand(1, 9, requires_grad=False), None, False),
            (torch.rand(1, 9, requires_grad=False), False, False),
            (torch.rand(1, 9, requires_grad=False), True, True),
            (torch.rand(1, 9, requires_grad=True), None, True),
            (torch.rand(1, 9, requires_grad=True), False, False),
            (torch.rand(1, 9, requires_grad=True), True, True),
        ],
    )
    def test_new_requires_grad(
        self,
        data: list[list[float]] | torch.Tensor[1, 9],
        input_requires_grad: bool | None,
        expected_requires_grad: bool,
    ) -> None:
        bbox = BoundingBoxes3D(
            data,
            format=BoundingBox3DFormat.XYZLWHYPR,
            requires_grad=input_requires_grad,
        )
        assert bbox.requires_grad is expected_requires_grad

    def test_ndim_validation(self) -> None:
        with pytest.raises(ValueError, match="1D or 2D"):
            BoundingBoxes3D(torch.rand(2, 3, 9), format=BoundingBox3DFormat.XYZLWHYPR)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.float64])
    def test_accepts_floating_point(self, dtype: torch.dtype) -> None:
        BoundingBoxes3D(
            torch.zeros(1, 9, dtype=dtype), format=BoundingBox3DFormat.XYZLWHYPR
        )

    @pytest.mark.parametrize(("format", "k"), FORMATS_AND_K)
    def test_rejects_integer(self, format: BoundingBox3DFormat, k: int) -> None:
        with pytest.raises(ValueError, match="floating point"):
            BoundingBoxes3D(torch.zeros(1, k, dtype=torch.int32), format=format)

    def test_wrapping_no_copy(self) -> None:
        tensor = torch.rand(3, 9)
        bbox = BoundingBoxes3D(tensor, format=BoundingBox3DFormat.XYZLWHYPR)
        assert bbox.data_ptr() == tensor.data_ptr()

    def test_repr_includes_format(self) -> None:
        bbox = make_bounding_boxes_3d(format=BoundingBox3DFormat.XYZLWHYPR)
        assert "XYZLWHYPR" in repr(bbox)


class TestTorchFunction:
    @pytest.mark.parametrize("return_type", ["Tensor", "TVTensor"])
    def test_to_tv_tensor_reference(self, return_type: str) -> None:
        tensor = torch.rand((2, 9), dtype=torch.float64)
        bbox = make_bounding_boxes_3d(num_boxes=2)

        with tv_tensors.set_return_type(return_type):
            tensor_to = tensor.to(bbox)

        expected_type = type(bbox) if return_type == "TVTensor" else torch.Tensor
        assert type(tensor_to) is expected_type
        assert tensor_to.dtype is bbox.dtype
        assert type(tensor) is torch.Tensor

    @pytest.mark.parametrize("return_type", ["Tensor", "TVTensor"])
    def test_to_wrapping(self, return_type: str) -> None:
        bbox = make_bounding_boxes_3d()
        with tv_tensors.set_return_type(return_type):
            bbox_to = bbox.to(torch.float64)
        assert type(bbox_to) is type(bbox)
        assert bbox_to.dtype is torch.float64

    @pytest.mark.parametrize("return_type", ["Tensor", "TVTensor"])
    def test_clone_wrapping(self, return_type: str) -> None:
        bbox = make_bounding_boxes_3d()
        with tv_tensors.set_return_type(return_type):
            bbox_clone = bbox.clone()
        assert type(bbox_clone) is type(bbox)
        assert bbox_clone.data_ptr() != bbox.data_ptr()

    @pytest.mark.parametrize("return_type", ["Tensor", "TVTensor"])
    def test_detach_wrapping(self, return_type: str) -> None:
        bbox = make_bounding_boxes_3d(dtype=torch.float32)
        bbox = bbox.requires_grad_(True)
        with tv_tensors.set_return_type(return_type):
            bbox_detached = bbox.detach()
        assert type(bbox_detached) is type(bbox)

    @pytest.mark.parametrize("return_type", ["Tensor", "TVTensor"])
    def test_requires_grad_wrapping(self, return_type: str) -> None:
        bbox = make_bounding_boxes_3d(dtype=torch.float32)
        assert not bbox.requires_grad
        with tv_tensors.set_return_type(return_type):
            bbox_rg = bbox.requires_grad_(True)
        assert type(bbox_rg) is type(bbox)
        assert bbox.requires_grad

    @pytest.mark.parametrize("return_type", ["Tensor", "TVTensor"])
    def test_force_subclass_preserves_metadata(self, return_type: str) -> None:
        bbox = make_bounding_boxes_3d(format=BoundingBox3DFormat.XYZLWHYPR)
        original_format = bbox.format

        tv_tensors.set_return_type(return_type)

        for op in [
            lambda b: b.clone(),
            lambda b: b.to(torch.float64),
            lambda b: b.detach(),
        ]:
            result = op(bbox)
            assert result.format == original_format

        if torch.cuda.is_available():
            pinned = bbox.pin_memory()
            if return_type == "TVTensor":
                assert type(pinned) is type(bbox)
                assert pinned.format == original_format

        tv_tensors.set_return_type("Tensor")

    @pytest.mark.parametrize("return_type", ["Tensor", "TVTensor"])
    def test_other_op_no_wrapping(self, return_type: str) -> None:
        bbox = make_bounding_boxes_3d()
        with tv_tensors.set_return_type(return_type):
            output = bbox * 2
        expected_type = type(bbox) if return_type == "TVTensor" else torch.Tensor
        assert type(output) is expected_type

    @pytest.mark.parametrize("return_type", ["Tensor", "TVTensor"])
    def test_inplace_op_no_wrapping(self, return_type: str) -> None:
        bbox = make_bounding_boxes_3d()
        original_type = type(bbox)
        with tv_tensors.set_return_type(return_type):
            output = bbox.add_(0)
        expected_type = type(bbox) if return_type == "TVTensor" else torch.Tensor
        assert type(output) is expected_type
        assert type(bbox) is original_type

    @pytest.mark.parametrize(
        "op",
        [
            lambda b: b.numpy(),
            lambda b: b.tolist(),
            lambda b: b.max(dim=-1),
        ],
    )
    def test_no_tensor_output_op_no_wrapping(
        self, op: Callable[[BoundingBoxes3D], object]
    ) -> None:
        bbox = make_bounding_boxes_3d()
        output = op(bbox)
        assert type(output) is not BoundingBoxes3D

    @pytest.mark.parametrize("return_type", ["Tensor", "TVTensor"])
    @pytest.mark.parametrize(
        "op",
        [
            lambda b: b + torch.rand(*b.shape),
            lambda b: torch.rand(*b.shape) + b,
            lambda b: b * torch.rand(*b.shape),
            lambda b: b + 3,
            lambda b: b + b,
            lambda b: b.sum(),
            lambda b: b.reshape(-1),
            lambda b: b.int(),
            lambda b: torch.stack([b, b]),
            lambda b: torch.chunk(b, 2)[0],
        ],
    )
    def test_usual_operations(
        self, return_type: str, op: Callable[[BoundingBoxes3D], object]
    ) -> None:
        bbox = make_bounding_boxes_3d(num_boxes=2)
        with tv_tensors.set_return_type(return_type):
            out = op(bbox)
        expected_type = type(bbox) if return_type == "TVTensor" else torch.Tensor
        assert type(out) is expected_type
        if return_type == "TVTensor":
            assert hasattr(out, "format")

    @pytest.mark.parametrize("return_type", ["Tensor", "TVTensor"])
    def test_unbind_preserves_metadata(self, return_type: str) -> None:
        bbox = make_bounding_boxes_3d(num_boxes=3)
        with tv_tensors.set_return_type(return_type):
            parts = torch.unbind(bbox)
        expected_type = type(bbox) if return_type == "TVTensor" else torch.Tensor
        for part in parts:
            assert type(part) is expected_type
        if return_type == "TVTensor":
            first = parts[0]
            assert isinstance(first, BoundingBoxes3D)
            assert first.format == bbox.format

    def test_force_subclass_preserves_custom_subclass(self) -> None:
        class MyBoundingBoxes3D(BoundingBoxes3D):
            pass

        bbox = MyBoundingBoxes3D(torch.rand(2, 9), format=BoundingBox3DFormat.XYZLWHYPR)

        with tv_tensors.set_return_type("TVTensor"):
            clone = bbox.clone()
            bbox_to = bbox.to(torch.float64)
            detached = bbox.detach()
            first = torch.unbind(bbox)[0]

        for output in [clone, bbox_to, detached, first]:
            assert type(output) is MyBoundingBoxes3D
            assert output.format == bbox.format

    @pytest.mark.parametrize("requires_grad", [False, True])
    def test_deepcopy(self, requires_grad: bool) -> None:
        bbox = make_bounding_boxes_3d(dtype=torch.float32)
        bbox = bbox.requires_grad_(requires_grad)
        assert isinstance(bbox, BoundingBoxes3D)
        original_format = bbox.format

        bbox_copy = deepcopy(bbox)

        assert bbox_copy is not bbox
        assert bbox_copy.data_ptr() != bbox.data_ptr()
        assert torch.equal(bbox_copy, bbox)
        assert type(bbox_copy) is BoundingBoxes3D
        assert isinstance(bbox_copy, BoundingBoxes3D)
        assert bbox_copy.format == original_format
        assert bbox_copy.requires_grad is requires_grad


class TestWrap:
    def test_preserves_type_and_format(self) -> None:
        bbox = make_bounding_boxes_3d(format=BoundingBox3DFormat.XYZLWHYPR)
        output = bbox * 2
        wrapped = wrap(output, like=bbox)
        assert type(wrapped) is BoundingBoxes3D
        assert wrapped.data_ptr() == output.data_ptr()
        assert wrapped.format == BoundingBox3DFormat.XYZLWHYPR

    def test_override_format(self) -> None:
        bbox = make_bounding_boxes_3d(format=BoundingBox3DFormat.XYZLWHYPR)
        new_data = torch.rand(1, 6)
        wrapped = wrap(new_data, like=bbox, format=BoundingBox3DFormat.XYZXYZ)
        assert isinstance(wrapped, BoundingBoxes3D)
        assert wrapped.format == BoundingBox3DFormat.XYZXYZ

    def test_shares_memory(self) -> None:
        bbox = make_bounding_boxes_3d(format=BoundingBox3DFormat.XYZLWH)
        new_data = torch.rand(1, 6)
        wrapped = wrap(new_data, like=bbox)
        assert wrapped.data_ptr() == new_data.data_ptr()

    def test_preserves_subclass(self) -> None:
        class MyBoundingBoxes3D(BoundingBoxes3D):
            pass

        bbox = MyBoundingBoxes3D(torch.rand(1, 9), format=BoundingBox3DFormat.XYZLWHYPR)
        output = bbox * 2
        wrapped = wrap(output, like=bbox)
        assert type(wrapped) is MyBoundingBoxes3D
