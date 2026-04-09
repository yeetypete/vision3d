from copy import deepcopy
from typing import TYPE_CHECKING

import pytest
import torch
from common_utils import (
    make_camera_extrinsics,
    make_camera_images,
    make_camera_intrinsics,
)
from torchvision import tv_tensors

from vision3d.tensors import CameraExtrinsics, CameraImages, CameraIntrinsics

if TYPE_CHECKING:
    from collections.abc import Generator


@pytest.fixture(autouse=True)
def _restore_tensor_return_type() -> Generator[None]:
    yield
    tv_tensors.set_return_type("Tensor")


class TestCameraImagesConstruction:
    def test_instance(self) -> None:
        imgs = make_camera_images(num_cameras=6)
        assert isinstance(imgs, torch.Tensor)
        assert isinstance(imgs, CameraImages)
        assert imgs.shape == (6, 3, 224, 224)

    def test_single_camera(self) -> None:
        imgs = CameraImages(torch.rand(1, 3, 32, 32))
        assert imgs.shape == (1, 3, 32, 32)

    def test_rejects_2d(self) -> None:
        with pytest.raises(ValueError, match="at least 3D"):
            CameraImages(torch.rand(3, 224))

    def test_wrapping_no_copy(self) -> None:
        tensor = torch.rand(2, 3, 32, 32)
        imgs = CameraImages(tensor)
        assert imgs.data_ptr() == tensor.data_ptr()


class TestCameraImagesTorchFunction:
    @pytest.mark.parametrize("return_type", ["Tensor", "TVTensor"])
    def test_clone_wrapping(self, return_type: str) -> None:
        imgs = make_camera_images(num_cameras=2, height=32, width=32)
        with tv_tensors.set_return_type(return_type):
            clone = imgs.clone()
        assert type(clone) is type(imgs)

    @pytest.mark.parametrize("return_type", ["Tensor", "TVTensor"])
    def test_other_op_no_wrapping(self, return_type: str) -> None:
        imgs = make_camera_images(num_cameras=2, height=32, width=32)
        with tv_tensors.set_return_type(return_type):
            out = imgs * 2
        expected_type = type(imgs) if return_type == "TVTensor" else torch.Tensor
        assert type(out) is expected_type

    def test_deepcopy(self) -> None:
        imgs = make_camera_images(num_cameras=2, height=32, width=32)
        copy = deepcopy(imgs)
        assert copy is not imgs
        assert type(copy) is CameraImages
        assert torch.equal(copy, imgs)


class TestCameraExtrinsicsConstruction:
    def test_instance(self) -> None:
        ext = make_camera_extrinsics(num_cameras=6)
        assert isinstance(ext, torch.Tensor)
        assert isinstance(ext, CameraExtrinsics)
        assert ext.shape == (6, 4, 4)

    def test_single_camera(self) -> None:
        ext = CameraExtrinsics(torch.eye(4))
        assert ext.shape == (4, 4)

    def test_rejects_wrong_shape(self) -> None:
        with pytest.raises(ValueError, match="4, 4"):
            CameraExtrinsics(torch.rand(6, 3, 3))

    def test_rejects_1d(self) -> None:
        with pytest.raises(ValueError, match="4, 4"):
            CameraExtrinsics(torch.rand(16))

    def test_rejects_integer(self) -> None:
        with pytest.raises(ValueError, match="floating point"):
            CameraExtrinsics(torch.eye(4, dtype=torch.int32))

    def test_wrapping_no_copy(self) -> None:
        tensor = torch.eye(4).unsqueeze(0)
        ext = CameraExtrinsics(tensor)
        assert ext.data_ptr() == tensor.data_ptr()


class TestCameraExtrinsicsTorchFunction:
    @pytest.mark.parametrize("return_type", ["Tensor", "TVTensor"])
    def test_clone_wrapping(self, return_type: str) -> None:
        ext = make_camera_extrinsics()
        with tv_tensors.set_return_type(return_type):
            clone = ext.clone()
        assert type(clone) is type(ext)

    @pytest.mark.parametrize("return_type", ["Tensor", "TVTensor"])
    def test_other_op_no_wrapping(self, return_type: str) -> None:
        ext = make_camera_extrinsics()
        with tv_tensors.set_return_type(return_type):
            out = ext * 2
        expected_type = type(ext) if return_type == "TVTensor" else torch.Tensor
        assert type(out) is expected_type

    def test_deepcopy(self) -> None:
        ext = make_camera_extrinsics()
        copy = deepcopy(ext)
        assert copy is not ext
        assert type(copy) is CameraExtrinsics
        assert torch.equal(copy, ext)


class TestCameraIntrinsicsConstruction:
    def test_instance(self) -> None:
        intr = make_camera_intrinsics(num_cameras=6)
        assert isinstance(intr, torch.Tensor)
        assert isinstance(intr, CameraIntrinsics)
        assert intr.shape == (6, 3, 3)

    def test_single_camera(self) -> None:
        intr = CameraIntrinsics(torch.eye(3), image_size=(480, 640))
        assert intr.shape == (3, 3)
        assert intr.image_size == (480, 640)

    def test_rejects_wrong_shape(self) -> None:
        with pytest.raises(ValueError, match="3, 3"):
            CameraIntrinsics(torch.rand(6, 4, 4), image_size=(480, 640))

    def test_rejects_1d(self) -> None:
        with pytest.raises(ValueError, match="3, 3"):
            CameraIntrinsics(torch.rand(9), image_size=(480, 640))

    def test_rejects_integer(self) -> None:
        with pytest.raises(ValueError, match="floating point"):
            CameraIntrinsics(torch.eye(3, dtype=torch.int32), image_size=(480, 640))

    def test_wrapping_no_copy(self) -> None:
        tensor = torch.eye(3).unsqueeze(0)
        intr = CameraIntrinsics(tensor, image_size=(480, 640))
        assert intr.data_ptr() == tensor.data_ptr()


class TestCameraIntrinsicsTorchFunction:
    @pytest.mark.parametrize("return_type", ["Tensor", "TVTensor"])
    def test_clone_wrapping(self, return_type: str) -> None:
        intr = make_camera_intrinsics()
        with tv_tensors.set_return_type(return_type):
            clone = intr.clone()
        assert type(clone) is type(intr)

    @pytest.mark.parametrize("return_type", ["Tensor", "TVTensor"])
    def test_other_op_no_wrapping(self, return_type: str) -> None:
        intr = make_camera_intrinsics()
        with tv_tensors.set_return_type(return_type):
            out = intr * 2
        expected_type = type(intr) if return_type == "TVTensor" else torch.Tensor
        assert type(out) is expected_type

    def test_deepcopy(self) -> None:
        intr = make_camera_intrinsics()
        copy = deepcopy(intr)
        assert copy is not intr
        assert type(copy) is CameraIntrinsics
        assert torch.equal(copy, intr)
