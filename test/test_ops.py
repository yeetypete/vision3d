import pytest
import torch

from vision3d.ops import box3d_convert
from vision3d.tensors import BoundingBox3DFormat


class TestBox3dConvert:
    def test_same_format_clones(self) -> None:
        boxes = torch.tensor([[1.0, 2, 3, 4, 5, 6]])
        out = box3d_convert(
            boxes, BoundingBox3DFormat.XYZWHD, BoundingBox3DFormat.XYZWHD
        )
        assert torch.equal(out, boxes)
        assert out.data_ptr() != boxes.data_ptr()

    # XYZXYZ <-> XYZWHD
    def test_xyzxyz_to_xyzwhd(self) -> None:
        boxes = torch.tensor([[0.0, 0, 0, 10, 20, 30]])
        out = box3d_convert(
            boxes, BoundingBox3DFormat.XYZXYZ, BoundingBox3DFormat.XYZWHD
        )
        expected = torch.tensor([[5.0, 10, 15, 10, 20, 30]])
        torch.testing.assert_close(out, expected)

    def test_xyzwhd_to_xyzxyz(self) -> None:
        boxes = torch.tensor([[5.0, 10, 15, 10, 20, 30]])
        out = box3d_convert(
            boxes, BoundingBox3DFormat.XYZWHD, BoundingBox3DFormat.XYZXYZ
        )
        expected = torch.tensor([[0.0, 0, 0, 10, 20, 30]])
        torch.testing.assert_close(out, expected)

    def test_xyzxyz_xyzwhd_roundtrip(self) -> None:
        boxes = torch.tensor([[-5.0, -3, -1, 5, 3, 1]])
        roundtripped = box3d_convert(
            box3d_convert(
                boxes, BoundingBox3DFormat.XYZXYZ, BoundingBox3DFormat.XYZWHD
            ),
            BoundingBox3DFormat.XYZWHD,
            BoundingBox3DFormat.XYZXYZ,
        )
        torch.testing.assert_close(roundtripped, boxes)

    # Unsupported conversions
    @pytest.mark.parametrize(
        ("in_fmt", "out_fmt"),
        [
            (BoundingBox3DFormat.XYZXYZ, BoundingBox3DFormat.XYZWHDY),
            (BoundingBox3DFormat.XYZXYZ, BoundingBox3DFormat.XYZWHDYPR),
            (BoundingBox3DFormat.XYZWHD, BoundingBox3DFormat.XYZWHDY),
            (BoundingBox3DFormat.XYZWHD, BoundingBox3DFormat.XYZWHDYPR),
            (BoundingBox3DFormat.XYZWHDY, BoundingBox3DFormat.XYZXYZ),
            (BoundingBox3DFormat.XYZWHDY, BoundingBox3DFormat.XYZWHD),
            (BoundingBox3DFormat.XYZWHDY, BoundingBox3DFormat.XYZWHDYPR),
            (BoundingBox3DFormat.XYZWHDYPR, BoundingBox3DFormat.XYZXYZ),
            (BoundingBox3DFormat.XYZWHDYPR, BoundingBox3DFormat.XYZWHD),
            (BoundingBox3DFormat.XYZWHDYPR, BoundingBox3DFormat.XYZWHDY),
        ],
    )
    def test_unsupported_raises(
        self, in_fmt: BoundingBox3DFormat, out_fmt: BoundingBox3DFormat
    ) -> None:
        k = {
            BoundingBox3DFormat.XYZXYZ: 6,
            BoundingBox3DFormat.XYZWHD: 6,
            BoundingBox3DFormat.XYZWHDY: 7,
            BoundingBox3DFormat.XYZWHDYPR: 9,
        }[in_fmt]
        boxes = torch.rand(1, k)
        with pytest.raises(NotImplementedError, match="not supported"):
            box3d_convert(boxes, in_fmt, out_fmt)

    # String format support
    def test_string_formats(self) -> None:
        boxes = torch.tensor([[0.0, 0, 0, 10, 20, 30]])
        out = box3d_convert(boxes, "xyzxyz", "xyzwhd")
        expected = torch.tensor([[5.0, 10, 15, 10, 20, 30]])
        torch.testing.assert_close(out, expected)

    # Batch dimensions
    def test_batch_dims(self) -> None:
        boxes = torch.rand(2, 5, 6)
        out = box3d_convert(
            boxes, BoundingBox3DFormat.XYZXYZ, BoundingBox3DFormat.XYZWHD
        )
        assert out.shape == (2, 5, 6)

    def test_single_box(self) -> None:
        boxes = torch.tensor([0.0, 0, 0, 10, 20, 30])
        out = box3d_convert(
            boxes, BoundingBox3DFormat.XYZXYZ, BoundingBox3DFormat.XYZWHD
        )
        expected = torch.tensor([5.0, 10, 15, 10, 20, 30])
        torch.testing.assert_close(out, expected)
