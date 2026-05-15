"""Tests for nms_3d and batched_nms_3d."""

import pytest
import torch
from common_utils import box_at

from vision3d.ops import batched_nms_3d, box3d_iou, nms_3d
from vision3d.tensors import BoundingBox3DFormat

_ALL_FORMATS = [
    BoundingBox3DFormat.XYZXYZ,
    BoundingBox3DFormat.XYZLWH,
    BoundingBox3DFormat.XYZLWHY,
    BoundingBox3DFormat.XYZLWHYPR,
]

_NUM_COLS = {
    BoundingBox3DFormat.XYZXYZ: 6,
    BoundingBox3DFormat.XYZLWH: 6,
    BoundingBox3DFormat.XYZLWHY: 7,
    BoundingBox3DFormat.XYZLWHYPR: 9,
}


def _stack(rows: list[list[float]]) -> torch.Tensor:
    return torch.tensor(rows, dtype=torch.float32)


@pytest.mark.parametrize("fmt", _ALL_FORMATS)
class TestNms3dBasics:
    def test_empty_input(self, fmt: BoundingBox3DFormat) -> None:
        boxes = torch.zeros((0, _NUM_COLS[fmt]), dtype=torch.float32)
        scores = torch.zeros(0)
        kept = nms_3d(boxes, scores, 0.5, fmt)
        assert kept.dtype == torch.long
        assert kept.numel() == 0

    def test_single_box(self, fmt: BoundingBox3DFormat) -> None:
        boxes = _stack([box_at(0, 0, fmt=fmt)])
        scores = torch.tensor([0.9])
        kept = nms_3d(boxes, scores, 0.5, fmt)
        assert kept.tolist() == [0]

    def test_disjoint_boxes_all_kept(self, fmt: BoundingBox3DFormat) -> None:
        boxes = _stack(
            [box_at(0, 0, fmt=fmt), box_at(10, 0, fmt=fmt), box_at(-10, 0, fmt=fmt)]
        )
        scores = torch.tensor([0.5, 0.9, 0.7])
        kept = nms_3d(boxes, scores, 0.5, fmt)
        # All three survive, indices returned in descending score order.
        assert sorted(kept.tolist()) == [0, 1, 2]
        assert kept.tolist() == [1, 2, 0]

    def test_identical_boxes_only_highest_kept(self, fmt: BoundingBox3DFormat) -> None:
        boxes = _stack(
            [box_at(0, 0, fmt=fmt), box_at(0, 0, fmt=fmt), box_at(0, 0, fmt=fmt)]
        )
        scores = torch.tensor([0.5, 0.9, 0.7])
        kept = nms_3d(boxes, scores, 0.5, fmt)
        assert kept.tolist() == [1]


@pytest.mark.parametrize("fmt", _ALL_FORMATS)
class TestNms3dThreshold:
    def test_partial_overlap_above_threshold_suppresses(
        self, fmt: BoundingBox3DFormat
    ) -> None:
        # Two boxes sharing the same center → IoU 1.0; strictly greater
        # than any threshold < 1 → second box is suppressed.
        boxes = _stack([box_at(0, 0, fmt=fmt), box_at(0, 0, fmt=fmt)])
        scores = torch.tensor([0.9, 0.8])
        kept = nms_3d(boxes, scores, 0.5, fmt)
        assert kept.tolist() == [0]

    def test_partial_overlap_below_threshold_keeps_both(
        self, fmt: BoundingBox3DFormat
    ) -> None:
        # Shift by 1.0 on a 2x2x2 box → IoU = 1/3 ≈ 0.333.
        # Threshold 0.5 → keep both.
        boxes = _stack([box_at(0, 0, fmt=fmt), box_at(1.0, 0, fmt=fmt)])
        scores = torch.tensor([0.9, 0.8])
        kept = nms_3d(boxes, scores, 0.5, fmt)
        assert sorted(kept.tolist()) == [0, 1]


@pytest.mark.parametrize("fmt", _ALL_FORMATS)
class TestNms3dOrdering:
    def test_output_sorted_by_descending_score(self, fmt: BoundingBox3DFormat) -> None:
        boxes = _stack(
            [
                box_at(0, 0, fmt=fmt),
                box_at(10, 0, fmt=fmt),
                box_at(20, 0, fmt=fmt),
                box_at(30, 0, fmt=fmt),
            ]
        )
        scores = torch.tensor([0.2, 0.9, 0.5, 0.7])
        kept = nms_3d(boxes, scores, 0.5, fmt)
        # All disjoint so all survive; order must be (0.9, 0.7, 0.5, 0.2).
        assert kept.tolist() == [1, 3, 2, 0]


@pytest.mark.parametrize("fmt", _ALL_FORMATS)
class TestNms3dNoSuppressionAcrossGaps:
    def test_chain_of_overlapping_pairs(self, fmt: BoundingBox3DFormat) -> None:
        # Three boxes in a line: A overlaps B, B overlaps C, but A does
        # not overlap C. Greedy picks the highest-score of each
        # overlapping pair. Scores: A=0.9, B=0.8, C=0.7.
        # A and B overlap (shift 1.0, IoU ≈ 0.33 > 0.3).
        # B and C overlap (same).
        # A and C are ~2 apart, zero IoU.
        # Expected after NMS @ 0.3: keep A (highest, suppresses B);
        # C survives because its only suppressor B is gone but the rule
        # is greedy by score, so A is kept, B is suppressed by A,
        # C is kept since it does not overlap A.
        boxes = _stack(
            [
                box_at(0.0, 0, fmt=fmt),
                box_at(1.0, 0, fmt=fmt),
                box_at(2.0, 0, fmt=fmt),
            ]
        )
        scores = torch.tensor([0.9, 0.8, 0.7])
        kept = nms_3d(boxes, scores, 0.3, fmt)
        assert sorted(kept.tolist()) == [0, 2]


@pytest.mark.parametrize("fmt", _ALL_FORMATS)
class TestBatchedNms3d:
    def test_different_classes_do_not_suppress(self, fmt: BoundingBox3DFormat) -> None:
        # Two overlapping boxes at the same location with different
        # class IDs → both must survive because NMS is per-class.
        boxes = _stack([box_at(0, 0, fmt=fmt), box_at(0, 0, fmt=fmt)])
        scores = torch.tensor([0.9, 0.8])
        idxs = torch.tensor([0, 1])
        kept = batched_nms_3d(boxes, scores, idxs, 0.5, fmt)
        assert sorted(kept.tolist()) == [0, 1]

    def test_same_class_suppresses(self, fmt: BoundingBox3DFormat) -> None:
        boxes = _stack([box_at(0, 0, fmt=fmt), box_at(0, 0, fmt=fmt)])
        scores = torch.tensor([0.9, 0.8])
        idxs = torch.tensor([0, 0])
        kept = batched_nms_3d(boxes, scores, idxs, 0.5, fmt)
        assert kept.tolist() == [0]

    def test_mixed_classes_independent_nms(self, fmt: BoundingBox3DFormat) -> None:
        # Class 0: two overlapping boxes (one should be suppressed).
        # Class 1: two overlapping boxes (one should be suppressed).
        # Class 2: one box (kept).
        boxes = _stack(
            [
                box_at(0, 0, fmt=fmt),
                box_at(0, 0, fmt=fmt),
                box_at(10, 0, fmt=fmt),
                box_at(10, 0, fmt=fmt),
                box_at(20, 0, fmt=fmt),
            ]
        )
        scores = torch.tensor([0.9, 0.8, 0.85, 0.7, 0.6])
        idxs = torch.tensor([0, 0, 1, 1, 2])
        kept = batched_nms_3d(boxes, scores, idxs, 0.5, fmt)
        # Expected survivors: index 0 (cls 0, top score), index 2
        # (cls 1, top score), index 4 (cls 2, only box).
        assert sorted(kept.tolist()) == [0, 2, 4]

    def test_output_sorted_by_descending_score(self, fmt: BoundingBox3DFormat) -> None:
        boxes = _stack(
            [
                box_at(0, 0, fmt=fmt),
                box_at(10, 0, fmt=fmt),
                box_at(20, 0, fmt=fmt),
            ]
        )
        scores = torch.tensor([0.5, 0.9, 0.7])
        idxs = torch.tensor([0, 1, 2])
        kept = batched_nms_3d(boxes, scores, idxs, 0.5, fmt)
        assert kept.tolist() == [1, 2, 0]

    def test_empty_input(self, fmt: BoundingBox3DFormat) -> None:
        boxes = torch.zeros((0, _NUM_COLS[fmt]), dtype=torch.float32)
        scores = torch.zeros(0)
        idxs = torch.zeros(0, dtype=torch.long)
        kept = batched_nms_3d(boxes, scores, idxs, 0.5, fmt)
        assert kept.dtype == torch.long
        assert kept.numel() == 0


class TestNmsConsistencyWithIou:
    """Cross-check: kept boxes must have pairwise IoU <= threshold."""

    def test_no_survivors_exceed_threshold(self) -> None:
        fmt = BoundingBox3DFormat.XYZLWHY
        torch.manual_seed(0)
        n = 20
        cxy = (torch.rand(n, 2) - 0.5) * 10
        cz = torch.zeros(n, 1)
        lwh = torch.rand(n, 3) * 1.5 + 0.5
        yaw = (torch.rand(n, 1) - 0.5) * 2 * 3.14159
        boxes = torch.cat([cxy, cz, lwh, yaw], dim=1)
        scores = torch.rand(n)

        threshold = 0.3
        kept = nms_3d(boxes, scores, threshold, fmt)
        kept_boxes = boxes[kept]
        iou = box3d_iou(kept_boxes, kept_boxes, fmt)
        # Mask the diagonal (self-IoU is 1).
        iou.fill_diagonal_(0.0)
        assert (iou <= threshold + 1e-5).all()
