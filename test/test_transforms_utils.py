"""Tests for the shared transform helpers in vision3d.transforms._utils."""

from typing import Any

import pytest
import torch
from torchvision.transforms.v2._utils import (
    _find_labels_default_heuristic as torchvision_getter,
)

from vision3d.transforms._utils import _default_labels_getter

# Samples covering every branch of the default heuristic.
_PARITY_CASES: dict[str, Any] = {
    "exact_key": {"labels": torch.tensor([0, 1])},
    "nested_dict_not_recursed": {"targets": {"labels": torch.tensor([0, 1])}},
    "substring_key": {"gt_labels": torch.tensor([0, 1])},
    "substring_anywhere": {"FooLaBeLBar": torch.tensor([0, 1])},
    "exact_beats_substring": {
        "gt_labels": torch.tensor([0, 1]),
        "labels": torch.tensor([2, 3]),
    },
    "sequence_bare_tensor": ({"x": 1}, torch.tensor([0, 1])),
    "sequence_of_three": ({"x": 1}, {"labels": torch.tensor([0, 1])}, {"z": 2}),
    "value_is_tensor_list": {"labels": [torch.tensor([0]), torch.tensor([1])]},
    "value_is_not_a_tensor": {"labels": ["a", "b"]},
    "no_label_key": {"boxes": torch.tensor([0, 1])},
    "not_a_mapping": 42,
}


class TestDefaultGetterTorchvisionParity:
    """The default getter must resolve labels exactly as torchvision does."""

    @pytest.mark.parametrize("sample", _PARITY_CASES.values(), ids=list(_PARITY_CASES))
    @pytest.mark.skip_device("cuda")
    def test_matches_torchvision(self, sample: Any) -> None:
        def outcome(getter: Any) -> Any:
            try:
                return getter(sample)
            except Exception as e:  # noqa: BLE001
                return type(e)

        expected = outcome(torchvision_getter)
        actual = outcome(_default_labels_getter)
        assert actual is expected, (
            f"diverged from torchvision: it gave {expected!r}, we gave {actual!r}"
        )
