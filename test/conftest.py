"""Shared fixtures for vision3d tests."""

from collections.abc import Generator

import pytest
import torch
from torchvision import tv_tensors

_DEVICES = ("cpu", "cuda")


def pytest_configure(config: pytest.Config) -> None:
    for device in _DEVICES:
        config.addinivalue_line(
            "markers", f"{device}: test variant running on device '{device.upper()}'."
        )
    config.addinivalue_line(
        "markers",
        "skip_device(*names): skip the test variants running on the listed devices.",
    )
    config.addinivalue_line(
        "markers",
        "only_device(*names): only run the test variants on the listed devices.",
    )


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Drop test variants excluded by a ``skip_device`` or ``only_device`` marker.

    An ``only_device`` marker takes precedence over ``skip_device`` markers.

    Args:
        items: List of collected test items; modified in place.

    Raises:
        UsageError: If a ``skip_device`` or ``only_device`` marker references a
            device not in ``_DEVICES``.
    """
    kept: list[pytest.Item] = []
    for item in items:
        for name in ("skip_device", "only_device"):
            for mark in item.iter_markers(name):
                unknown = set(mark.args) - set(_DEVICES)
                if unknown:
                    raise pytest.UsageError(
                        f"{item.nodeid}: {name} got unknown devices "
                        f"{sorted(unknown)} (known: {sorted(_DEVICES)})"
                    )
        callspec = getattr(item, "callspec", None)
        device = callspec.params.get("device") if callspec is not None else None
        if device is not None and _excludes_device(item, device):
            continue
        kept.append(item)
    items[:] = kept


def _excludes_device(item: pytest.Item, device: str) -> bool:
    only = item.get_closest_marker("only_device")
    if only is not None:
        return device not in only.args
    return any(device in mark.args for mark in item.iter_markers("skip_device"))


@pytest.fixture(
    params=[pytest.param(d, marks=getattr(pytest.mark, d)) for d in _DEVICES],
    autouse=True,
)
def device(request: pytest.FixtureRequest) -> Generator[torch.device]:
    """Run every test once per supported device.

    This fixture activates a ``torch.device`` context so that any tensor
    construction that omits an explicit ``device=`` argument (e.g.
    ``torch.rand(3)``, ``torch.tensor([1.0])``) runs on the parametrized
    device. Tests that want to pin a device explicitly are free to do so and
    the override will win.

    Each parametrization carries a matching marker (``cpu``, ``cuda``, ...)
    so variants can be filtered from the CLI. Mark a test with
    ``@pytest.mark.skip_device("cuda")`` to drop the listed devices' variants,
    or ``@pytest.mark.only_device("cuda")`` to keep only those variants. See
    :func:`pytest_collection_modifyitems`.

    Example:
        Filter test runs by device from the command line::

            pytest -m cpu              # only CPU variants
            pytest -m cuda             # only CUDA variants
            pytest -m "not cuda"       # everything except CUDA

    Args:
        request: The pytest fixture request, used to read the parametrized
            device.

    Yields:
        The active :class:`torch.device` for the test invocation.
    """
    device = torch.device(request.param)
    with device:
        yield device


@pytest.fixture(autouse=True)
def restore_tensor_return_type() -> Generator[None]:
    """Reset torchvision's TVTensor return-type to ``"Tensor"`` after each test.

    Guards against leakage from tests that use
    :func:`torchvision.tv_tensors.set_return_type` and would otherwise
    contaminate later tests with a non-default global.
    """
    yield
    tv_tensors.set_return_type("Tensor")
