"""Shared fixtures for vision3d tests."""

from collections.abc import Generator

import pytest
import torch

_DEVICES = ("cpu", "cuda")


def pytest_configure(config: pytest.Config) -> None:
    for dev in _DEVICES:
        config.addinivalue_line(
            "markers", f"{dev}: test variant running on device '{dev.upper()}'."
        )
    config.addinivalue_line(
        "markers",
        "skip_device(*names): skip the test variants running on the listed devices.",
    )


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Drop test variants whose device matches a ``skip_device`` marker.

    Args:
        items: List of collected test items; modified in place.

    Raises:
        UsageError: If a ``skip_device`` marker references a device not in
            ``_DEVICES``.
    """
    kept: list[pytest.Item] = []
    for item in items:
        for mark in item.iter_markers("skip_device"):
            unknown = set(mark.args) - set(_DEVICES)
            if unknown:
                raise pytest.UsageError(
                    f"{item.nodeid}: skip_device got unknown devices "
                    f"{sorted(unknown)} (known: {sorted(_DEVICES)})"
                )
        callspec = getattr(item, "callspec", None)
        device = callspec.params.get("device") if callspec is not None else None
        if device is not None and any(
            device in mark.args for mark in item.iter_markers("skip_device")
        ):
            continue
        kept.append(item)
    items[:] = kept


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
    so variants can be filtered from the CLI. Tests decorated with
    ``@pytest.mark.skip_device("cuda")`` are dropped from collection for the
    named device variant(s); see :func:`pytest_collection_modifyitems`.

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
    dev = torch.device(request.param)
    with dev:
        yield dev
