"""Sphinx-gallery integration for rerun recordings.

Wires :func:`rerun_scraper` into :mod:`sphinx_gallery`'s ``image_scrapers``
extension point so gallery scripts that log to rerun get their recording
captured, saved to ``docs/source/_static/<app_id>.rrd``, and embedded
automatically. Scripts need only::

    import rerun as rr

    rr.init("<app_id>", spawn=True)
    # ... log ...

The interactive Rerun viewer renders inline below the captured output of
the cell that first calls ``rr.init``.
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import rerun as rr
import sphinx_gallery.gen_rst as _sg_gen_rst
from sphinx.application import Sphinx
from sphinx.util.typing import ExtensionMetadata
from sphinx_gallery.gen_rst import GalleryConfig
from sphinx_gallery.py_source_parser import Block

# Track (script, app_id) pairs we've already saved+embedded this build.
# rr.save is incremental, so we deliberately save each recording exactly
# once: at the end of the cell that first calls rr.init for that app_id.
_embedded: set[tuple[str, str]] = set()

# Whether the current gallery script has called ``rr.init``. Rerun's
# global recording state otherwise persists across scripts in the same
# build, which would falsely signal "this script logged to rerun" for
# every cell after the first script's ``rr.init`` call.
_init_called: bool = False

# Capture the true ``rr.init`` once at module load and define a single
# wrapper that always forces ``spawn=False`` to avoid spawning the rerun
# viewer during sphinx-gallery builds. ``_reset_rerun_init`` rebinds
# ``rr.init`` to this wrapper before each gallery script.
_original_rr_init = rr.init


def _patched_rr_init(*args: Any, spawn: bool = False, **kwargs: Any) -> Any:
    global _init_called
    _init_called = True
    return _original_rr_init(*args, spawn=False, **kwargs)


def rerun_scraper(
    block: Block,
    block_vars: dict[str, Any],
    gallery_conf: GalleryConfig,
) -> str:
    """Save the script's rerun recording and emit the embed directive.

    Fires after each gallery cell. Saves and emits an embed exactly once
    per ``(script, app_id)`` pair, at the end of the cell that first
    activates that recording. Subsequent cells touching the same
    ``app_id`` are no-ops.

    Args:
        block: Sphinx-gallery code block.
        block_vars: Per-block context. Uses ``src_file``.
        gallery_conf: Sphinx-gallery configuration. Uses ``src_dir``.

    Returns:
        A ``.. rerun-embed::`` directive the first time a given
        ``(script, app_id)`` is seen, empty string otherwise.
    """
    if not _init_called:
        return ""

    app_id = rr.get_application_id()
    if not app_id:
        return ""

    src_file = block_vars["src_file"]
    key = (src_file, app_id)
    if key in _embedded:
        return ""

    static_dir = Path(gallery_conf["src_dir"]) / "_static"
    static_dir.mkdir(parents=True, exist_ok=True)
    out_path = static_dir / f"{app_id}.rrd"
    rr.save(str(out_path))
    _embedded.add(key)

    return f"\n.. rerun-embed:: {app_id}.rrd\n   :height: 60vh\n"


def _reset_rerun_init(gallery_conf: GalleryConfig, fname: str | None) -> None:
    """Rebind ``rr.init`` and clear the per-script init flag."""
    global _init_called
    rr.init = _patched_rr_init
    _init_called = False


def _get_sg_image_scraper() -> Callable[[Block, dict[str, Any], GalleryConfig], str]:
    """Sphinx-gallery scraper factory.

    Invoked by sphinx-gallery via the string form ``"rerun_scraper"`` in
    ``sphinx_gallery_conf["image_scrapers"]`` (see ``docs/source/conf.py``).
    Lets the conf reference this module by name instead of holding a function
    reference, which keeps it pickleable for Sphinx's config cache.

    Returns:
        The :func:`rerun_scraper` callable that sphinx-gallery will invoke
        after each script cell.
    """
    return rerun_scraper


def setup(app: Sphinx) -> ExtensionMetadata:
    """Register the ``"rerun"`` reset alias and clear per-build state.

    Args:
        app: The Sphinx application.

    Returns:
        Extension metadata advertising parallel-safety.
    """
    _sg_gen_rst._reset_dict["rerun"] = _reset_rerun_init
    _embedded.clear()
    return {
        "version": "0.1",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
