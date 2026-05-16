"""Sphinx configuration for vision3d documentation."""

import sys
from pathlib import Path

from docutils.nodes import Element
from sphinx.addnodes import pending_xref
from sphinx.application import Sphinx
from sphinx.environment import BuildEnvironment
from sphinx.util.typing import ExtensionMetadata

import vision3d

sys.path.insert(0, str(Path(__file__).parent / "_ext"))

project = "vision3d"
copyright = "2026, Peter Siegel"
author = "Peter Siegel"
release = vision3d.__version__

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_gallery.gen_gallery",
    "gfm_admonitions",
    "rerun_embed",
    "rerun_scraper",
]

autosummary_ignore_module_all = False

sphinx_gallery_conf = {
    "examples_dirs": "../../gallery",
    "gallery_dirs": "auto_examples",
    "remove_config_comments": True,
    "show_signature": False,
    "image_scrapers": ("rerun_scraper",),
    "reset_modules": ("rerun",),
}

autodoc_default_options = {
    "members": True,
    "show-inheritance": True,
}
autodoc_inherit_docstrings = False
autodoc_member_order = "bysource"
autodoc_preserve_defaults = True
autodoc_typehints = "description"

napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_preprocess_types = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_ivar = True

# Private base classes that appear in autodoc's "Bases:" line for our
# public transform subclasses; they aren't documented separately on
# purpose, so suppress the unresolved cross-reference.
nitpick_ignore = [
    ("py:class", "vision3d.transforms._transform._RandomApplyTransform"),
    ("py:class", "vision3d.transforms.v2._Refuse3DAwareMixin"),
    ("py:class", "torchvision.transforms.v2._transform._RandomApplyTransform"),
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "torch": ("https://docs.pytorch.org/docs/stable/", None),
    "torchvision": ("https://docs.pytorch.org/vision/stable/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "rerun": ("https://ref.rerun.io/docs/python/stable/", None),
}

# Loose type names that surface in torchvision-inherited docstrings on the
# vision3d.transforms.v2 wrappers. Suppressed only on v2 autodoc pages.
_TORCHVISION_V2_INHERITED_REF_TARGETS = {
    "sequence",
    "number",
    "float (min",
    "max)",
    "bool,optional",
    "InterpolationMode",
    "torchvision.transforms.InterpolationMode",
    "PIL Image",
    "Tensor",
}


def _suppress_v2_inherited_refs(
    app: Sphinx,
    env: BuildEnvironment,
    node: pending_xref,
    contnode: Element,
) -> Element | None:
    """Mark loose type references resolved on v2 autodoc pages.

    Returns:
        ``contnode`` to short-circuit Sphinx's missing-reference warning
        when the failing reference matches the allowlist; ``None`` to let
        Sphinx handle the reference normally otherwise.
    """
    del app, env
    if node.get("reftype") != "class":
        return None
    if node.get("reftarget") not in _TORCHVISION_V2_INHERITED_REF_TARGETS:
        return None
    if not node.get("refdoc", "").startswith("api/generated/vision3d.transforms.v2"):
        return None
    return contnode


def setup(app: Sphinx) -> ExtensionMetadata:
    """Register the v2-scoped missing-reference handler with Sphinx.

    Returns:
        Extension metadata advertising version and parallel-safety.
    """
    app.connect("missing-reference", _suppress_v2_inherited_refs)
    return {
        "version": "0.1",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }


templates_path = ["_templates"]
html_static_path = ["_static"]

html_theme = "pydata_sphinx_theme"
html_title = f"vision3d {release}"
html_theme_options = {
    "github_url": "https://github.com/yeetypete/vision3d",
    "footer_start": ["copyright"],
    "footer_end": "",
    "primary_sidebar_end": "",
}
