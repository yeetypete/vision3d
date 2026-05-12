"""Sphinx configuration for vision3d documentation."""

import sys
from pathlib import Path

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

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "torch": ("https://docs.pytorch.org/docs/stable/", None),
    "torchvision": ("https://docs.pytorch.org/vision/stable/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "rerun": ("https://ref.rerun.io/docs/python/stable/", None),
}

# Suppress nitpicky reference errors for type names that torchvision uses
# loosely in its docstrings (e.g. "sequence", "number", "float (min, max)").
# vision3d.transforms.v2 copies the parent class docstring on each wrapped
# transform, so these surface even though they originate upstream.
nitpick_ignore = [
    ("py:class", "sequence"),
    ("py:class", "number"),
    ("py:class", "float (min"),
    ("py:class", "max)"),
    ("py:class", "bool, optional"),
    ("py:class", "bool,optional"),
    ("py:class", "InterpolationMode"),
    ("py:class", "torchvision.transforms.InterpolationMode"),
    ("py:class", "PIL Image"),
    ("py:class", "Tensor"),
]

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
