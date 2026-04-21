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
    "gfm_admonitions",
]

autosummary_ignore_module_all = False

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
}

templates_path = ["_templates"]

html_theme = "pydata_sphinx_theme"
html_title = f"vision3d {release}"
html_theme_options = {
    "github_url": "https://github.com/yeetypete/vision3d",
    "footer_start": ["copyright"],
    "footer_end": "",
    "primary_sidebar_end": "",
}
