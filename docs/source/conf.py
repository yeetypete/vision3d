"""Sphinx configuration for vision3d documentation."""

import vision3d

project = "vision3d"
copyright = "2026, Peter Siegel"
author = "Peter Siegel"
release = vision3d.__version__

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

autosummary_generate = True
napoleon_google_docstring = True
napoleon_numpy_docstring = False

templates_path = ["_templates"]

html_theme = "furo"
html_static_path = ["_static"]
