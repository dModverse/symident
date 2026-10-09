"""Sphinx configuration of the symident documentation; builds from the sources without
the compiled kernel."""

import os
import sys

import tomllib

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
with open(os.path.join(ROOT, "pyproject.toml"), "rb") as fh:
    release = tomllib.load(fh)["project"]["version"]

project = "symident"
author = "Simon Beyer"
copyright = "2026, Simon Beyer"
version = release
autodoc_mock_imports = ["symident._core"]

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "numpydoc",
    "myst_parser",
]
autosummary_generate = True
numpydoc_show_class_members = False
autodoc_member_order = "bysource"
myst_enable_extensions = ["dollarmath"]
source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
exclude_patterns = ["_build"]

html_theme = "pydata_sphinx_theme"
html_title = f"symident {release}"
html_theme_options = {"navigation_with_keys": False, "show_toc_level": 2}
html_static_path = []
