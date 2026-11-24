"""Sphinx configuration of the symident documentation; builds from the sources without
the compiled kernel."""

import os
import sys

import tomllib
from pybtex.plugin import register_plugin
from pybtex.style.formatting.unsrt import Style as _Unsrt
from pybtex.style.formatting.unsrt import dashify
from pybtex.style.template import field, href, join, node, optional, tag

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
    "sphinxcontrib.bibtex",
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
html_static_path = ["_static"]
html_css_files = ["custom.css"]
# every page is top level, so the section navigation of the left sidebar stays empty
html_sidebars = {"**": []}
templates_path = ["_templates"]


def _title(doc):
    """The title of the page `doc`, the line above the first underline."""
    with open(os.path.join(ROOT, "docs", doc + ".rst"), encoding="utf-8") as fh:
        lines = [line.rstrip() for line in fh]
    return next(a for a, b in zip(lines, lines[1:]) if a and b and set(b) <= set("=-~#*"))


def _chapters(doc):
    """The pages in the toctree of `doc`, with their titles."""
    with open(os.path.join(ROOT, "docs", doc + ".rst"), encoding="utf-8") as fh:
        text = fh.read().split(".. toctree::", 1)
    if len(text) < 2:
        return []
    entries = [line.strip() for line in text[1].split("\n\n", 2)[1].splitlines()]
    return [(e, _title(e)) for e in entries if e]


# the header lists the pages of the root toctree; the guide and the method also open a menu of their chapters
html_context = {"symident_nav": [(d, _title(d), _chapters(d) if d in ("guide", "method") else [])
                                 for d in ("install", "quickstart", "guide", "method", "api")]
                + [("changelog", "Changelog", [])]}

# citations: numbered in the order cited, in the reference format of IEEE, as on the dMod2 and cppDE sites


def _pages():
    return field("pages", apply_func=dashify)


@node
def _authors(children, context, role="author"):
    """The authors, the first one with et al. past six names or with a trailing others."""
    persons = context["entry"].persons[role]
    style = context["style"]
    names = [style.format_name(p, style.abbreviate_names) for p in persons if str(p) != "others"]
    if len(persons) > 6 or str(persons[-1]) == "others":
        return join[names[0], " et al."].format_data(context)
    return join(sep=", ", sep2=" and ", last_sep=", and ")[names].format_data(context)


class _IEEE(_Unsrt):
    default_sorting_style = "none"
    default_label_style = "number"

    def __init__(self, **kwargs):
        kwargs.setdefault("abbreviate_names", True)
        super().__init__(**kwargs)

    def _title(self):
        return join["\u201c", field("title"), ",\u201d "]

    def _series(self):
        return optional[", ser. ", field("series"), optional[", vol. ", field("volume")]]

    def _doi(self):
        doi = field("doi", raw=True)
        return optional[", doi: ", href[join["https://doi.org/", doi], doi]]

    def get_article_template(self, e):
        return join[
            _authors(), ", ", self._title(), tag("em")[field("journal")],
            optional[", vol. ", field("volume")], optional[", no. ", field("number")],
            optional[", pp. ", _pages()], optional[", Art. no. ", field("eid")], ", ", field("year"),
            self._doi(), "."]

    def get_inproceedings_template(self, e):
        return join[
            _authors(), ", ", self._title(), "in ", tag("em")[field("booktitle")], self._series(),
            ", ", field("address"), ": ", field("publisher"), ", ", field("year"),
            optional[", pp. ", _pages()], self._doi(), "."]

    def get_book_template(self, e):
        return join[
            _authors(), ", ", tag("em")[field("title")], optional[", ", field("edition"), " ed."],
            self._series(), ". ", field("address"), ": ", field("publisher"), ", ", field("year"),
            self._doi(), "."]


register_plugin("pybtex.style.formatting", "ieee", _IEEE)
bibtex_bibfiles = ["refs.bib"]
bibtex_default_style = "ieee"
bibtex_reference_style = "label"
