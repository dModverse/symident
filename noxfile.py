#!/usr/bin/env -S uv run --script
# /// script
# dependencies = ["nox>=2025.2.9"]
# ///
"""Development sessions: `nox -s lint`, `nox -s tests`, `nox -s docs`, `nox -s build`."""

import shutil
from pathlib import Path

import nox

nox.needs_version = ">=2025.2.9"
nox.options.default_venv_backend = "uv|virtualenv"

DIR = Path(__file__).parent.resolve()
PYPROJECT = nox.project.load_toml(DIR / "pyproject.toml")


@nox.session(default=True)
def lint(session: nox.Session) -> None:
    """The pre-commit hooks on all files."""
    session.install("pre-commit")
    session.run("pre-commit", "run", "--all-files", "--show-diff-on-failure", *session.posargs)


@nox.session(default=True)
def tests(session: nox.Session) -> None:
    """The test suite against a source build; arguments go to pytest."""
    session.install(str(DIR), *nox.project.dependency_groups(PYPROJECT, "test"))
    # outside the checkout, so that the tests import the installed package
    session.chdir(session.create_tmp())
    session.run("pytest", "-n", "auto", str(DIR / "tests"), *session.posargs)


@nox.session(reuse_venv=True, default=False)
def docs(session: nox.Session) -> None:
    """The documentation as HTML in docs/_build/html, warnings as errors."""
    session.install(str(DIR), *nox.project.dependency_groups(PYPROJECT, "docs"))
    session.run("sphinx-build", "-W", "-b", "html", str(DIR / "docs"), str(DIR / "docs/_build/html"), *session.posargs)


@nox.session(default=False)
def build(session: nox.Session) -> None:
    """The sdist and the wheel in dist/."""
    shutil.rmtree(DIR / "dist", ignore_errors=True)
    session.install("build")
    session.run("python", "-m", "build", str(DIR), "--outdir", str(DIR / "dist"), *session.posargs)


if __name__ == "__main__":
    nox.main()
