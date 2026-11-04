"""Every test runs with the DEBUG switch: an exception that a route would record in
info["errors"] fails the test instead."""

import os

os.environ.setdefault("SYMIDENT_DEBUG", "1")

import pytest  # noqa: E402

from symident.sym.msolve import MsolveMissingError  # noqa: E402


@pytest.hookimpl(wrapper=True)
def pytest_runtest_call(item):
    """A test that needs msolve, where it is missing, is skipped, not failed."""
    try:
        return (yield)
    except MsolveMissingError:
        pytest.skip("needs msolve")
