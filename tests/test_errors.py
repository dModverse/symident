"""SymPy failures that a route turns into a failure are recorded; other errors raise under DEBUG."""

import pytest
import sympy as spy

import symident as si
from symident import session
from symident.errors import InapplicableError, sympy_fail


def _route(exc):
    try:
        raise exc
    except (TypeError, ValueError) as e:
        sympy_fail(e)


def test_a_sympy_failure_is_recorded_and_counted():
    with session.opened({}) as ses, si.switches(DEBUG=False):
        try:
            spy.Poly(spy.sin(spy.Symbol("x")), spy.Symbol("x"), domain="ZZ")
        except Exception as e:
            sympy_fail(e)
        _route(InapplicableError("not handled"))
    assert len(ses.notes["errors"]) == 2
    assert ses.notes["errors"][1] == "test_errors._route: InapplicableError: not handled"


def test_debug_raises_a_type_error_that_sympy_did_not_raise():
    with session.opened({}), si.switches(DEBUG=True):
        with pytest.raises(TypeError):
            _route(TypeError("a bug"))
        _route(InapplicableError("not handled"))


def test_the_summary_counts_the_errors():
    from symident import report
    res = {"method": "observability", "identifiable": False, "rank": 1, "dim": 2, "symmetries": [],
           "info": {"engine": "modular", "errors": ["a: TypeError: x", "b: ValueError: y"]}}
    assert "  errors: 2 exceptions turned into failed routes (info['errors'])" in report.summary_lines(res, width=80)
