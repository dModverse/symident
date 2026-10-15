"""Expected failures of a route, and the record of every other exception a route absorbs."""

import sys
from tokenize import TokenError

from sympy.polys.polyerrors import BasePolynomialError

from . import session, state


class RouteError(Exception):
    """A route that does not apply or does not close; the message says why."""


class InapplicableError(TypeError, ValueError):
    """An expression a helper does not handle; caught where TypeError or ValueError is."""


# failures of SymPy on a valid expression: no polynomial form, no solution method, a
# value that is not a number, a pole, an expression too deep to traverse or to parse
SYMPY_FAILS = (BasePolynomialError, NotImplementedError, TypeError, ValueError, ArithmeticError,
               RecursionError, SyntaxError, TokenError)


def swallow(exc, where):
    """Record `exc`, which the route at `where` turns into a failure, in
    info["errors"]; with the DEBUG switch, or for a missing msolve or a prime rerun, raise
    it instead."""
    from .gfp import NeedPrimesError
    from .sym.msolve import MsolveMissingError
    if state.env("DEBUG") or isinstance(exc, (NeedPrimesError, MsolveMissingError)):
        raise exc
    session.note("errors", f"{where}: {type(exc).__name__}: {str(exc)[:200]}")


# records beyond this many are dropped; summary() says how many it shows
ERRORS_KEPT = 200


def sympy_fail(exc):
    """Record a failure of SymPy that the caller turns into a failed route; with the DEBUG
    switch, raise a TypeError or ValueError that SymPy did not raise."""
    tb, in_sympy = exc.__traceback__, False
    while tb is not None:
        in_sympy = in_sympy or "/sympy/" in tb.tb_frame.f_code.co_filename.replace("\\", "/")
        tb = tb.tb_next
    if state.env("DEBUG") and isinstance(exc, (TypeError, ValueError)) and not isinstance(exc, InapplicableError) \
            and not in_sympy:
        raise exc
    s = state.get("session")
    if s is None or len(s.notes["errors"]) >= ERRORS_KEPT:
        return
    f = sys._getframe(1)
    where = f"{f.f_globals.get('__name__', '?').removeprefix('symident.')}.{f.f_code.co_name}"
    session.note("errors", f"{where}: {type(exc).__name__}: {str(exc)[:120]}")
