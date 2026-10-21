"""The number of parameter values with the output of a generic one: a count of discrete symmetries."""

import os
import subprocess
import tempfile

import sympy as spy

from .. import session
from ..errors import SYMPY_FAILS, sympy_fail
from ..sym.modp import _modp_rational
from ..sym.parse import model_sympify


def _lie(h, F):
    return spy.together(sum((spy.diff(h, x) * fx for x, fx in F.items()), spy.Integer(0)))


def output_coefficients(f, g, ic, order):
    """The Taylor coefficients of the observables at the start, up to `order`, as rational functions."""
    F = {spy.Symbol(k): model_sympify(v) for k, v in f.items()}
    x0 = {spy.Symbol(k): model_sympify(v) for k, v in ic.items()}
    out = []
    for h in g.values():
        e = model_sympify(h)
        for _ in range(order + 1):
            out.append(spy.cancel(e.xreplace(x0)))
            e = _lie(e, F)
    return out


def _msolve_degree(eqs, names, p, timeout):
    """(dimension, degree) of the ideal of `eqs` over GF(p) from msolve, or None on failure."""
    import ast

    from ..sym.msolve import _msolve_binary
    with tempfile.TemporaryDirectory() as td:
        fi, fo = os.path.join(td, "in.ms"), os.path.join(td, "out.ms")
        with open(fi, "w") as fh:
            fh.write(", ".join(names) + "\n" + str(p) + "\n" + ",\n".join(eqs) + "\n")
        try:
            r = subprocess.run([_msolve_binary(), "-f", fi, "-o", fo, "-P", "1"], capture_output=True,
                               text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return None
        txt = open(fo).read() if os.path.exists(fo) else ""
    if r.returncode != 0 or not txt.strip():
        return None
    data = ast.literal_eval(txt.strip().rstrip(":"))
    if data[0] != 0:
        return (data[0], None)
    return (0, int(data[1][2]))


def solution_count(f, g, ic, coords, order, p=1000003, timeout=600):
    """{"count", "order", "prime"} of points over GF(p) sharing the output coefficients of a random one, or None."""
    try:
        eqs, names = _fiber_equations(f, g, ic, coords, order, p)
    except SYMPY_FAILS as exc:
        sympy_fail(exc)
        return None
    if eqs is None:
        return None
    out = _msolve_degree(eqs, names, p, timeout)
    if out is None or out[1] is None:
        return None
    return {"count": out[1], "order": int(order), "prime": int(p)}


def _fiber_equations(f, g, ic, coords, order, p):
    """(equations, variables) over GF(p) of the points with the output coefficients of a random one."""
    coeffs = output_coefficients(f, g, ic, order)
    syms = [spy.Symbol(c) for c in coords]
    rng = session.current().rng("fiber", tuple(coords))
    star = {s: spy.Integer(rng.randrange(2, p)) for s in syms}
    # the known symbols take one random value in the point and in the solutions
    known = {s: spy.Integer(rng.randrange(2, p)) for c in coeffs for s in c.free_symbols if s not in star}
    names = [f"x{i}" for i in range(len(syms))]
    gens = [spy.Symbol(n) for n in names]
    ren = dict(zip(syms, gens))
    eqs, dens = [], []
    for c in coeffs:
        c = c.xreplace(known)
        num, den = spy.fraction(spy.cancel(c))
        cv = c.xreplace(star)
        if not cv.is_Rational:
            return None, None
        v = _modp_rational(cv, p)
        P = spy.Poly(spy.expand(num.xreplace(ren)) - v * spy.expand(den.xreplace(ren)), *gens)
        P = P.clear_denoms()[1].set_modulus(p)
        if not P.is_zero:
            eqs.append(str(P.as_expr()).replace("**", "^"))
        if den.free_symbols:
            D = spy.Poly(spy.expand(den.xreplace(ren)), *gens).clear_denoms()[1].set_modulus(p)
            dens.append(D.as_expr())
    # points where a denominator or a coordinate vanishes do not count
    w = "si_w"
    sat = spy.expand(spy.Mul(*dens, *[spy.Symbol(n) for n in names]))
    eqs.append(f"{w}*({str(sat).replace('**', '^')})-1")
    return eqs, names + [w]

